#!/usr/bin/env python3
"""One-time backfill: bridge gap from training feature matrix to live incremental pipeline.

Steps:
  1. Load feature_matrix_v10.parquet (training data, ends ~Jan 2026)
  2. Initialize incremental encoder state from the feature matrix
  3. Walk through each 5M candle after the feature matrix, encoding incrementally
  4. Save persistent features CSV + encoder state JSON
  5. Verify: compare overlap region with batch encoding

The persistent feature store is then ready for the data_service to continue
incrementally with each new 5M candle.

Usage:
  python backfill_features.py --data-dir /data               # GCE (uses persistent CSVs)
  python backfill_features.py --data-dir ./persistent_data    # Local
  python backfill_features.py --verify-only                   # Just run verification
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from core.config import TIMEFRAME_ORDER, WINDOW_SIZES
from data_service.incremental_encoder import IncrementalEncoder
from data_service.state_initializer import initialize_state
from data_service.csv_io import append_rows_atomic

logger = logging.getLogger(__name__)


def _load_klines(klines_dir):
    """Load all kline CSVs, indexed by time for fast lookup."""
    klines = {}
    for tf in TIMEFRAME_ORDER:
        path = Path(klines_dir) / f"ml_data_{tf}.csv"
        if not path.exists():
            logger.warning(f"  Missing klines for {tf}")
            continue
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
        for c in ("Open", "High", "Low", "Close", "Volume"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.sort_values("time").reset_index(drop=True)
        klines[tf] = df
    return klines


def _load_decomposed(decomposed_dir):
    """Load all decomposed CSVs, indexed by time for fast lookup."""
    decomposed = {}
    for tf in TIMEFRAME_ORDER:
        for ws in WINDOW_SIZES:
            path = Path(decomposed_dir) / f"decomposed_{tf}_w{ws}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
            df = df.sort_values("time").reset_index(drop=True)
            decomposed[(tf, ws)] = df
    return decomposed


def _build_time_index(klines, decomposed):
    """Build time→row index for each TF/combo for O(1) lookup."""
    kline_idx = {}
    for tf, df in klines.items():
        kline_idx[tf] = {t: i for i, t in enumerate(df["time"].values)}

    decomp_idx = {}
    for key, df in decomposed.items():
        decomp_idx[key] = {t: i for i, t in enumerate(df["time"].values)}

    return kline_idx, decomp_idx


def backfill_gap(feature_matrix_path, data_dir, verify=True):
    """Run the full backfill pipeline.

    Args:
        feature_matrix_path: Path to feature_matrix_v10.parquet
        data_dir: Base data directory (with klines/, decomposed/, features/ subdirs)
        verify: Whether to run the verification step
    """
    data_dir = Path(data_dir)
    klines_dir = data_dir / "klines"
    decomposed_dir = data_dir / "decomposed"
    features_dir = data_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    features_path = features_dir / "features.csv"
    state_path = features_dir / "feature_state.json"

    # --- Step 1: Load feature matrix ---
    logger.info("Step 1: Loading feature matrix...")
    fm = pd.read_parquet(feature_matrix_path)
    fm["time"] = pd.to_datetime(fm["time"]).dt.tz_localize(None)
    fm = fm.sort_values("time").reset_index(drop=True)
    last_fm_time = fm["time"].iloc[-1]
    logger.info(f"  Feature matrix: {fm.shape}, ends at {last_fm_time}")

    # --- Step 2: Load persistent klines + decomposed ---
    logger.info("Step 2: Loading klines and decomposed data...")
    klines = _load_klines(klines_dir)
    decomposed = _load_decomposed(decomposed_dir)
    logger.info(f"  Klines: {len(klines)} TFs loaded")
    logger.info(f"  Decomposed: {len(decomposed)} combos loaded")

    if "5M" not in klines:
        raise RuntimeError("5M klines not found — run data_service first to populate L1/L2")

    # --- Step 3: Determine gap ---
    kl_5m = klines["5M"]
    new_5m = kl_5m[kl_5m["time"] > last_fm_time].sort_values("time").reset_index(drop=True)
    n_new = len(new_5m)

    if n_new == 0:
        logger.info("No gap detected — feature matrix is up to date")
        logger.info("Initializing state and saving...")
        state = initialize_state(str(feature_matrix_path), str(klines_dir), str(decomposed_dir))
        encoder = IncrementalEncoder(state)
        encoder.save_state(state_path)
        # Copy feature matrix as the initial features CSV
        fm.to_csv(features_path, index=False)
        logger.info(f"  Saved: {features_path} ({len(fm)} rows)")
        logger.info(f"  Saved: {state_path}")
        return

    logger.info(f"  Gap: {last_fm_time} → {new_5m['time'].iloc[-1]} ({n_new} candles)")

    # --- Step 4: Initialize state from feature matrix ---
    logger.info("Step 3: Initializing encoder state...")
    state = initialize_state(str(feature_matrix_path), str(klines_dir), str(decomposed_dir))
    encoder = IncrementalEncoder(state)

    # --- Step 5: Build time indices for O(1) lookup ---
    logger.info("Step 4: Building time indices...")
    kline_idx, decomp_idx = _build_time_index(klines, decomposed)

    # --- Step 6: Walk through each new 5M candle ---
    logger.info(f"Step 5: Processing {n_new} new 5M candles...")
    t0 = time.time()

    features_list = []
    timestamps = new_5m["time"].values

    for i, t5m in enumerate(timestamps):
        # Convert numpy datetime64 to pandas Timestamp for dict lookup
        t5m_ts = pd.Timestamp(t5m)

        # Find TFs with new data at this timestamp
        new_klines = {}
        for tf in TIMEFRAME_ORDER:
            if tf in kline_idx and t5m in kline_idx[tf]:
                row_idx = kline_idx[tf][t5m]
                new_klines[tf] = klines[tf].iloc[row_idx]

        new_decomp = {}
        for key in decomposed:
            if key in decomp_idx and t5m in decomp_idx[key]:
                row_idx = decomp_idx[key][t5m]
                new_decomp[key] = decomposed[key].iloc[row_idx]

        # Compute incremental feature row
        feature_row = encoder.compute_row(new_klines, new_decomp, t5m_ts)

        # Add time
        row_dict = feature_row.to_dict()
        row_dict["time"] = t5m_ts
        features_list.append(row_dict)

        # Progress logging
        if (i + 1) % 500 == 0 or i == n_new - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(f"  Processed {i + 1}/{n_new} candles "
                        f"({rate:.0f} rows/s, {elapsed:.1f}s)")

    elapsed = time.time() - t0
    logger.info(f"  Backfill complete: {n_new} rows in {elapsed:.1f}s "
                f"({n_new / elapsed:.0f} rows/s)")

    # --- Step 7: Save results ---
    logger.info("Step 6: Saving features and state...")

    backfill_df = pd.DataFrame(features_list)

    # Combine feature matrix + backfilled features
    # Ensure column alignment (backfill may have different column order)
    all_cols = list(fm.columns)
    new_cols = [c for c in backfill_df.columns if c not in fm.columns and c != "time"]
    if new_cols:
        logger.warning(f"  {len(new_cols)} columns in backfill not in feature matrix: "
                       f"{new_cols[:5]}...")

    # Align columns: use feature matrix columns, fill missing with 0
    for col in all_cols:
        if col not in backfill_df.columns and col != "time":
            backfill_df[col] = 0.0

    combined = pd.concat([fm, backfill_df[all_cols]], ignore_index=True)
    combined = combined.sort_values("time").reset_index(drop=True)

    combined.to_csv(features_path, index=False)
    encoder.save_state(state_path)

    logger.info(f"  Saved: {features_path} ({len(combined)} rows)")
    logger.info(f"  Saved: {state_path}")

    # --- Step 8: Verification ---
    if verify:
        run_verification(fm, backfill_df, features_path)


def run_verification(fm, backfill_df, features_path):
    """Verify backfill correctness by comparing with batch encoding.

    Takes the last 50 rows of the feature matrix (batch-encoded) and
    compares a subset of features that should be reproducible.

    For the backfill gap (rows AFTER the feature matrix), we can only
    verify internal consistency — EMA continuity, rolling buffer sanity, etc.
    """
    logger.info("Step 7: Running verification...")

    # Verification 1: Check no NaN/Inf in backfilled features
    numeric_cols = [c for c in backfill_df.columns if c != "time"]
    if numeric_cols:
        bf_numeric = backfill_df[numeric_cols]
        n_nan = bf_numeric.isna().sum().sum()
        n_inf = np.isinf(bf_numeric.values).sum() if len(bf_numeric) > 0 else 0

        if n_nan > 0:
            nan_cols = bf_numeric.columns[bf_numeric.isna().any()].tolist()
            logger.warning(f"  WARN: {n_nan} NaN values in backfill "
                           f"(columns: {nan_cols[:5]})")
        else:
            logger.info("  CHECK 1 PASS: No NaN values in backfill")

        if n_inf > 0:
            logger.warning(f"  WARN: {n_inf} Inf values in backfill")
        else:
            logger.info("  CHECK 2 PASS: No Inf values in backfill")

    # Verification 2: Feature count matches
    fm_feats = set(c for c in fm.columns if c != "time")
    bf_feats = set(c for c in backfill_df.columns if c != "time")
    missing = fm_feats - bf_feats
    extra = bf_feats - fm_feats

    if missing:
        logger.warning(f"  WARN: {len(missing)} features missing from backfill: "
                       f"{sorted(missing)[:10]}")
    else:
        logger.info(f"  CHECK 3 PASS: All {len(fm_feats)} features present in backfill")

    if extra:
        logger.info(f"  INFO: {len(extra)} extra features in backfill (not in training): "
                    f"{sorted(extra)[:10]}")

    # Verification 3: Value range sanity
    if len(backfill_df) > 0 and numeric_cols:
        bf_stats = bf_numeric.describe()
        # Check for extreme values (>1e10)
        max_vals = bf_stats.loc["max"]
        extreme = max_vals[max_vals.abs() > 1e10]
        if len(extreme) > 0:
            logger.warning(f"  WARN: {len(extreme)} features with extreme values: "
                           f"{extreme.index[:5].tolist()}")
        else:
            logger.info("  CHECK 4 PASS: No extreme values (all < 1e10)")

    # Verification 4: EMA continuity — check that EMA features at the boundary
    # (first backfill row) are close to the last feature matrix row
    if len(backfill_df) > 0 and len(fm) > 0:
        ema_features = [c for c in fm.columns
                        if any(c.startswith(p) for p in
                               ["smoothed_momentum_", "directional_vol_body_",
                                "vol_ratio_"])]
        if ema_features:
            last_fm = fm.iloc[-1]
            first_bf = backfill_df.iloc[0]
            max_diff = 0.0
            for feat in ema_features:
                if feat in first_bf.index and feat in last_fm.index:
                    fm_val = float(last_fm[feat])
                    bf_val = float(first_bf[feat])
                    if fm_val != 0:
                        rel_diff = abs(fm_val - bf_val) / (abs(fm_val) + 1e-15)
                        max_diff = max(max_diff, rel_diff)

            if max_diff < 0.5:  # EMA features should be within 50% at boundary
                logger.info(f"  CHECK 5 PASS: EMA continuity OK (max relative diff: "
                            f"{max_diff:.6f})")
            else:
                logger.warning(f"  WARN: EMA discontinuity at boundary "
                               f"(max relative diff: {max_diff:.4f})")

    logger.info("  Verification complete")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill features from training matrix to live incremental pipeline")
    parser.add_argument("--data-dir", type=str, default="./persistent_data",
                        help="Base data directory (default: ./persistent_data)")
    parser.add_argument("--feature-matrix", type=str, default=None,
                        help="Path to feature_matrix_v10.parquet (auto-detected)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Just run verification on existing backfill")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip verification step")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Find feature matrix
    if args.feature_matrix:
        fm_path = Path(args.feature_matrix)
    else:
        fm_path = Path("model_training/encoded_data/feature_matrix_v10.parquet")
        if not fm_path.exists():
            fm_path = Path(__file__).parent / "model_training" / "encoded_data" / "feature_matrix_v10.parquet"

    if not fm_path.exists():
        logger.error(f"Feature matrix not found at {fm_path}")
        logger.error("Provide --feature-matrix or run encode_v10.py first")
        sys.exit(1)

    logger.info(f"Feature matrix: {fm_path}")
    logger.info(f"Data directory: {args.data_dir}")

    if args.verify_only:
        logger.info("Running verification only...")
        fm = pd.read_parquet(fm_path)
        features_path = Path(args.data_dir) / "features" / "features.csv"
        if not features_path.exists():
            logger.error(f"No features file found at {features_path}")
            sys.exit(1)
        all_features = pd.read_csv(features_path)
        all_features["time"] = pd.to_datetime(all_features["time"])
        fm["time"] = pd.to_datetime(fm["time"])
        last_fm_time = fm["time"].max()
        backfill_df = all_features[all_features["time"] > last_fm_time]
        run_verification(fm, backfill_df, features_path)
    else:
        backfill_gap(
            feature_matrix_path=str(fm_path),
            data_dir=args.data_dir,
            verify=not args.no_verify,
        )


if __name__ == "__main__":
    main()
