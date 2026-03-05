"""
Full Pipeline Audit — Local Backfill vs GCE Docker Predictions.

Pulls GCE data, runs local backfill, and compares at 5 stages with hard PASS/FAIL
thresholds to verify both pipelines produce identical (or acceptably close) results.

Usage:
    python audit_pipeline_match.py                    # Full audit
    python audit_pipeline_match.py --skip-pull        # Skip GCE pull (reuse audit_data/)
    python audit_pipeline_match.py --skip-local       # Skip local backfill (reuse cached)
    python audit_pipeline_match.py --debug            # Verbose logging
"""
import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import TIMEFRAME_ORDER, WINDOW_SIZES
from model_training.live_predict import (
    download_live_klines_extended,
    run_live_etl,
    encode_live_features,
    load_production_models,
    batch_ensemble_predict,
)
from data_service.gap_detector import BOOTSTRAP_BARS

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

AUDIT_DIR = Path(__file__).parent / "audit_data"
GCE_DIR = AUDIT_DIR / "gce"
LOCAL_DIR = AUDIT_DIR / "local"
LOGS_DIR = Path(__file__).parent / "trading_logs"

# GCE connection details
GCE_INSTANCE = "instance-20260303-232149"
GCE_ZONE = "asia-southeast1-b"
GCE_PROJECT = "project-4d1ee130-e5dc-4495-a47"

# PASS/FAIL thresholds
STAGE1_OHLCV_MATCH_MIN = 1.00        # 100% match at common timestamps
STAGE1_MAX_GAPS = 0                   # 0 gaps in GCE data
STAGE2_SLOPE_MAE_MAX = 1e-6           # slope_f / slope_b MAE
STAGE2_ANGLE_MAE_MAX = 0.01           # angle MAE (degrees)
STAGE2_CORRELATION_MIN = 0.9999       # Pearson correlation for slope_f
STAGE3_TEMPORAL_MAE_MAX = 0.0001      # Temporal features (nearly exact)
STAGE3_FEATURE_MAE_THRESHOLD = 0.05   # 95% of features within this
STAGE3_FEATURE_MAE_HARD_MAX = 0.5     # No feature above this
STAGE3_PERCENT_WITHIN = 0.95          # 95% must be within threshold
STAGE4_PROB_MAE_MAX = 0.03            # Probability MAE
PREDICTION_THRESHOLD = 0.70           # Match GCE config


# ============================================================================
# Stage 0: Pull GCE Data
# ============================================================================

def pull_gce_data() -> Path:
    """Pull /data from GCE Docker container via gcloud SSH.

    Returns path to extracted GCE data directory.
    """
    print("\n" + "=" * 60)
    print(" STAGE 0: Pulling GCE Data")
    print("=" * 60)

    GCE_DIR.mkdir(parents=True, exist_ok=True)

    # Build the SSH command to tar the data directory from the container
    tar_cmd = "docker exec tamagochi-data tar czf - /data/"
    ssh_cmd = [
        "gcloud", "compute", "ssh", GCE_INSTANCE,
        f"--zone={GCE_ZONE}",
        f"--project={GCE_PROJECT}",
        f"--command={tar_cmd}",
        "--quiet",
    ]

    print(f"  Running: gcloud compute ssh ... --command=\"{tar_cmd}\"")
    t0 = time.time()

    result = subprocess.run(
        ssh_cmd,
        capture_output=True,
        timeout=120,
    )

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"GCE SSH failed (rc={result.returncode}):\n{stderr}")

    # Extract tar from stdout
    tar_bytes = BytesIO(result.stdout)
    with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tar:
        tar.extractall(path=str(GCE_DIR))

    pull_time = time.time() - t0
    print(f"  Extracted to {GCE_DIR} ({pull_time:.1f}s)")

    # The tar extracts as /data/ inside GCE_DIR
    gce_data_dir = GCE_DIR / "data"
    if not gce_data_dir.exists():
        # Try without leading slash
        for item in GCE_DIR.iterdir():
            if item.is_dir():
                gce_data_dir = item
                break

    print(f"  GCE data at: {gce_data_dir}")

    # List what we got
    for subdir in sorted(gce_data_dir.iterdir()):
        if subdir.is_dir():
            files = list(subdir.iterdir())
            print(f"    {subdir.name}/: {len(files)} files")

    # Also pull model checksums from the container
    md5_cmd = "docker exec tamagochi-data md5sum /app/model_training/results_v10/production/production_model_s*.cbm"
    md5_result = subprocess.run(
        ["gcloud", "compute", "ssh", GCE_INSTANCE,
         f"--zone={GCE_ZONE}", f"--project={GCE_PROJECT}",
         f"--command={md5_cmd}", "--quiet"],
        capture_output=True, timeout=60,
    )
    gce_checksums = {}
    if md5_result.returncode == 0:
        for line in md5_result.stdout.decode().strip().split("\n"):
            if line.strip():
                parts = line.strip().split()
                if len(parts) == 2:
                    checksum, fpath = parts
                    fname = fpath.split("/")[-1]
                    gce_checksums[fname] = checksum
        print(f"  Model checksums: {len(gce_checksums)} files")

    # Save checksums for Stage 5
    with open(AUDIT_DIR / "gce_model_checksums.json", "w") as f:
        json.dump(gce_checksums, f)

    return gce_data_dir


# ============================================================================
# Stage 0b: Run Local Backfill
# ============================================================================

def run_local_backfill() -> dict:
    """Run local backfill pipeline and cache intermediate results.

    Returns dict with klines_dict, decomposed, features_df, predictions.
    """
    print("\n" + "=" * 60)
    print(" STAGE 0b: Running Local Backfill")
    print("=" * 60)

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Load models
    print("  [1/5] Loading production models...")
    models, metadata = load_production_models()
    feature_names = metadata["feature_names"]

    # Download klines (1400 5M bars to match GCE BOOTSTRAP_BARS)
    print(f"  [2/5] Downloading klines (5M: {BOOTSTRAP_BARS} bars)...")
    klines_dict = download_live_klines_extended(bars_5m=BOOTSTRAP_BARS)

    # Run ETL
    print("  [3/5] Running ETL...")
    decomposed = run_live_etl(klines_dict)

    # Encode features
    print("  [4/5] Encoding features...")
    features_df = encode_live_features(klines_dict, decomposed)

    # Predict last row
    print("  [5/5] Running prediction...")
    predictions = batch_ensemble_predict(
        models, features_df.iloc[[-1]], feature_names,
        threshold=PREDICTION_THRESHOLD,
    )

    total = time.time() - t0
    print(f"  Local backfill complete: {total:.1f}s")
    print(f"    Klines: {len(klines_dict)} TFs")
    print(f"    Decomposed: {len(decomposed)} combos")
    print(f"    Features: {features_df.shape}")
    print(f"    Prediction: {predictions['signal'].iloc[0]} "
          f"(conf={predictions['confidence'].iloc[0]:.4f})")

    return {
        "models": models,
        "metadata": metadata,
        "klines_dict": klines_dict,
        "decomposed": decomposed,
        "features_df": features_df,
        "predictions": predictions,
    }


# ============================================================================
# Stage 1: Raw Klines (L1)
# ============================================================================

def audit_stage1(gce_data_dir: Path, local_data: dict) -> list[dict]:
    """Compare raw klines at common timestamps.

    Returns list of check results.
    """
    print("\n" + "=" * 60)
    print(" STAGE 1: RAW KLINES (L1)")
    print("=" * 60)

    checks = []
    klines_dict = local_data["klines_dict"]
    gce_klines_dir = gce_data_dir / "klines"

    for tf in TIMEFRAME_ORDER:
        gce_path = gce_klines_dir / f"ml_data_{tf}.csv"
        if not gce_path.exists():
            print(f"  {tf}: SKIP (no GCE file)")
            checks.append({"name": f"L1_{tf}_exists", "passed": False,
                           "detail": "No GCE kline file"})
            continue

        gce_df = pd.read_csv(gce_path)
        gce_df["time"] = pd.to_datetime(gce_df["time"])
        local_df = klines_dict[tf].copy()
        local_df["time"] = pd.to_datetime(local_df["time"]).dt.tz_localize(None)

        # Find common timestamps
        common_times = set(gce_df["time"]) & set(local_df["time"])
        n_common = len(common_times)
        n_gce = len(gce_df)
        n_local = len(local_df)

        overlap = n_common / max(n_gce, n_local) if max(n_gce, n_local) > 0 else 0

        print(f"  {tf}: {n_gce} rows (GCE) vs {n_local} rows (local) "
              f"- {n_common} common ({overlap:.1%})")

        if n_common == 0:
            checks.append({"name": f"L1_{tf}_overlap", "passed": False,
                           "detail": "No common timestamps"})
            continue

        # OHLCV match at common timestamps
        gce_common = gce_df[gce_df["time"].isin(common_times)].sort_values("time")
        local_common = local_df[local_df["time"].isin(common_times)].sort_values("time")

        # Align by time
        gce_common = gce_common.set_index("time")
        local_common = local_common.set_index("time")
        common_idx = gce_common.index.intersection(local_common.index)

        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
        available_cols = [c for c in ohlcv_cols if c in gce_common.columns
                         and c in local_common.columns]

        n_match = 0
        n_total = len(common_idx)

        for col in available_cols:
            gce_vals = pd.to_numeric(gce_common.loc[common_idx, col], errors="coerce")
            local_vals = pd.to_numeric(local_common.loc[common_idx, col], errors="coerce")
            # Compare with tolerance for float precision
            match = np.isclose(gce_vals, local_vals, rtol=1e-10, atol=1e-10)
            n_col_match = match.sum()
            if n_col_match < n_total:
                n_mismatch = n_total - n_col_match
                max_diff = np.abs(gce_vals - local_vals).max()
                print(f"    {col}: {n_col_match}/{n_total} match "
                      f"({n_mismatch} differ, max_diff={max_diff:.2e})")

        # Overall OHLCV match: compare all columns at once
        gce_ohlcv = gce_common.loc[common_idx, available_cols].astype(float)
        local_ohlcv = local_common.loc[common_idx, available_cols].astype(float)
        all_close = np.isclose(gce_ohlcv, local_ohlcv, rtol=1e-10, atol=1e-10)
        match_pct = all_close.values.mean()

        passed = match_pct >= STAGE1_OHLCV_MATCH_MIN
        print(f"    OHLCV match: {match_pct:.4%}{'':>30s}"
              f"{'PASS' if passed else 'FAIL'}")
        checks.append({"name": f"L1_{tf}_ohlcv", "passed": passed,
                       "detail": f"{match_pct:.4%} match at {n_common} common timestamps"})

        # Gap check on GCE data
        gce_sorted = gce_df.sort_values("time")
        time_diffs = gce_sorted["time"].diff().dropna()
        from data_service.gap_detector import TF_MINUTES
        expected_delta = pd.Timedelta(minutes=TF_MINUTES[tf])
        gaps = (time_diffs > expected_delta * 1.5).sum()
        gap_passed = gaps <= STAGE1_MAX_GAPS
        print(f"    Gaps: {gaps}{'':>43s}"
              f"{'PASS' if gap_passed else 'FAIL'}")
        checks.append({"name": f"L1_{tf}_gaps", "passed": gap_passed,
                       "detail": f"{gaps} gaps detected"})

    return checks


# ============================================================================
# Stage 2: Decomposed Regression (L2)
# ============================================================================

def audit_stage2(gce_data_dir: Path, local_data: dict) -> list[dict]:
    """Compare decomposed regression at common timestamps.

    Returns list of check results.
    """
    print("\n" + "=" * 60)
    print(" STAGE 2: DECOMPOSED REGRESSION (L2)")
    print("=" * 60)

    checks = []
    decomposed = local_data["decomposed"]
    gce_decomp_dir = gce_data_dir / "decomposed"

    worst_slope_mae = 0
    worst_angle_mae = 0
    worst_combo = ""

    n_combos = 0
    n_passed = 0

    for tf in TIMEFRAME_ORDER:
        for ws in WINDOW_SIZES:
            key = (tf, ws)
            gce_path = gce_decomp_dir / f"decomposed_{tf}_w{ws}.csv"

            if not gce_path.exists():
                continue
            if key not in decomposed:
                continue

            n_combos += 1

            gce_df = pd.read_csv(gce_path)
            gce_df["time"] = pd.to_datetime(gce_df["time"])
            local_df = decomposed[key].copy()
            local_df["time"] = pd.to_datetime(local_df["time"]).dt.tz_localize(None)

            # Common timestamps
            common_times = sorted(set(gce_df["time"]) & set(local_df["time"]))
            if len(common_times) == 0:
                checks.append({"name": f"L2_{tf}_w{ws}", "passed": False,
                               "detail": "No common timestamps"})
                continue

            # Use last N common timestamps (most relevant for prediction)
            n_compare = min(100, len(common_times))
            compare_times = common_times[-n_compare:]

            gce_sub = gce_df[gce_df["time"].isin(compare_times)].sort_values("time")
            local_sub = local_df[local_df["time"].isin(compare_times)].sort_values("time")

            # Align
            gce_sub = gce_sub.set_index("time")
            local_sub = local_sub.set_index("time")
            common_idx = gce_sub.index.intersection(local_sub.index)

            if len(common_idx) == 0:
                continue

            # slope_f MAE
            sf_mae = np.abs(
                gce_sub.loc[common_idx, "slope_f"].astype(float) -
                local_sub.loc[common_idx, "slope_f"].astype(float)
            ).mean()

            # slope_b MAE
            sb_mae = np.abs(
                gce_sub.loc[common_idx, "slope_b"].astype(float) -
                local_sub.loc[common_idx, "slope_b"].astype(float)
            ).mean()

            # angle MAE
            angle_mae = np.abs(
                gce_sub.loc[common_idx, "angle"].astype(float) -
                local_sub.loc[common_idx, "angle"].astype(float)
            ).mean()

            # Correlation
            gce_sf = gce_sub.loc[common_idx, "slope_f"].astype(float)
            local_sf = local_sub.loc[common_idx, "slope_f"].astype(float)
            if gce_sf.std() > 0 and local_sf.std() > 0:
                corr = gce_sf.corr(local_sf)
            else:
                corr = 1.0 if np.allclose(gce_sf, local_sf) else 0.0

            slope_pass = sf_mae < STAGE2_SLOPE_MAE_MAX and sb_mae < STAGE2_SLOPE_MAE_MAX
            angle_pass = angle_mae < STAGE2_ANGLE_MAE_MAX
            corr_pass = corr >= STAGE2_CORRELATION_MIN
            combo_pass = slope_pass and angle_pass and corr_pass

            if combo_pass:
                n_passed += 1

            # Track worst case
            if sf_mae > worst_slope_mae:
                worst_slope_mae = sf_mae
                worst_combo = f"{tf}_w{ws}"
            if angle_mae > worst_angle_mae:
                worst_angle_mae = angle_mae

            status = "PASS" if combo_pass else "FAIL"
            print(f"  {tf}_w{ws}: sf_MAE={sf_mae:.2e} sb_MAE={sb_mae:.2e} "
                  f"angle_MAE={angle_mae:.4f} corr={corr:.6f}  {status}")

            checks.append({
                "name": f"L2_{tf}_w{ws}",
                "passed": combo_pass,
                "detail": f"sf_MAE={sf_mae:.2e}, angle_MAE={angle_mae:.4f}, corr={corr:.6f}",
            })

    print(f"\n  Summary: {n_passed}/{n_combos} combos PASS")
    print(f"  Worst slope_f MAE: {worst_slope_mae:.2e} ({worst_combo})")
    print(f"  Worst angle MAE: {worst_angle_mae:.4f}")

    return checks


# ============================================================================
# Stage 3: Encoded Features (L3 input)
# ============================================================================

def audit_stage3(gce_data_dir: Path, local_data: dict) -> list[dict]:
    """Compare encoded features at the last common 5M timestamp.

    Returns list of check results.
    """
    print("\n" + "=" * 60)
    print(" STAGE 3: ENCODED FEATURES (L3 input)")
    print("=" * 60)

    checks = []
    local_features = local_data["features_df"]
    metadata = local_data["metadata"]
    feature_names = metadata["feature_names"]

    # For GCE, we need to re-encode from GCE data to get the feature row.
    # The GCE pipeline doesn't store intermediate features — only predictions.
    # So we build a GCE feature row by running the encode pipeline on GCE's
    # klines + decomposed data.
    print("  Re-encoding GCE data locally (same code, GCE's klines/decomposed)...")

    gce_klines_dir = gce_data_dir / "klines"
    gce_decomp_dir = gce_data_dir / "decomposed"

    # Load GCE klines
    gce_klines_dict = {}
    for tf in TIMEFRAME_ORDER:
        kl_path = gce_klines_dir / f"ml_data_{tf}.csv"
        if not kl_path.exists():
            print(f"    WARN: No GCE klines for {tf}")
            continue
        df = pd.read_csv(kl_path)
        df["Open Time"] = pd.to_datetime(df["Open Time"])
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
        for c in ("Open", "High", "Low", "Close", "Volume"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        gce_klines_dict[tf] = df.tail(BOOTSTRAP_BARS).reset_index(drop=True)

    # Load GCE decomposed
    gce_decomposed = {}
    for tf in TIMEFRAME_ORDER:
        for ws in WINDOW_SIZES:
            decomp_path = gce_decomp_dir / f"decomposed_{tf}_w{ws}.csv"
            if not decomp_path.exists():
                continue
            df = pd.read_csv(decomp_path)
            df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
            df = df.sort_values("time").reset_index(drop=True)
            gce_decomposed[(tf, ws)] = df.tail(BOOTSTRAP_BARS).reset_index(drop=True)

    # Encode GCE features
    gce_features = encode_live_features(gce_klines_dict, gce_decomposed)

    # Find last common timestamp
    local_time = pd.to_datetime(local_features["time"].iloc[-1])
    gce_time = pd.to_datetime(gce_features["time"].iloc[-1])

    print(f"  Local last time:  {local_time}")
    print(f"  GCE last time:    {gce_time}")

    # Use the earlier of the two as comparison point
    if local_time == gce_time:
        compare_time = local_time
    else:
        compare_time = min(local_time, gce_time)
        print(f"  Compare at:       {compare_time}")

    # Get feature rows
    local_row = local_features[
        pd.to_datetime(local_features["time"]) == compare_time
    ]
    gce_row = gce_features[
        pd.to_datetime(gce_features["time"]) == compare_time
    ]

    if len(local_row) == 0 or len(gce_row) == 0:
        # Fall back to closest match
        print("  WARN: Exact time match failed, using last rows")
        local_row = local_features.iloc[[-1]]
        gce_row = gce_features.iloc[[-1]]

    # Check feature names
    local_feat_cols = sorted([c for c in local_features.columns if c != "time"])
    gce_feat_cols = sorted([c for c in gce_features.columns if c != "time"])

    n_local = len(local_feat_cols)
    n_gce = len(gce_feat_cols)
    names_match = set(local_feat_cols) == set(gce_feat_cols)
    print(f"  Features: {n_local} (local) vs {n_gce} (GCE) — "
          f"{'MATCH' if names_match else 'MISMATCH'}")
    checks.append({"name": "L3_feature_names", "passed": names_match,
                   "detail": f"{n_local} local, {n_gce} GCE"})

    # Compare feature values
    common_features = sorted(set(local_feat_cols) & set(gce_feat_cols))
    print(f"  Comparing {len(common_features)} common features...")

    local_vals = local_row[common_features].iloc[0].astype(float)
    gce_vals = gce_row[common_features].iloc[0].astype(float)

    abs_diffs = np.abs(local_vals - gce_vals)

    # Categorize features
    temporal_feats = [f for f in common_features
                      if any(f.startswith(p) for p in
                             ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin"])]
    regression_feats = [f for f in common_features
                        if any(p in f for p in
                               ["slope_f_mag", "slope_b_slow", "angle_slow",
                                "corr_slow", "accel_raw", "accel_mag"])]
    crossing_feats = [f for f in common_features
                      if any(p in f for p in
                             ["cross_div", "cross_traj", "cross_dir"])]
    volume_feats = [f for f in common_features
                    if any(p in f for p in
                           ["vol_body_product", "vol_ratio", "cumsum_body",
                            "ema_"])]
    stoch_feats = [f for f in common_features
                   if "stoch_pos" in f]
    atr_feats = [f for f in common_features if "atr_" in f]

    feature_groups = {
        "Temporal": temporal_feats,
        "Regression": regression_feats,
        "Crossings": crossing_feats,
        "Volume/EMA": volume_feats,
        "Stochastic": stoch_feats,
        "ATR": atr_feats,
    }
    categorized = set()
    for feats in feature_groups.values():
        categorized.update(feats)
    other_feats = [f for f in common_features if f not in categorized]
    feature_groups["Other"] = other_feats

    for group_name, feats in feature_groups.items():
        if not feats:
            continue
        group_diffs = abs_diffs[feats]
        group_mae = group_diffs.mean()
        group_max = group_diffs.max()
        n_within = (group_diffs < STAGE3_FEATURE_MAE_THRESHOLD).sum()
        print(f"    {group_name} ({len(feats)}): "
              f"MAE={group_mae:.4f}, max={group_max:.4f}, "
              f"{n_within}/{len(feats)} within {STAGE3_FEATURE_MAE_THRESHOLD}")

    # Temporal exact match
    if temporal_feats:
        temp_mae = abs_diffs[temporal_feats].max()
        temp_pass = temp_mae < STAGE3_TEMPORAL_MAE_MAX
        checks.append({"name": "L3_temporal", "passed": temp_pass,
                       "detail": f"max MAE={temp_mae:.6f}"})
        print(f"    Temporal exact: max_diff={temp_mae:.6f}{'':>20s}"
              f"{'PASS' if temp_pass else 'FAIL'}")

    # Overall: 95% within threshold
    n_within = (abs_diffs < STAGE3_FEATURE_MAE_THRESHOLD).sum()
    pct_within = n_within / len(common_features) if common_features else 0
    pct_pass = pct_within >= STAGE3_PERCENT_WITHIN
    print(f"    Overall: {n_within}/{len(common_features)} within "
          f"{STAGE3_FEATURE_MAE_THRESHOLD} ({pct_within:.1%})"
          f"{'':>10s}{'PASS' if pct_pass else 'FAIL'}")
    checks.append({"name": "L3_pct_within", "passed": pct_pass,
                   "detail": f"{pct_within:.1%} within {STAGE3_FEATURE_MAE_THRESHOLD}"})

    # No feature above hard max
    max_diff = abs_diffs.max()
    max_feat = abs_diffs.idxmax()
    hard_pass = max_diff <= STAGE3_FEATURE_MAE_HARD_MAX
    print(f"    Max diff: {max_diff:.4f} ({max_feat})"
          f"{'':>20s}{'PASS' if hard_pass else 'FAIL'}")
    checks.append({"name": "L3_hard_max", "passed": hard_pass,
                   "detail": f"max={max_diff:.4f} ({max_feat})"})

    # Print top 10 biggest diffs for debugging
    top_diffs = abs_diffs.nlargest(10)
    print("\n    Top 10 differences:")
    for feat, diff in top_diffs.items():
        local_v = local_vals[feat]
        gce_v = gce_vals[feat]
        print(f"      {feat}: {local_v:.6f} vs {gce_v:.6f} (diff={diff:.6f})")

    return checks


# ============================================================================
# Stage 4: Predictions (L3 output)
# ============================================================================

def audit_stage4(gce_data_dir: Path, local_data: dict) -> list[dict]:
    """Compare final prediction probabilities.

    Returns list of check results.
    """
    print("\n" + "=" * 60)
    print(" STAGE 4: PREDICTIONS (L3 output)")
    print("=" * 60)

    checks = []
    local_pred = local_data["predictions"]

    # Load GCE predictions
    gce_pred_path = gce_data_dir / "predictions" / "predictions.csv"
    if not gce_pred_path.exists():
        print("  SKIP: No GCE predictions file")
        checks.append({"name": "L4_exists", "passed": False,
                       "detail": "No GCE predictions file"})
        return checks

    gce_preds = pd.read_csv(gce_pred_path)
    gce_preds["time"] = pd.to_datetime(gce_preds["time"])

    print(f"  GCE predictions: {len(gce_preds)} rows "
          f"({gce_preds['time'].min()} -> {gce_preds['time'].max()})")

    # Get last GCE prediction
    gce_last = gce_preds.iloc[-1]
    local_last = local_pred.iloc[0]  # We predicted only 1 row

    # Find matching timestamp
    local_time = pd.to_datetime(local_last["time"])
    gce_time = pd.to_datetime(gce_last["time"])

    print(f"  Local pred time: {local_time}")
    print(f"  GCE pred time:   {gce_time}")

    # Try to find matching time in GCE predictions
    match_row = gce_preds[pd.to_datetime(gce_preds["time"]) == local_time]
    if len(match_row) > 0:
        gce_compare = match_row.iloc[-1]
        print(f"  Matched at: {local_time}")
    else:
        # Use last GCE prediction
        gce_compare = gce_last
        time_diff = abs((gce_time - local_time).total_seconds())
        print(f"  No exact match. Using last GCE pred (time diff={time_diff:.0f}s)")
        if time_diff > 600:
            print(f"  WARNING: Predictions are >10min apart — comparison may be misleading")

    # Signal agreement
    local_signal = local_last["signal"]
    gce_signal = gce_compare["signal"]
    signal_match = local_signal == gce_signal
    print(f"  Signal: {local_signal} vs {gce_signal}{'':>20s}"
          f"{'PASS' if signal_match else 'FAIL'}")
    checks.append({"name": "L4_signal", "passed": signal_match,
                   "detail": f"{local_signal} vs {gce_signal}"})

    # Probability comparison
    for prob_col in ["prob_no_trade", "prob_long", "prob_short"]:
        local_p = float(local_last[prob_col])
        gce_p = float(gce_compare[prob_col])
        diff = abs(local_p - gce_p)
        passed = diff <= STAGE4_PROB_MAE_MAX
        print(f"  {prob_col}: {local_p:.4f} vs {gce_p:.4f} "
              f"(diff={diff:.4f}){'':>10s}{'PASS' if passed else 'FAIL'}")
        checks.append({"name": f"L4_{prob_col}", "passed": passed,
                       "detail": f"{local_p:.4f} vs {gce_p:.4f} (diff={diff:.4f})"})

    # Confidence comparison
    local_conf = float(local_last["confidence"])
    gce_conf = float(gce_compare["confidence"])
    conf_diff = abs(local_conf - gce_conf)
    conf_pass = conf_diff <= STAGE4_PROB_MAE_MAX
    print(f"  confidence: {local_conf:.4f} vs {gce_conf:.4f} "
          f"(diff={conf_diff:.4f}){'':>10s}{'PASS' if conf_pass else 'FAIL'}")
    checks.append({"name": "L4_confidence", "passed": conf_pass,
                   "detail": f"{local_conf:.4f} vs {gce_conf:.4f} (diff={conf_diff:.4f})"})

    return checks


# ============================================================================
# Stage 5: Governance
# ============================================================================

def audit_stage5(local_data: dict) -> list[dict]:
    """Check model checksums, feature names, NaN, config.

    Returns list of check results.
    """
    print("\n" + "=" * 60)
    print(" STAGE 5: GOVERNANCE")
    print("=" * 60)

    checks = []
    metadata = local_data["metadata"]
    features_df = local_data["features_df"]

    # Model checksums
    gce_checksums_path = AUDIT_DIR / "gce_model_checksums.json"
    if gce_checksums_path.exists():
        with open(gce_checksums_path) as f:
            gce_checksums = json.load(f)

        local_checksums = {}
        model_dir = Path("model_training/results_v10/production")
        for seed in [42, 123, 777]:
            model_path = model_dir / f"production_model_s{seed}.cbm"
            if model_path.exists():
                md5 = hashlib.md5(model_path.read_bytes()).hexdigest()
                local_checksums[f"production_model_s{seed}.cbm"] = md5

        n_match = 0
        n_total = len(local_checksums)
        for fname, local_md5 in local_checksums.items():
            gce_md5 = gce_checksums.get(fname, "")
            if local_md5 == gce_md5:
                n_match += 1
            else:
                print(f"    MISMATCH: {fname}: local={local_md5[:8]}... "
                      f"gce={gce_md5[:8] if gce_md5 else 'N/A'}...")

        checksum_pass = n_match == n_total and n_total > 0
        print(f"  Model checksums: {n_match}/{n_total} match"
              f"{'':>20s}{'PASS' if checksum_pass else 'FAIL'}")
        checks.append({"name": "L5_checksums", "passed": checksum_pass,
                       "detail": f"{n_match}/{n_total} match"})
    else:
        print("  Model checksums: SKIP (no GCE checksums pulled)")
        checks.append({"name": "L5_checksums", "passed": False,
                       "detail": "GCE checksums not available"})

    # Feature name list from metadata
    feature_names = metadata.get("feature_names", [])
    n_expected = metadata.get("n_features", 508)
    n_actual = len(feature_names)
    feat_count_pass = n_actual == n_expected
    print(f"  Feature names: {n_actual}/{n_expected}"
          f"{'':>28s}{'PASS' if feat_count_pass else 'FAIL'}")
    checks.append({"name": "L5_feature_count", "passed": feat_count_pass,
                   "detail": f"{n_actual}/{n_expected} features in metadata"})

    # NaN in last feature row
    last_row = features_df.iloc[-1]
    feat_cols = [c for c in features_df.columns if c != "time"]
    n_nan = last_row[feat_cols].isna().sum()
    nan_pass = n_nan == 0
    print(f"  NaN count: {n_nan}"
          f"{'':>38s}{'PASS' if nan_pass else 'FAIL'}")
    checks.append({"name": "L5_nan", "passed": nan_pass,
                   "detail": f"{n_nan} NaN in last feature row"})

    if n_nan > 0:
        nan_feats = [c for c in feat_cols if pd.isna(last_row[c])]
        print(f"    NaN features: {nan_feats[:10]}...")

    # Config match
    print(f"  BOOTSTRAP_BARS: {BOOTSTRAP_BARS}")
    print(f"  Threshold: {PREDICTION_THRESHOLD}")
    print(f"  Model version: {metadata.get('model_version', 'unknown')}")

    config_pass = (BOOTSTRAP_BARS == 1400 and
                   metadata.get("model_version") == "V10")
    print(f"  Config consistency:"
          f"{'':>28s}{'PASS' if config_pass else 'FAIL'}")
    checks.append({"name": "L5_config", "passed": config_pass,
                   "detail": f"BOOTSTRAP={BOOTSTRAP_BARS}, version={metadata.get('model_version')}"})

    return checks


# ============================================================================
# Report
# ============================================================================

def print_report(all_checks: list[dict], start_time: float):
    """Print final audit report."""
    total_time = time.time() - start_time
    n_pass = sum(1 for c in all_checks if c["passed"])
    n_total = len(all_checks)
    overall = "PASS" if n_pass == n_total else "FAIL"

    report = []
    report.append("")
    report.append("=" * 60)
    report.append(" PIPELINE AUDIT REPORT")
    report.append(" Local Backfill vs GCE Docker")
    report.append(f" Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    report.append(f" Duration: {total_time:.1f}s")
    report.append("=" * 60)
    report.append("")

    # Group by stage
    stages = {}
    for check in all_checks:
        name = check["name"]
        stage = name.split("_")[0]
        if stage not in stages:
            stages[stage] = []
        stages[stage].append(check)

    stage_names = {
        "L1": "RAW KLINES",
        "L2": "DECOMPOSED",
        "L3": "FEATURES",
        "L4": "PREDICTIONS",
        "L5": "GOVERNANCE",
    }

    for stage_key in ["L1", "L2", "L3", "L4", "L5"]:
        if stage_key not in stages:
            continue
        stage_checks = stages[stage_key]
        n_stage_pass = sum(1 for c in stage_checks if c["passed"])
        n_stage = len(stage_checks)
        stage_status = "PASS" if n_stage_pass == n_stage else "FAIL"

        report.append(f"STAGE {stage_key[1:]}: {stage_names.get(stage_key, stage_key)}")
        for check in stage_checks:
            status = "PASS" if check["passed"] else "FAIL"
            report.append(f"  {check['name']}: {check['detail']:<50s} {status}")
        report.append(f"  {'':->56s} {n_stage_pass}/{n_stage} {stage_status}")
        report.append("")

    report.append("=" * 60)
    report.append(f" OVERALL: {overall} ({n_pass}/{n_total} checks)")
    report.append("=" * 60)

    report_text = "\n".join(report)
    print(report_text)

    # Save report
    LOGS_DIR.mkdir(exist_ok=True)
    report_path = LOGS_DIR / f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\nReport saved: {report_path}")

    return overall == "PASS"


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full Pipeline Audit — Local Backfill vs GCE Docker")
    parser.add_argument("--skip-pull", action="store_true",
                        help="Skip GCE data pull (reuse existing audit_data/gce/)")
    parser.add_argument("--skip-local", action="store_true",
                        help="Skip local backfill (reuse cached data — not implemented)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    start_time = time.time()
    all_checks = []

    print("=" * 60)
    print(" FULL PIPELINE AUDIT")
    print(" Local Backfill vs GCE Docker")
    print(f" Started: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print("=" * 60)

    # Stage 0: Pull GCE data
    if args.skip_pull:
        gce_data_dir = GCE_DIR / "data"
        if not gce_data_dir.exists():
            # Try finding the data directory
            for item in GCE_DIR.iterdir():
                if item.is_dir():
                    gce_data_dir = item
                    break
        print(f"\n  Reusing GCE data at: {gce_data_dir}")
    else:
        gce_data_dir = pull_gce_data()

    # Stage 0b: Run local backfill
    local_data = run_local_backfill()

    # Stage 1: Raw Klines
    checks_1 = audit_stage1(gce_data_dir, local_data)
    all_checks.extend(checks_1)

    # Stage 2: Decomposed Regression
    checks_2 = audit_stage2(gce_data_dir, local_data)
    all_checks.extend(checks_2)

    # Stage 3: Encoded Features
    checks_3 = audit_stage3(gce_data_dir, local_data)
    all_checks.extend(checks_3)

    # Stage 4: Predictions
    checks_4 = audit_stage4(gce_data_dir, local_data)
    all_checks.extend(checks_4)

    # Stage 5: Governance
    checks_5 = audit_stage5(local_data)
    all_checks.extend(checks_5)

    # Final report
    passed = print_report(all_checks, start_time)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
