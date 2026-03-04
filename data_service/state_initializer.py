"""Extract incremental encoder state from a batch-encoded feature matrix + raw data.

Recovers EMA states, rolling buffers, and lag values so that the incremental
encoder produces identical output to continuing the batch encoder.

EMA state recovery (mathematically exact):
  ema_volume   = volume / vol_ratio        (encode_v10.py:354)
  ema_vbp      = directional_vol_body      (feature IS the EMA, encode_v10.py:361)
  ema_momentum = smoothed_momentum         (feature IS the EMA, encode_v10.py:274)
  atr          = atr_normalized * close    (encode_v10.py:401)

Rolling buffers: read last N rows of native TF klines/decomposed.
Lag values: read last rows of features + decomposed.

Usage:
    state = initialize_state(feature_matrix_path, klines_dir, decomposed_dir)
    encoder = IncrementalEncoder(state)
"""

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import TIMEFRAME_ORDER, WINDOW_SIZES
from .incremental_encoder import (
    IncrementalEncoder,
    W_FAST, W_SLOW, ALL_WINDOWS, CROSSING_PAIRS, ATR_TFS,
    D_SLOPE_F_BUF_SIZE, CORR_BUF_SIZE, WICK_BUF_SIZE, VOL_BUF_SIZE,
    BODY_PCT_BUF_SIZE, STOCH_WINDOWS, UP_BAR_BUF_SIZE, REVERSAL_WINDOW,
    KEEP_D_SLOPE_F_TFS, KEEP_D_SLOPE_B_TFS,
)

logger = logging.getLogger(__name__)


def _load_native_klines(klines_dir, tf, n_rows=50):
    """Load last n_rows of native TF kline data."""
    path = Path(klines_dir) / f"ml_data_{tf}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Handle both data_service format (has "time") and actual_data format (has "Open Time")
    if "time" not in df.columns and "Open Time" in df.columns:
        df["time"] = pd.to_datetime(df["Open Time"]).dt.tz_localize(None)
    else:
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)
    return df.tail(n_rows).reset_index(drop=True)


def _load_native_decomposed(decomposed_dir, tf, ws, n_rows=50):
    """Load last n_rows of native TF decomposed data."""
    path = Path(decomposed_dir) / f"decomposed_{tf}_w{ws}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df.sort_values("time").reset_index(drop=True)
    return df.tail(n_rows).reset_index(drop=True)


def initialize_state(feature_matrix_path, klines_dir, decomposed_dir):
    """Extract incremental encoder state from batch-encoded data.

    Args:
        feature_matrix_path: Path to feature_matrix_v10.parquet
        klines_dir: Directory with ml_data_{tf}.csv files
        decomposed_dir: Directory with decomposed_{tf}_w{ws}.csv files

    Returns:
        dict: State ready for IncrementalEncoder(state)
    """
    logger.info("Initializing incremental encoder state from batch data...")

    fm = pd.read_parquet(feature_matrix_path)
    fm["time"] = pd.to_datetime(fm["time"])
    fm = fm.sort_values("time").reset_index(drop=True)
    logger.info(f"  Feature matrix: {fm.shape}, {fm['time'].min()} → {fm['time'].max()}")

    state = IncrementalEncoder._empty_state()
    state["last_timestamp"] = str(fm["time"].iloc[-1])

    # xtf_total_converging at last row (for convergence_momentum lag)
    if "xtf_total_converging" in fm.columns:
        state["prev_xtf_total_converging"] = float(fm["xtf_total_converging"].iloc[-1])

    for tf in TIMEFRAME_ORDER:
        logger.info(f"  Initializing {tf}...")

        # === Decomposed state ===
        ds = state["decomposed"][tf]

        df_fast = _load_native_decomposed(decomposed_dir, tf, W_FAST, n_rows=D_SLOPE_F_BUF_SIZE + 5)
        df_slow = _load_native_decomposed(decomposed_dir, tf, W_SLOW, n_rows=D_SLOPE_F_BUF_SIZE + 5)

        if df_fast is not None and df_slow is not None and len(df_fast) >= 2 and len(df_slow) >= 2:
            # Merge w30 and w120 on time
            merged = pd.merge(
                df_fast[["time", "slope_f", "angle"]].rename(
                    columns={"slope_f": "sf_fast", "angle": "angle_fast"}),
                df_slow[["time", "slope_f", "slope_b", "angle", "corr",
                         "acceleration", "p_value_f"]].rename(
                    columns={"slope_f": "sf_slow", "slope_b": "sb_slow",
                             "angle": "angle_slow", "corr": "corr_slow",
                             "acceleration": "accel_slow",
                             "p_value_f": "pf_slow"}),
                on="time", how="inner"
            ).sort_values("time").reset_index(drop=True)

            if len(merged) >= 2:
                last = merged.iloc[-1]
                prev = merged.iloc[-2]

                # Current values
                ds["sf_fast"] = float(last.get("sf_fast", 0))
                ds["sf_slow"] = float(last.get("sf_slow", 0))
                ds["sb_slow"] = float(last.get("sb_slow", 0))
                ds["angle_fast"] = float(last.get("angle_fast", 0))
                ds["angle_slow"] = float(last.get("angle_slow", 0))
                ds["corr_slow"] = float(last.get("corr_slow", 0))
                ds["accel_slow"] = float(last.get("accel_slow", 0))
                ds["pf_slow"] = float(last.get("pf_slow", 1.0))

                # Previous values
                ds["prev_sf_fast"] = float(prev.get("sf_fast", 0))
                ds["prev_sf_slow"] = float(prev.get("sf_slow", 0))
                ds["prev_sb_slow"] = float(prev.get("sb_slow", 0))
                ds["prev_angle_fast"] = float(prev.get("angle_fast", 0))
                ds["prev_angle_slow"] = float(prev.get("angle_slow", 0))
                ds["prev_corr_slow"] = float(prev.get("corr_slow", 0))

                # cross_div lags
                cross_divs = (merged["angle_fast"] - merged["angle_slow"]).values
                ds["prev_cross_div"] = float(cross_divs[-1])
                ds["prev2_cross_div"] = float(cross_divs[-2]) if len(cross_divs) >= 2 else 0.0

                # cross_dir lag
                if len(merged) >= 3:
                    prev_af = merged["angle_fast"].iloc[-2] - merged["angle_fast"].iloc[-3]
                    prev_as = merged["angle_slow"].iloc[-2] - merged["angle_slow"].iloc[-3]
                    opposing = (prev_af * prev_as) < 0
                    if opposing and prev_as > 0:
                        ds["prev_cross_dir"] = 1
                    elif opposing and prev_as < 0:
                        ds["prev_cross_dir"] = -1

                # trend_certainty lag (from feature matrix)
                tc_col = f"trend_certainty_{tf}"
                if tc_col in fm.columns:
                    ds["prev_trend_certainty"] = float(fm[tc_col].iloc[-1])

                # d_slope_f buffer (last 23 values)
                sf_vals = merged["sf_slow"].values
                d_sf = np.diff(sf_vals)
                ds["d_sf_buf"] = list(d_sf[-D_SLOPE_F_BUF_SIZE:].astype(float))

                # corr buffer (last 10 values)
                corr_vals = merged["corr_slow"].fillna(0).values
                ds["corr_buf"] = list(corr_vals[-CORR_BUF_SIZE:].astype(float))

                # EMA momentum recovery: smoothed_momentum IS the EMA value
                sm_col = f"smoothed_momentum_{tf}"
                if sm_col in fm.columns:
                    ds["ema_momentum_5"] = float(fm[sm_col].iloc[-1])

                ds["initialized"] = True

        # === Kline state ===
        ks = state["klines"][tf]
        kl = _load_native_klines(klines_dir, tf, n_rows=VOL_BUF_SIZE + 5)

        if kl is not None and len(kl) >= 2:
            close = kl["Close"].values.astype(float)
            open_ = kl["Open"].values.astype(float)
            high = kl["High"].values.astype(float)
            low = kl["Low"].values.astype(float)
            volume = kl["Volume"].values.astype(float)

            last_c = float(close[-1])
            last_v = float(volume[-1])

            # EMA recovery: ema_vol50 = volume / vol_ratio
            vr_col = f"vol_ratio_{tf}"
            if vr_col in fm.columns:
                vr = float(fm[vr_col].iloc[-1])
                if vr > 0:
                    ks["ema_vol50"] = last_v / vr
                else:
                    ks["ema_vol50"] = last_v

            # EMA recovery: ema_vbp10 = directional_vol_body (IS the EMA)
            dvb_col = f"directional_vol_body_{tf}"
            if dvb_col in fm.columns:
                ks["ema_vbp10"] = float(fm[dvb_col].iloc[-1])

            ks["prev_close"] = last_c

            # Rolling buffers from native data
            body = close - open_
            body_sign = np.sign(body)
            up_bar = (close > open_).astype(float)

            # Body buffers
            ks["body_buf_10"] = list(body[-10:].astype(float))
            ks["body_buf_50"] = list(body[-50:].astype(float))

            # Stoch buffers
            for w in STOCH_WINDOWS:
                n = min(w, len(kl))
                ks[f"high_buf_{w}"] = list(high[-n:].astype(float))
                ks[f"low_buf_{w}"] = list(low[-n:].astype(float))
                ks[f"close_buf_{w}"] = list(close[-n:].astype(float))

            # up_bar_ratio buffer (only 3D)
            if tf == "3D":
                n = min(UP_BAR_BUF_SIZE, len(kl))
                ks["up_bar_buf_50"] = list(up_bar[-n:].astype(float))

            # Wick buffer
            body_top = np.maximum(close, open_)
            body_bot = np.minimum(close, open_)
            bar_range = high - low
            eps = 1e-10
            wick_raw = ((high - body_top) - (body_bot - low)) / (bar_range + eps)
            n = min(WICK_BUF_SIZE, len(wick_raw))
            ks["wick_buf"] = list(wick_raw[-n:].astype(float))

            # Volume buffer
            n = min(VOL_BUF_SIZE, len(volume))
            ks["vol_buf"] = list(volume[-n:].astype(float))

            # Body pct buffer
            safe_close = np.where(close > 0, close, 1.0)
            body_pct = body_sign * np.abs(body) / safe_close
            n = min(BODY_PCT_BUF_SIZE, len(body_pct))
            ks["body_pct_buf"] = list(body_pct[-n:].astype(float))

            # ATR recovery: atr = atr_normalized * close
            if tf in ATR_TFS:
                atr_col = f"atr_normalized_{tf}"
                if atr_col in fm.columns:
                    atr_norm = float(fm[atr_col].iloc[-1])
                    ks["atr"] = atr_norm * last_c

            ks["initialized"] = True

        # === Window state (for F1) ===
        ws_state = state["windows"][tf]
        for ws in ALL_WINDOWS:
            ws_key = str(ws)
            df_ws = _load_native_decomposed(
                decomposed_dir, tf, ws, n_rows=REVERSAL_WINDOW + 2)
            if df_ws is not None and len(df_ws) >= 2:
                angles = df_ws["angle"].values.astype(float)
                slope_fs = df_ws["slope_f"].values.astype(float)

                ws_state["ws"][ws_key]["angle"] = float(angles[-1])
                ws_state["ws"][ws_key]["slope_f"] = float(slope_fs[-1])
                ws_state["ws"][ws_key]["prev_angle"] = float(angles[-2])
                n = min(REVERSAL_WINDOW, len(angles))
                ws_state["ws"][ws_key]["angle_buf"] = list(angles[-n:])

        # Initialize crossing pair state
        for ws1, ws2 in CROSSING_PAIRS:
            pair_key = f"{ws1}_{ws2}"
            a1 = ws_state["ws"][str(ws1)]["angle"]
            a2 = ws_state["ws"][str(ws2)]["angle"]
            diff = a1 - a2
            ws_state["pairs"][pair_key]["prev_diff"] = diff
            ws_state["pairs"][pair_key]["prev_abs_diff"] = abs(diff)

        ws_state["initialized"] = True

        # === Store latest feature values from feature matrix ===
        latest = state["latest"][tf]
        last_row = fm.iloc[-1]
        for col in fm.columns:
            if col == "time":
                continue
            if col.endswith(f"_{tf}") or col == f"up_bar_ratio_{tf}_w50":
                latest[col] = float(last_row[col])
            # Handle columns with tf in middle: cross_div_{tf}_lag1, etc.
            elif f"_{tf}_" in col:
                latest[col] = float(last_row[col])

        # Add d_slope_f/b specific KEEP features
        for col in [f"d_slope_f_{tf}", f"d_slope_b_{tf}"]:
            if col in fm.columns:
                latest[col] = float(last_row[col])

        # Add internal slope_f_slow for Phase D
        sf_col = f"slope_f_mag_{tf}"
        if sf_col in fm.columns:
            latest[f"_slope_f_slow_{tf}"] = float(last_row[sf_col])

    # Store cross-TF features (not per-TF)
    cross_tf_cols = [
        "cross_tf_slope_agreement", "cross_tf_weighted_slope",
        "xtf_total_crosses", "xtf_total_long", "xtf_total_short",
        "xtf_total_converging", "xtf_tfs_with_crosses",
        "xtf_young_crosses", "xtf_adult_crosses",
        "xtf_balzak_crosses", "xtf_gran_crosses",
        "xtf_cascade_score", "xtf_direction_net", "xtf_direction_agreement",
        "xtf_reversal_total", "xtf_reversal_confirmed",
        "xtf_convergence_momentum", "xtf_corr_agreement",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_ny_session",
        "convergence_volume", "crossing_atr", "cascade_volume",
        "reversal_conviction",
    ]
    # These will be recomputed in compute_row, but store for completeness

    n_initialized = sum(
        1 for tf in TIMEFRAME_ORDER
        if state["decomposed"][tf]["initialized"]
    )
    logger.info(f"  State initialized: {n_initialized}/11 TFs from feature matrix")
    return state
