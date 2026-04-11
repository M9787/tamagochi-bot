"""Incremental V10 feature encoder — computes ONE feature row from persisted state.

Matches encode_v3.py, encode_v5.py, encode_v10.py exactly.
Each formula includes a source reference comment.

State structure (~18 KB JSON):
  - Per-TF decomposed state: EMA, rolling buffers, lag values
  - Per-TF kline state: EMA, rolling buffers
  - Per-TF window state: angle buffers, crossing pair state
  - Latest computed feature values per TF (for merge_asof behavior)
  - Cross-TF state: prev_xtf_total_converging

Usage:
    encoder = IncrementalEncoder(state)
    features = encoder.compute_row(new_klines, new_decomposed, timestamp)
    save_state(encoder.state)
"""

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import TIMEFRAME_ORDER

logger = logging.getLogger(__name__)

# --- Constants (matching batch encoders) ---
W_FAST = 30
W_SLOW = 120
ALL_WINDOWS = [30, 60, 100, 120, 160]
CROSSING_PAIRS = [
    (30, 60), (30, 100), (60, 100), (60, 120),
    (100, 120), (100, 160), (120, 160),
]
ATR_TFS = ["5M", "1D", "4H"]
TOP_8_TFS = ["3D", "1D", "12H", "8H", "6H", "4H", "2H", "1H"]
TF_WEIGHTS = {
    "3D": 5, "1D": 4, "12H": 3.5, "8H": 3, "6H": 2.5,
    "4H": 2, "2H": 1.5, "1H": 1.2, "30M": 1.1, "15M": 1.0, "5M": 0.8,
}
TF_YOUNGS = ["5M", "15M", "30M"]
TF_ADULTS = ["1H", "2H", "4H"]
TF_BALZAKS = ["6H", "8H", "12H"]
TF_GRANS = ["1D", "3D"]

# KEEP list specifics
KEEP_D_SLOPE_F_TFS = {"3D", "8H"}  # d_slope_f_3D, d_slope_f_8H
KEEP_D_SLOPE_B_TFS = {"1D", "3D"}  # d_slope_b_1D, d_slope_b_3D

# Buffer sizes
D_SLOPE_F_BUF_SIZE = 23   # for cs_dsf_w23
CORR_BUF_SIZE = 10        # for corr_stability
WICK_BUF_SIZE = 10         # for wick_asymmetry
VOL_BUF_SIZE = 50          # for vol_impulse
BODY_PCT_BUF_SIZE = 20     # for norm_body_accum
STOCH_WINDOWS = [10, 20, 50]
UP_BAR_BUF_SIZE = 50       # for up_bar_ratio_3D_w50
REVERSAL_WINDOW = 5        # for 5-point reversal pattern

# Phase G: Bollinger Band (3, 35) extreme features — matching encode_v10.py:488-490
BB_TFS = {"5M", "15M", "1H", "4H", "1D"}
BB_PERIOD = 35
BB_STD = 3


class IncrementalEncoder:
    """Compute one V10 feature row incrementally, matching batch encoding exactly."""

    def __init__(self, state: dict | None = None):
        if state is None:
            self.state = self._empty_state()
        else:
            self.state = state

    # =================================================================
    # State Structure
    # =================================================================

    @classmethod
    def _empty_state(cls):
        state = {
            "version": 1,
            "last_timestamp": None,
            "decomposed": {},
            "klines": {},
            "windows": {},
            "latest": {},
            "prev_xtf_total_converging": 0.0,
        }
        for tf in TIMEFRAME_ORDER:
            state["decomposed"][tf] = cls._empty_tf_decomposed()
            state["klines"][tf] = cls._empty_tf_klines()
            state["windows"][tf] = cls._empty_tf_windows()
            state["latest"][tf] = {}
        return state

    @staticmethod
    def _empty_tf_decomposed():
        return {
            "sf_fast": 0.0, "sf_slow": 0.0, "sb_slow": 0.0,
            "angle_fast": 0.0, "angle_slow": 0.0,
            "corr_slow": 0.0, "accel_slow": 0.0, "pf_slow": 1.0,
            "prev_sf_fast": 0.0, "prev_sf_slow": 0.0, "prev_sb_slow": 0.0,
            "prev_angle_fast": 0.0, "prev_angle_slow": 0.0,
            "prev_corr_slow": 0.0,
            "prev_cross_div": 0.0, "prev2_cross_div": 0.0,
            "prev_cross_dir": 0, "prev_trend_certainty": 0.0,
            "d_sf_buf": [], "corr_buf": [],
            "ema_momentum_5": 0.0,
            "initialized": False,
        }

    @staticmethod
    def _empty_tf_klines():
        state = {
            "ema_vol50": 0.0, "ema_vbp10": 0.0, "prev_close": 0.0,
            "wick_buf": [], "vol_buf": [], "body_pct_buf": [],
            "body_buf_10": [], "body_buf_50": [],
            "up_bar_buf_50": [],
            "atr": 0.0, "initialized": False,
            "bb_close_buf": [],  # Phase G: rolling close for BB (maxlen=BB_PERIOD)
            "bb_low_buf": [],    # Phase G: rolling low for BB (maxlen=BB_PERIOD)
        }
        for w in STOCH_WINDOWS:
            state[f"high_buf_{w}"] = []
            state[f"low_buf_{w}"] = []
            state[f"close_buf_{w}"] = []
        return state

    @staticmethod
    def _empty_tf_windows():
        windows = {}
        for ws in ALL_WINDOWS:
            windows[str(ws)] = {
                "angle": 0.0, "slope_f": 0.0,
                "prev_angle": 0.0,
                "angle_buf": [],
            }
        pairs = {}
        for ws1, ws2 in CROSSING_PAIRS:
            pairs[f"{ws1}_{ws2}"] = {"prev_diff": 0.0, "prev_abs_diff": 0.0}
        return {"ws": windows, "pairs": pairs, "initialized": False}

    # =================================================================
    # Utility Methods
    # =================================================================

    @staticmethod
    def _ema_update(prev_ema, new_value, span):
        """pandas ewm(span=S, min_periods=1).mean() update rule.
        Source: encode_v10.py:274, 354, 361
        """
        alpha = 2.0 / (span + 1)
        return alpha * new_value + (1 - alpha) * prev_ema

    @staticmethod
    def _buf_append(buf, value, max_len):
        """Append to list buffer, maintaining max length."""
        buf.append(value)
        if len(buf) > max_len:
            buf.pop(0)

    @staticmethod
    def _rolling_sum(buf):
        return sum(buf) if buf else 0.0

    @staticmethod
    def _rolling_mean(buf):
        return sum(buf) / len(buf) if buf else 0.0

    @staticmethod
    def _rolling_std(buf):
        """Sample std (ddof=1), matching pandas rolling.std().
        Source: encode_v10.py:259 (corr_stability)
        """
        n = len(buf)
        if n < 2:
            return 0.0
        mean = sum(buf) / n
        variance = sum((x - mean) ** 2 for x in buf) / (n - 1)
        return math.sqrt(max(variance, 0.0))

    @staticmethod
    def _sign(x):
        """math.copysign-based sign matching numpy.sign behavior."""
        if x > 0:
            return 1.0
        elif x < 0:
            return -1.0
        return 0.0

    # =================================================================
    # TF Update: Decomposed (Phase A KEEP + Phase B + Phase F3)
    # =================================================================

    def _update_decomposed(self, tf, row_w30, row_w120):
        """Update decomposed state and compute features for one TF.

        Computes Phase A KEEP (decomposed-origin) + Phase B + F3 corr_velocity.
        Source: encode_v3.py:146-303, encode_v10.py:199-314
        """
        ds = self.state["decomposed"][tf]
        feats = {}

        # Shift current → previous
        ds["prev_sf_fast"] = ds["sf_fast"]
        ds["prev_sf_slow"] = ds["sf_slow"]
        ds["prev_sb_slow"] = ds["sb_slow"]
        ds["prev_angle_fast"] = ds["angle_fast"]
        ds["prev_angle_slow"] = ds["angle_slow"]
        ds["prev_corr_slow"] = ds["corr_slow"]

        # Load new values
        ds["sf_fast"] = float(row_w30.get("slope_f", 0))
        ds["sf_slow"] = float(row_w120.get("slope_f", 0))
        ds["sb_slow"] = float(row_w120.get("slope_b", 0))
        ds["angle_fast"] = float(row_w30.get("angle", 0))
        ds["angle_slow"] = float(row_w120.get("angle", 0))
        ds["corr_slow"] = float(row_w120.get("corr", 0))
        ds["accel_slow"] = float(row_w120.get("acceleration", 0))
        ds["pf_slow"] = float(row_w120.get("p_value_f", 1.0))

        sf_slow = ds["sf_slow"]
        sb_slow = ds["sb_slow"]
        sign_sf = self._sign(sf_slow)
        sign_sb = self._sign(sb_slow)

        # --- Phase A KEEP: V3 features ---

        # cross_div = angle_fast - angle_slow  (encode_v3.py:187)
        cross_div = ds["angle_fast"] - ds["angle_slow"]
        feats[f"cross_div_{tf}"] = cross_div

        # slope_f_mag = slope_f_w120  (encode_v3.py:181)
        feats[f"slope_f_mag_{tf}"] = sf_slow

        # angle_slow = angle_w120  (encode_v3.py:185)
        feats[f"angle_slow_{tf}"] = ds["angle_slow"]

        # corr_slow = corr_w120  (encode_v3.py:183)
        feats[f"corr_slow_{tf}"] = ds["corr_slow"]

        # slope_b_slow = slope_b_w120  (encode_v3.py:182)
        feats[f"slope_b_slow_{tf}"] = sb_slow

        # cross_div_lag1/lag2  (encode_v3.py:229-232)
        feats[f"cross_div_{tf}_lag1"] = ds["prev_cross_div"]
        feats[f"cross_div_{tf}_lag2"] = ds["prev2_cross_div"]

        # cross_traj = diff(cross_div)  (encode_v3.py:190-191)
        feats[f"cross_traj_{tf}"] = cross_div - ds["prev_cross_div"]

        # accel_raw = acceleration_w120  (encode_v3.py:193)
        feats[f"accel_raw_{tf}"] = ds["accel_slow"]

        # accel_mag = |accel_raw|  (encode_v3.py:194)
        feats[f"accel_mag_{tf}"] = abs(ds["accel_slow"])

        # cross_dir — opposing derivative crossing  (encode_v3.py:197-207)
        d_fast = ds["angle_fast"] - ds["prev_angle_fast"]
        d_slow = ds["angle_slow"] - ds["prev_angle_slow"]
        opposing = (d_fast * d_slow) < 0
        if opposing and d_slow > 0:
            cross_dir = 1
        elif opposing and d_slow < 0:
            cross_dir = -1
        else:
            cross_dir = 0
        feats[f"cross_dir_{tf}"] = cross_dir

        # cross_dir_lag1  (encode_v3.py:234-235)
        feats[f"cross_dir_{tf}_lag1"] = ds["prev_cross_dir"]

        # d_slope_f and buffer for cs_dsf  (encode_v3.py:210-211, 225-226)
        d_sf = sf_slow - ds["prev_sf_slow"]
        self._buf_append(ds["d_sf_buf"], d_sf, D_SLOPE_F_BUF_SIZE)

        # cs_dsf_w23 = rolling_sum(d_slope_f, 23)  (encode_v3.py:226)
        feats[f"cs_dsf_{tf}_w23"] = self._rolling_sum(ds["d_sf_buf"])

        # d_slope_f for KEEP TFs  (KEEP: d_slope_f_3D, d_slope_f_8H)
        if tf in KEEP_D_SLOPE_F_TFS:
            feats[f"d_slope_f_{tf}"] = d_sf

        # d_slope_b for KEEP TFs  (KEEP: d_slope_b_1D, d_slope_b_3D)
        if tf in KEEP_D_SLOPE_B_TFS:
            d_sb = sb_slow - ds["prev_sb_slow"]
            feats[f"d_slope_b_{tf}"] = d_sb

        # Update lag state
        ds["prev2_cross_div"] = ds["prev_cross_div"]
        ds["prev_cross_div"] = cross_div
        ds["prev_cross_dir"] = cross_dir

        # --- Phase B features (encode_v10.py:199-314) ---

        # trend_certainty = -log10(clip(p_value_f, 1e-15, 1.0))  (encode_v10.py:248-250)
        pf = max(min(ds["pf_slow"], 1.0), 1e-15)
        trend_certainty = -math.log10(pf)
        feats[f"trend_certainty_{tf}"] = trend_certainty

        # regime_quadrant = sign(sf)*2 + sign(sb)  (encode_v10.py:253)
        feats[f"regime_quadrant_{tf}"] = sign_sf * 2 + sign_sb

        # corr_regime = corr * sign(sf)  (encode_v10.py:256)
        feats[f"corr_regime_{tf}"] = ds["corr_slow"] * sign_sf

        # corr_stability = rolling_std(corr, 10)  (encode_v10.py:259-260)
        self._buf_append(ds["corr_buf"], ds["corr_slow"], CORR_BUF_SIZE)
        feats[f"corr_stability_{tf}"] = self._rolling_std(ds["corr_buf"])

        # slope_sign_gradient = sign(sf_fast) - sign(sf_slow)  (encode_v10.py:263)
        feats[f"slope_sign_gradient_{tf}"] = self._sign(ds["sf_fast"]) - sign_sf

        # angle_regime = min(angle, 30) * sign(sf)  (encode_v10.py:266)
        feats[f"angle_regime_{tf}"] = min(ds["angle_slow"], 30.0) * sign_sf

        # regime_change_strength  (encode_v10.py:269-271)
        if sign_sf != sign_sb:
            feats[f"regime_change_strength_{tf}"] = abs(sf_slow - sb_slow) * trend_certainty
        else:
            feats[f"regime_change_strength_{tf}"] = 0.0

        # smoothed_momentum = ema(sf - sb, span=5)  (encode_v10.py:274-275)
        momentum = sf_slow - sb_slow
        if ds["initialized"]:
            ds["ema_momentum_5"] = self._ema_update(ds["ema_momentum_5"], momentum, 5)
        else:
            ds["ema_momentum_5"] = momentum
        feats[f"smoothed_momentum_{tf}"] = ds["ema_momentum_5"]

        # accel_signed = accel * sign(sf)  (encode_v10.py:278)
        feats[f"accel_signed_{tf}"] = ds["accel_slow"] * sign_sf

        # p_value_trend = tc - tc_lag1  (encode_v10.py:281-283)
        feats[f"p_value_trend_{tf}"] = trend_certainty - ds["prev_trend_certainty"]
        ds["prev_trend_certainty"] = trend_certainty

        # Store slope_f_slow for Phase D (internal, prefixed with _)
        feats[f"_slope_f_slow_{tf}"] = sf_slow

        # --- Phase F3: corr_velocity (encode_v10.py:756-758) ---
        feats[f"corr_velocity_{tf}"] = ds["corr_slow"] - ds["prev_corr_slow"]

        ds["initialized"] = True
        self.state["latest"][tf].update(feats)

    # =================================================================
    # TF Update: Klines (Phase A KEEP kline + Phase C)
    # =================================================================

    def _update_klines(self, tf, kline_row):
        """Update kline state and compute features for one TF.

        Computes Phase A KEEP (kline-origin) + Phase C features.
        Source: encode_v5.py:59-143, encode_v10.py:321-417
        """
        ks = self.state["klines"][tf]
        feats = {}

        o = float(kline_row["Open"])
        h = float(kline_row["High"])
        lo = float(kline_row["Low"])
        c = float(kline_row["Close"])
        v = float(kline_row["Volume"])

        body = c - o
        body_sign = self._sign(body)
        up_bar = 1.0 if c > o else 0.0

        # --- Phase A KEEP: V5 cumsum_body (encode_v5.py:111-114) ---
        self._buf_append(ks["body_buf_10"], body, 10)
        self._buf_append(ks["body_buf_50"], body, 50)
        feats[f"cumsum_body_{tf}_w10"] = self._rolling_sum(ks["body_buf_10"])
        feats[f"cumsum_body_{tf}_w50"] = self._rolling_sum(ks["body_buf_50"])

        # --- Phase A KEEP: V5 stoch_pos (encode_v5.py:120-125) ---
        for w in STOCH_WINDOWS:
            self._buf_append(ks[f"high_buf_{w}"], h, w)
            self._buf_append(ks[f"low_buf_{w}"], lo, w)
            self._buf_append(ks[f"close_buf_{w}"], c, w)

            roll_min = min(ks[f"low_buf_{w}"])
            roll_max = max(ks[f"high_buf_{w}"])
            range_val = roll_max - roll_min
            stoch = (c - roll_min) / range_val if range_val > 0 else 0.5
            feats[f"stoch_pos_{tf}_w{w}"] = stoch

        # --- Phase A KEEP: up_bar_ratio_3D_w50 (encode_v5.py:117-118) ---
        if tf == "3D":
            self._buf_append(ks["up_bar_buf_50"], up_bar, UP_BAR_BUF_SIZE)
            feats["up_bar_ratio_3D_w50"] = self._rolling_mean(ks["up_bar_buf_50"])

        # --- Phase C: vol_ratio (encode_v10.py:353-355) ---
        if ks["initialized"]:
            ks["ema_vol50"] = self._ema_update(ks["ema_vol50"], v, 50)
        else:
            ks["ema_vol50"] = v
        vol_ratio = v / ks["ema_vol50"] if ks["ema_vol50"] > 0 else 1.0
        feats[f"vol_ratio_{tf}"] = vol_ratio

        # vol_body_product = sign(C-O) * vol_ratio  (encode_v10.py:358)
        vol_body_product = body_sign * vol_ratio
        feats[f"vol_body_product_{tf}"] = vol_body_product

        # directional_vol_body = ema(vol_body_product, 10)  (encode_v10.py:361-362)
        if ks["initialized"]:
            ks["ema_vbp10"] = self._ema_update(ks["ema_vbp10"], vol_body_product, 10)
        else:
            ks["ema_vbp10"] = vol_body_product
        feats[f"directional_vol_body_{tf}"] = ks["ema_vbp10"]

        # wick_asymmetry = rolling_mean(wick_raw, 10)  (encode_v10.py:364-373)
        body_top = max(c, o)
        body_bot = min(c, o)
        bar_range = h - lo
        eps = 1e-10
        wick_raw = ((h - body_top) - (body_bot - lo)) / (bar_range + eps)
        self._buf_append(ks["wick_buf"], wick_raw, WICK_BUF_SIZE)
        feats[f"wick_asymmetry_{tf}"] = self._rolling_mean(ks["wick_buf"])

        # vol_impulse = z-score(volume, 50)  (encode_v10.py:375-379)
        self._buf_append(ks["vol_buf"], v, VOL_BUF_SIZE)
        vol_mean = self._rolling_mean(ks["vol_buf"])
        vol_std = self._rolling_std(ks["vol_buf"])
        vol_std = vol_std if vol_std > 0 else 1.0
        feats[f"vol_impulse_{tf}"] = (v - vol_mean) / vol_std

        # norm_body_accum = rolling_sum(body_pct, 20)  (encode_v10.py:381-385)
        safe_close = c if c > 0 else 1.0
        body_pct = body_sign * abs(body) / safe_close
        self._buf_append(ks["body_pct_buf"], body_pct, BODY_PCT_BUF_SIZE)
        feats[f"norm_body_accum_{tf}"] = self._rolling_sum(ks["body_pct_buf"])

        # atr_normalized (only 5M, 1D, 4H)  (encode_v10.py:399-402)
        if tf in ATR_TFS:
            prev_c = ks["prev_close"]
            if ks["initialized"] and prev_c > 0:
                tr = max(h - lo, abs(h - prev_c), abs(lo - prev_c))
                alpha = 2.0 / (14 + 1)
                ks["atr"] = alpha * tr + (1 - alpha) * ks["atr"]
            elif not ks["initialized"]:
                ks["atr"] = h - lo
            atr_norm = ks["atr"] / c if c > 0 else 0.0
            feats[f"atr_normalized_{tf}"] = atr_norm

        # --- Phase G: Update BB buffers (encode_v10.py:519-524) ---
        if tf in BB_TFS:
            self._buf_append(ks["bb_close_buf"], c, BB_PERIOD)
            self._buf_append(ks["bb_low_buf"], lo, BB_PERIOD)

        ks["prev_close"] = c
        ks["initialized"] = True
        self.state["latest"][tf].update(feats)

    # =================================================================
    # TF Update: All-Windows (Phase F1 cross-window features)
    # =================================================================

    def _update_windows(self, tf, new_window_data):
        """Update window state and compute F1 features for one TF.

        new_window_data: {ws_int: {"angle": float, "slope_f": float}}
        Source: encode_v10.py:518-662
        """
        ws_state = self.state["windows"][tf]
        feats = {}

        # Update per-window state
        available_windows = []
        for ws in ALL_WINDOWS:
            ws_key = str(ws)
            if ws in new_window_data:
                row = new_window_data[ws]
                ws_s = ws_state["ws"][ws_key]
                ws_s["prev_angle"] = ws_s["angle"]
                ws_s["angle"] = float(row["angle"])
                ws_s["slope_f"] = float(row["slope_f"])
                self._buf_append(ws_s["angle_buf"], ws_s["angle"], REVERSAL_WINDOW)
                available_windows.append(ws)
            elif ws_state["ws"][ws_key].get("angle_buf"):
                available_windows.append(ws)

        if not available_windows:
            for feat in ["xw_crosses_active", "xw_crosses_long", "xw_crosses_short",
                         "xw_converging", "xw_reversal_count",
                         "xw_direction_agreement", "xw_cross_reversal"]:
                feats[f"{feat}_{tf}"] = 0.0
            self.state["latest"][tf].update(feats)
            return

        # Crossing detection  (encode_v10.py:533-558)
        crosses_active = 0
        crosses_long = 0
        crosses_short = 0
        converging = 0

        for ws1, ws2 in CROSSING_PAIRS:
            if ws1 not in available_windows or ws2 not in available_windows:
                continue

            pair_key = f"{ws1}_{ws2}"
            pair_s = ws_state["pairs"][pair_key]

            a1 = ws_state["ws"][str(ws1)]["angle"]
            a2 = ws_state["ws"][str(ws2)]["angle"]
            curr_diff = a1 - a2
            prev_diff = pair_s["prev_diff"]
            prev_abs_diff = pair_s["prev_abs_diff"]

            # Crossing: sign change in angle difference  (encode_v10.py:544)
            if prev_diff * curr_diff < 0:
                crosses_active += 1
                # Direction from derivatives  (encode_v10.py:547-552)
                d1 = ws_state["ws"][str(ws1)]["angle"] - ws_state["ws"][str(ws1)]["prev_angle"]
                d2 = ws_state["ws"][str(ws2)]["angle"] - ws_state["ws"][str(ws2)]["prev_angle"]
                if d1 < 0 and d2 > 0:
                    crosses_long += 1    # younger falling + elder rising
                elif d1 > 0 and d2 < 0:
                    crosses_short += 1   # younger rising + elder falling

            # Converging: |diff| decreasing  (encode_v10.py:555-556)
            if abs(curr_diff) < prev_abs_diff:
                converging += 1

            pair_s["prev_diff"] = curr_diff
            pair_s["prev_abs_diff"] = abs(curr_diff)

        feats[f"xw_crosses_active_{tf}"] = float(crosses_active)
        feats[f"xw_crosses_long_{tf}"] = float(crosses_long)
        feats[f"xw_crosses_short_{tf}"] = float(crosses_short)
        feats[f"xw_converging_{tf}"] = float(converging)

        # Reversal count  (encode_v10.py:561-582, 617-619)
        rev_count = 0
        for ws in available_windows:
            buf = ws_state["ws"][str(ws)]["angle_buf"]
            if len(buf) >= REVERSAL_WINDOW:
                w = buf[-REVERSAL_WINDOW:]
                # BOTTOM: a>b>c>d<e  (encode_v10.py:576-577)
                if w[0] > w[1] > w[2] > w[3] and w[3] < w[4]:
                    rev_count += 1
                # PEAK: a<b<c<d>e  (encode_v10.py:579-580)
                elif w[0] < w[1] < w[2] < w[3] and w[3] > w[4]:
                    rev_count += 1
        feats[f"xw_reversal_count_{tf}"] = float(rev_count)

        # Direction agreement  (encode_v10.py:622-634)
        n_windows = len(available_windows)
        if n_windows > 0:
            signs = [self._sign(ws_state["ws"][str(ws)]["slope_f"])
                     for ws in available_windows]
            sign_sum = sum(signs)
            mode_sign = self._sign(sign_sum)
            if mode_sign == 0.0:
                agreement = 0.5
            else:
                agreement = sum(1.0 for s in signs if s == mode_sign) / n_windows
        else:
            agreement = 0.5
        feats[f"xw_direction_agreement_{tf}"] = agreement

        # Cross-reversal overlap  (encode_v10.py:637)
        feats[f"xw_cross_reversal_{tf}"] = 1.0 if (crosses_active > 0 and rev_count > 0) else 0.0

        ws_state["initialized"] = True
        self.state["latest"][tf].update(feats)

    # =================================================================
    # Main: compute_row
    # =================================================================

    def compute_row(self, new_klines, new_decomposed, timestamp):
        """Compute one V10 feature row incrementally.

        Args:
            new_klines: {tf: {Open, High, Low, Close, Volume}} for updated TFs
            new_decomposed: {(tf, ws): {slope_f, slope_b, angle, corr,
                             acceleration, p_value_f}} for updated combos
            timestamp: pd.Timestamp of the 5M candle

        Returns:
            pd.Series with 518 named feature values
        """
        # Step 1: Update state for TFs with new data
        for tf in TIMEFRAME_ORDER:
            has_w30 = (tf, W_FAST) in new_decomposed
            has_w120 = (tf, W_SLOW) in new_decomposed
            if has_w30 and has_w120:
                self._update_decomposed(
                    tf, new_decomposed[(tf, W_FAST)], new_decomposed[(tf, W_SLOW)])

            if tf in new_klines:
                self._update_klines(tf, new_klines[tf])

            # Update all-window state
            window_data = {}
            for ws in ALL_WINDOWS:
                if (tf, ws) in new_decomposed:
                    window_data[ws] = new_decomposed[(tf, ws)]
            if window_data:
                self._update_windows(tf, window_data)

        # Step 2: Assemble features from latest stored values
        features = {}
        slope_f_per_tf = {}

        for tf in TIMEFRAME_ORDER:
            latest = self.state["latest"][tf]
            for key, val in latest.items():
                if not key.startswith("_"):  # skip internal keys
                    features[key] = val
            # Collect slope_f_slow for Phase D
            sf_key = f"_slope_f_slow_{tf}"
            if sf_key in latest:
                slope_f_per_tf[tf] = latest[sf_key]

        # --- Phase D: Cross-TF summaries (encode_v10.py:424-456) ---

        # cross_tf_slope_agreement  (encode_v10.py:432-441)
        sign_sum = 0.0
        count = 0
        for tf in TOP_8_TFS:
            if tf in slope_f_per_tf:
                sign_sum += self._sign(slope_f_per_tf[tf])
                count += 1
        features["cross_tf_slope_agreement"] = sign_sum / count if count > 0 else 0.0

        # cross_tf_weighted_slope  (encode_v10.py:444-454)
        weighted_sum = 0.0
        weight_total = 0.0
        for tf in TIMEFRAME_ORDER:
            if tf in slope_f_per_tf:
                w = TF_WEIGHTS.get(tf, 1.0)
                weighted_sum += slope_f_per_tf[tf] * w
                weight_total += w
        features["cross_tf_weighted_slope"] = (
            weighted_sum / weight_total if weight_total > 0 else 0.0)

        # --- Phase E: range_position_signed (encode_v10.py:463-481) ---
        for tf in TIMEFRAME_ORDER:
            stoch = features.get(f"stoch_pos_{tf}_w20", 0.5)
            slope_f = features.get(f"slope_f_mag_{tf}", 0.0)
            features[f"range_position_signed_{tf}"] = (
                (2 * stoch - 1) * self._sign(slope_f))

        # --- Phase F2: Cross-TF composites (encode_v10.py:669-737) ---

        # F2-1..4: Totals  (encode_v10.py:673-682)
        for feat_src, feat_dst in [
            ("xw_crosses_active", "xtf_total_crosses"),
            ("xw_crosses_long", "xtf_total_long"),
            ("xw_crosses_short", "xtf_total_short"),
            ("xw_converging", "xtf_total_converging"),
        ]:
            total = 0.0
            for tf in TIMEFRAME_ORDER:
                total += features.get(f"{feat_src}_{tf}", 0.0)
            features[feat_dst] = total

        # F2-5: tfs_with_crosses  (encode_v10.py:685-688)
        tfs_with = 0.0
        for tf in TIMEFRAME_ORDER:
            if features.get(f"xw_crosses_active_{tf}", 0.0) >= 1:
                tfs_with += 1.0
        features["xtf_tfs_with_crosses"] = tfs_with

        # F2-6..9: Group crosses  (encode_v10.py:691-700)
        for group_name, tf_list in [
            ("xtf_young_crosses", TF_YOUNGS),
            ("xtf_adult_crosses", TF_ADULTS),
            ("xtf_balzak_crosses", TF_BALZAKS),
            ("xtf_gran_crosses", TF_GRANS),
        ]:
            total = 0.0
            for tf in tf_list:
                total += features.get(f"xw_crosses_active_{tf}", 0.0)
            features[group_name] = total

        # F2-10: cascade_score  (encode_v10.py:703-708)
        features["xtf_cascade_score"] = (
            features["xtf_young_crosses"] * 1 +
            features["xtf_adult_crosses"] * 2 +
            features["xtf_balzak_crosses"] * 3 +
            features["xtf_gran_crosses"] * 4
        )

        # F2-11: direction_net  (encode_v10.py:711-713)
        features["xtf_direction_net"] = (
            features["xtf_total_long"] - features["xtf_total_short"])

        # F2-12: direction_agreement  (encode_v10.py:716-719)
        total_dir = features["xtf_total_long"] + features["xtf_total_short"] + 1e-10
        features["xtf_direction_agreement"] = (
            abs(features["xtf_direction_net"]) / total_dir)

        # F2-13: reversal_total  (encode_v10.py:722-725)
        rev_total = 0.0
        for tf in TIMEFRAME_ORDER:
            rev_total += features.get(f"xw_reversal_count_{tf}", 0.0)
        features["xtf_reversal_total"] = rev_total

        # F2-14: reversal_confirmed  (encode_v10.py:728-731)
        rev_confirmed = 0.0
        for tf in TIMEFRAME_ORDER:
            if features.get(f"xw_cross_reversal_{tf}", 0.0) >= 1:
                rev_confirmed += 1.0
        features["xtf_reversal_confirmed"] = rev_confirmed

        # F2-15: convergence_momentum  (encode_v10.py:734-737)
        conv_total = features["xtf_total_converging"]
        features["xtf_convergence_momentum"] = (
            conv_total - self.state["prev_xtf_total_converging"])
        self.state["prev_xtf_total_converging"] = conv_total

        # --- Phase F3: xtf_corr_agreement (encode_v10.py:776-781) ---
        corr_signs = []
        for tf in TIMEFRAME_ORDER:
            cv_key = f"corr_velocity_{tf}"
            # Use corr_slow sign, not velocity — matching encode_v10.py:763
            cs_key = f"corr_slow_{tf}"
            corr_val = features.get(cs_key, 0.0)
            corr_signs.append(self._sign(corr_val))
        features["xtf_corr_agreement"] = (
            sum(corr_signs) / len(corr_signs) if corr_signs else 0.0)

        # --- Phase F4: Temporal features (encode_v10.py:784-793) ---
        ts = pd.Timestamp(timestamp)
        hour = ts.hour + ts.minute / 60.0
        dow = float(ts.dayofweek)

        features["hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
        features["hour_cos"] = math.cos(2 * math.pi * hour / 24.0)
        features["dow_sin"] = math.sin(2 * math.pi * dow / 7.0)
        features["dow_cos"] = math.cos(2 * math.pi * dow / 7.0)
        features["is_ny_session"] = 1.0 if 13 <= ts.hour <= 21 else 0.0

        # --- Phase F5: Interaction features (encode_v10.py:798-832) ---

        # convergence_volume = xtf_total_crosses * vol_ratio_1D  (encode_v10.py:814-815)
        features["convergence_volume"] = (
            features.get("xtf_total_crosses", 0.0) *
            features.get("vol_ratio_1D", 0.0))

        # crossing_atr = xtf_tfs_with_crosses * atr_normalized_1D  (encode_v10.py:819-820)
        features["crossing_atr"] = (
            features.get("xtf_tfs_with_crosses", 0.0) *
            features.get("atr_normalized_1D", 0.0))

        # cascade_volume = xtf_cascade_score * vol_body_product_1D  (encode_v10.py:823-824)
        features["cascade_volume"] = (
            features.get("xtf_cascade_score", 0.0) *
            features.get("vol_body_product_1D", 0.0))

        # reversal_conviction = xtf_reversal_confirmed * vol_ratio_1D  (encode_v10.py:829-830)
        features["reversal_conviction"] = (
            features.get("xtf_reversal_confirmed", 0.0) *
            features.get("vol_ratio_1D", 0.0))

        # --- Phase G: Bollinger Band extreme features (encode_v10.py:493-543) ---
        for tf in BB_TFS:
            ks = self.state["klines"][tf]
            bb_close_buf = ks["bb_close_buf"]
            bb_low_buf = ks["bb_low_buf"]

            if len(bb_close_buf) >= 1:
                # SMA and std matching pandas rolling(BB_PERIOD, min_periods=1)
                sma = sum(bb_close_buf) / len(bb_close_buf)
                n_buf = len(bb_close_buf)
                if n_buf >= 2:
                    variance = sum((x - sma) ** 2 for x in bb_close_buf) / (n_buf - 1)
                    std = math.sqrt(max(variance, 0.0))
                else:
                    std = 0.0  # pandas rolling().std() returns NaN for n=1, fillna(0) -> 0

                bb_lower = sma - BB_STD * std
                bb_upper = sma + BB_STD * std

                close_val = bb_close_buf[-1]
                low_val = bb_low_buf[-1]
                safe_close = close_val if close_val > 0 else 1.0

                features[f"bb_lower_pierce_{tf}"] = (bb_lower - low_val) / safe_close
                features[f"bb_upper_dist_{tf}"] = (bb_upper - close_val) / safe_close
            else:
                features[f"bb_lower_pierce_{tf}"] = 0.0
                features[f"bb_upper_dist_{tf}"] = 0.0

        # --- Fill missing features with 0 ---
        # Remove internal keys
        features = {k: v for k, v in features.items() if not k.startswith("_")}

        self.state["last_timestamp"] = str(timestamp)
        return pd.Series(features)

    # =================================================================
    # State I/O
    # =================================================================

    MAX_BACKUPS = 3

    def save_state(self, path: Path):
        """Save state to JSON with backup rotation.

        Before writing, rotates: state.json → .bak1 → .bak2 → .bak3.
        Keeps MAX_BACKUPS backup files for corruption recovery.
        Uses temp file + atomic rename to prevent partial writes.
        """
        import os
        import tempfile

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Rotate backups: .bak3 → delete, .bak2 → .bak3, .bak1 → .bak2, current → .bak1
        for i in range(self.MAX_BACKUPS, 0, -1):
            src = path.with_suffix(f".json.bak{i}") if i > 1 else path
            if i == 1:
                src = path
            else:
                src = path.with_suffix(f".json.bak{i - 1}")
            dst = path.with_suffix(f".json.bak{i}")
            if src.exists():
                try:
                    os.replace(str(src), str(dst))
                except OSError:
                    pass

        # Atomic write: temp file + rename
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp")
        os.close(tmp_fd)
        try:
            with open(tmp_path, "w") as f:
                json.dump(self.state, f)
            os.replace(tmp_path, str(path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    @classmethod
    def _validate_state(cls, state: dict) -> bool:
        """Check that state dict has required structure."""
        try:
            if not isinstance(state, dict):
                return False
            if state.get("version") != 1:
                return False
            for key in ("decomposed", "klines", "windows", "latest"):
                if key not in state or not isinstance(state[key], dict):
                    return False
            # Check at least a few TFs exist
            if len(state["decomposed"]) < 5:
                return False
            return True
        except Exception:
            return False

    @classmethod
    def load_state(cls, path: Path) -> "IncrementalEncoder":
        """Load encoder from saved state with corruption recovery.

        Tries main file first, then .bak1, .bak2, .bak3 in order.
        Raises FileNotFoundError if all files are missing/corrupt.
        """
        path = Path(path)
        candidates = [path]
        for i in range(1, cls.MAX_BACKUPS + 1):
            candidates.append(path.with_suffix(f".json.bak{i}"))

        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                with open(candidate, "r") as f:
                    state = json.load(f)
                if cls._validate_state(state):
                    if candidate != path:
                        logger.warning(f"Primary state corrupt, recovered from {candidate.name}")
                    return cls(state=state)
                else:
                    logger.warning(f"State file {candidate.name} failed validation")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to read {candidate.name}: {e}")

        raise FileNotFoundError(
            f"No valid state file found at {path} or any backup")
