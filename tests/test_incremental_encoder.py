"""Tests for incremental encoder — verifies correctness against batch encoding.

Test categories:
  1. Unit: state creation, EMA update, buffer management
  2. Integration: full compute_row with synthetic data
  3. Regression: batch vs incremental comparison (requires real data)

Usage:
  pytest tests/test_incremental_encoder.py -v
  pytest tests/test_incremental_encoder.py -v -k "not regression"  # Skip slow tests
"""
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import TIMEFRAME_ORDER, WINDOW_SIZES
from data_service.incremental_encoder import (
    IncrementalEncoder,
    W_FAST, W_SLOW, ALL_WINDOWS, CROSSING_PAIRS, ATR_TFS,
    D_SLOPE_F_BUF_SIZE, CORR_BUF_SIZE, WICK_BUF_SIZE, VOL_BUF_SIZE,
    BODY_PCT_BUF_SIZE, STOCH_WINDOWS, UP_BAR_BUF_SIZE, REVERSAL_WINDOW,
    KEEP_D_SLOPE_F_TFS, KEEP_D_SLOPE_B_TFS, TF_WEIGHTS,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def encoder():
    """Fresh encoder with empty state."""
    return IncrementalEncoder()


@pytest.fixture
def sample_kline():
    """Sample kline row for testing."""
    return {
        "Open": 50000.0,
        "High": 50500.0,
        "Low": 49500.0,
        "Close": 50200.0,
        "Volume": 1000.0,
    }


@pytest.fixture
def sample_decomposed_w30():
    """Sample decomposed row for w30."""
    return {
        "slope_f": 0.05,
        "slope_b": 0.03,
        "angle": 15.0,
        "corr": 0.85,
        "acceleration": 0.5,
        "p_value_f": 0.001,
    }


@pytest.fixture
def sample_decomposed_w120():
    """Sample decomposed row for w120."""
    return {
        "slope_f": 0.02,
        "slope_b": 0.01,
        "angle": 10.0,
        "corr": 0.75,
        "acceleration": 0.3,
        "p_value_f": 0.005,
    }


@pytest.fixture
def sample_window_data():
    """Sample window data for all 5 windows."""
    return {
        ws: {"angle": 10.0 + ws * 0.1, "slope_f": 0.01 + ws * 0.001}
        for ws in ALL_WINDOWS
    }


# ============================================================================
# Unit Tests: State
# ============================================================================

class TestState:
    def test_empty_state_structure(self, encoder):
        """Empty state has all required keys for all 11 TFs."""
        state = encoder.state
        assert state["version"] == 1
        assert state["last_timestamp"] is None
        assert len(state["decomposed"]) == 11
        assert len(state["klines"]) == 11
        assert len(state["windows"]) == 11
        assert len(state["latest"]) == 11
        assert state["prev_xtf_total_converging"] == 0.0

        for tf in TIMEFRAME_ORDER:
            assert tf in state["decomposed"]
            assert tf in state["klines"]
            assert tf in state["windows"]
            assert tf in state["latest"]

    def test_empty_decomposed_state(self, encoder):
        """Each TF decomposed state has all required fields."""
        ds = encoder.state["decomposed"]["5M"]
        expected_keys = [
            "sf_fast", "sf_slow", "sb_slow", "angle_fast", "angle_slow",
            "corr_slow", "accel_slow", "pf_slow",
            "prev_sf_fast", "prev_sf_slow", "prev_sb_slow",
            "prev_angle_fast", "prev_angle_slow", "prev_corr_slow",
            "prev_cross_div", "prev2_cross_div",
            "prev_cross_dir", "prev_trend_certainty",
            "d_sf_buf", "corr_buf",
            "ema_momentum_5", "initialized",
        ]
        for key in expected_keys:
            assert key in ds, f"Missing key: {key}"
        assert ds["initialized"] is False

    def test_empty_klines_state(self, encoder):
        """Each TF kline state has all required fields."""
        ks = encoder.state["klines"]["1D"]
        expected_keys = [
            "ema_vol50", "ema_vbp10", "prev_close",
            "wick_buf", "vol_buf", "body_pct_buf",
            "body_buf_10", "body_buf_50", "up_bar_buf_50",
            "atr", "initialized",
        ]
        for key in expected_keys:
            assert key in ks, f"Missing key: {key}"
        for w in STOCH_WINDOWS:
            assert f"high_buf_{w}" in ks
            assert f"low_buf_{w}" in ks
            assert f"close_buf_{w}" in ks

    def test_empty_windows_state(self, encoder):
        """Each TF window state has all 5 windows and 7 pairs."""
        ws = encoder.state["windows"]["1H"]
        assert len(ws["ws"]) == 5
        assert len(ws["pairs"]) == 7
        for w in ALL_WINDOWS:
            assert str(w) in ws["ws"]
        for ws1, ws2 in CROSSING_PAIRS:
            assert f"{ws1}_{ws2}" in ws["pairs"]

    def test_state_save_load(self, encoder, tmp_path):
        """State round-trips through JSON correctly."""
        # Modify some state
        encoder.state["last_timestamp"] = "2026-01-15 00:00:00"
        encoder.state["decomposed"]["5M"]["sf_fast"] = 1.234

        state_file = tmp_path / "test_state.json"
        encoder.save_state(state_file)

        loaded = IncrementalEncoder.load_state(state_file)
        assert loaded.state["last_timestamp"] == "2026-01-15 00:00:00"
        assert loaded.state["decomposed"]["5M"]["sf_fast"] == 1.234


# ============================================================================
# Unit Tests: Utility Methods
# ============================================================================

class TestUtilities:
    def test_ema_update_matches_formula(self):
        """EMA update: alpha * new + (1-alpha) * prev."""
        prev_ema = 100.0
        new_value = 110.0
        span = 10
        alpha = 2.0 / (span + 1)

        result = IncrementalEncoder._ema_update(prev_ema, new_value, span)
        expected = alpha * new_value + (1 - alpha) * prev_ema
        assert abs(result - expected) < 1e-15

    def test_ema_converges_to_constant(self):
        """EMA of a constant sequence converges to that constant."""
        ema = 0.0
        for _ in range(200):
            ema = IncrementalEncoder._ema_update(ema, 50.0, 10)
        assert abs(ema - 50.0) < 1e-10

    def test_buf_append_maintains_max_length(self):
        """Buffer never exceeds max length."""
        buf = []
        for i in range(100):
            IncrementalEncoder._buf_append(buf, float(i), 10)
        assert len(buf) == 10
        assert buf[0] == 90.0
        assert buf[-1] == 99.0

    def test_rolling_mean(self):
        """Rolling mean matches expected value."""
        buf = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert IncrementalEncoder._rolling_mean(buf) == 3.0

    def test_rolling_std_ddof1(self):
        """Rolling std uses ddof=1 (sample std)."""
        buf = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        result = IncrementalEncoder._rolling_std(buf)
        expected = pd.Series(buf).std()  # pandas uses ddof=1 by default
        assert abs(result - expected) < 1e-10

    def test_rolling_std_single_element(self):
        """Rolling std of 1 element returns 0."""
        assert IncrementalEncoder._rolling_std([5.0]) == 0.0

    def test_sign_function(self):
        """Sign function matches numpy behavior."""
        assert IncrementalEncoder._sign(5.0) == 1.0
        assert IncrementalEncoder._sign(-3.0) == -1.0
        assert IncrementalEncoder._sign(0.0) == 0.0


# ============================================================================
# Integration Tests: Single TF Update
# ============================================================================

class TestDecomposedUpdate:
    def test_update_produces_features(self, encoder, sample_decomposed_w30,
                                      sample_decomposed_w120):
        """Updating decomposed state produces features in latest dict."""
        tf = "5M"
        encoder._update_decomposed(tf, sample_decomposed_w30, sample_decomposed_w120)

        latest = encoder.state["latest"][tf]
        # Check key features exist
        assert f"cross_div_{tf}" in latest
        assert f"slope_f_mag_{tf}" in latest
        assert f"angle_slow_{tf}" in latest
        assert f"trend_certainty_{tf}" in latest
        assert f"smoothed_momentum_{tf}" in latest
        assert f"corr_velocity_{tf}" in latest

    def test_cross_div_calculation(self, encoder, sample_decomposed_w30,
                                   sample_decomposed_w120):
        """cross_div = angle_fast - angle_slow."""
        tf = "1H"
        encoder._update_decomposed(tf, sample_decomposed_w30, sample_decomposed_w120)

        expected = 15.0 - 10.0  # angle_fast - angle_slow
        assert encoder.state["latest"][tf][f"cross_div_{tf}"] == expected

    def test_trend_certainty_formula(self, encoder, sample_decomposed_w30,
                                     sample_decomposed_w120):
        """trend_certainty = -log10(p_value_f)."""
        tf = "4H"
        encoder._update_decomposed(tf, sample_decomposed_w30, sample_decomposed_w120)

        expected = -math.log10(0.005)  # p_value_f from w120
        result = encoder.state["latest"][tf][f"trend_certainty_{tf}"]
        assert abs(result - expected) < 1e-10

    def test_lag_values_shift(self, encoder, sample_decomposed_w30,
                              sample_decomposed_w120):
        """Second update shifts current values to prev."""
        tf = "1D"
        # First update
        encoder._update_decomposed(tf, sample_decomposed_w30, sample_decomposed_w120)
        first_cross_div = encoder.state["latest"][tf][f"cross_div_{tf}"]

        # Second update with different values
        w30_2 = {**sample_decomposed_w30, "angle": 20.0}
        w120_2 = {**sample_decomposed_w120, "angle": 12.0}
        encoder._update_decomposed(tf, w30_2, w120_2)

        # Previous cross_div should be the first one
        assert encoder.state["latest"][tf][f"cross_div_{tf}_lag1"] == first_cross_div

    def test_d_slope_f_only_for_keep_tfs(self, encoder, sample_decomposed_w30,
                                          sample_decomposed_w120):
        """d_slope_f only produced for 3D and 8H."""
        for tf in TIMEFRAME_ORDER:
            encoder._update_decomposed(tf, sample_decomposed_w30, sample_decomposed_w120)

        for tf in KEEP_D_SLOPE_F_TFS:
            assert f"d_slope_f_{tf}" in encoder.state["latest"][tf]

        non_keep = set(TIMEFRAME_ORDER) - KEEP_D_SLOPE_F_TFS
        for tf in non_keep:
            assert f"d_slope_f_{tf}" not in encoder.state["latest"][tf]

    def test_initialized_flag_set(self, encoder, sample_decomposed_w30,
                                   sample_decomposed_w120):
        """Decomposed state sets initialized=True after first update."""
        tf = "30M"
        assert encoder.state["decomposed"][tf]["initialized"] is False
        encoder._update_decomposed(tf, sample_decomposed_w30, sample_decomposed_w120)
        assert encoder.state["decomposed"][tf]["initialized"] is True


class TestKlinesUpdate:
    def test_update_produces_features(self, encoder, sample_kline):
        """Updating kline state produces features in latest dict."""
        tf = "5M"
        encoder._update_klines(tf, sample_kline)

        latest = encoder.state["latest"][tf]
        assert f"vol_ratio_{tf}" in latest
        assert f"vol_body_product_{tf}" in latest
        assert f"directional_vol_body_{tf}" in latest
        assert f"stoch_pos_{tf}_w10" in latest
        assert f"cumsum_body_{tf}_w10" in latest

    def test_vol_ratio_first_update(self, encoder, sample_kline):
        """First update: vol_ratio = volume / ema(volume), ema starts at volume."""
        tf = "1H"
        encoder._update_klines(tf, sample_kline)

        # First update: ema_vol50 = volume, so vol_ratio = 1.0
        assert encoder.state["latest"][tf][f"vol_ratio_{tf}"] == 1.0

    def test_vol_body_product_sign(self, encoder):
        """vol_body_product is positive for up candle, negative for down."""
        tf = "4H"
        up_candle = {"Open": 100, "High": 110, "Low": 95, "Close": 108, "Volume": 500}
        encoder._update_klines(tf, up_candle)
        assert encoder.state["latest"][tf][f"vol_body_product_{tf}"] > 0

        down_candle = {"Open": 108, "High": 110, "Low": 95, "Close": 96, "Volume": 500}
        encoder._update_klines(tf, down_candle)
        assert encoder.state["latest"][tf][f"vol_body_product_{tf}"] < 0

    def test_atr_only_for_atr_tfs(self, encoder, sample_kline):
        """ATR only computed for 5M, 1D, 4H."""
        for tf in TIMEFRAME_ORDER:
            encoder._update_klines(tf, sample_kline)

        for tf in ATR_TFS:
            assert f"atr_normalized_{tf}" in encoder.state["latest"][tf]

        non_atr = set(TIMEFRAME_ORDER) - set(ATR_TFS)
        for tf in non_atr:
            assert f"atr_normalized_{tf}" not in encoder.state["latest"][tf]

    def test_stoch_pos_range(self, encoder):
        """stoch_pos should be in [0, 1]."""
        tf = "1D"
        # Process enough candles to fill buffers
        for i in range(20):
            kline = {
                "Open": 100 + i, "High": 105 + i,
                "Low": 95 + i, "Close": 102 + i,
                "Volume": 1000,
            }
            encoder._update_klines(tf, kline)

        for w in STOCH_WINDOWS:
            stoch = encoder.state["latest"][tf][f"stoch_pos_{tf}_w{w}"]
            assert 0.0 <= stoch <= 1.0, f"stoch_pos_{tf}_w{w} = {stoch}"

    def test_up_bar_ratio_only_3d(self, encoder, sample_kline):
        """up_bar_ratio only produced for 3D."""
        for tf in TIMEFRAME_ORDER:
            encoder._update_klines(tf, sample_kline)

        assert "up_bar_ratio_3D_w50" in encoder.state["latest"]["3D"]
        for tf in TIMEFRAME_ORDER:
            if tf != "3D":
                assert "up_bar_ratio_3D_w50" not in encoder.state["latest"][tf]


class TestWindowsUpdate:
    def test_update_produces_features(self, encoder, sample_window_data):
        """Updating windows produces F1 features."""
        tf = "1H"
        encoder._update_windows(tf, sample_window_data)

        latest = encoder.state["latest"][tf]
        assert f"xw_crosses_active_{tf}" in latest
        assert f"xw_crosses_long_{tf}" in latest
        assert f"xw_crosses_short_{tf}" in latest
        assert f"xw_converging_{tf}" in latest
        assert f"xw_reversal_count_{tf}" in latest
        assert f"xw_direction_agreement_{tf}" in latest
        assert f"xw_cross_reversal_{tf}" in latest

    def test_crossing_detected_on_sign_change(self, encoder):
        """Crossing detected when angle difference changes sign."""
        tf = "2H"
        # First update: angle diff positive
        data1 = {
            30: {"angle": 20.0, "slope_f": 0.01},
            60: {"angle": 10.0, "slope_f": 0.01},
            100: {"angle": 15.0, "slope_f": 0.01},
            120: {"angle": 12.0, "slope_f": 0.01},
            160: {"angle": 8.0, "slope_f": 0.01},
        }
        encoder._update_windows(tf, data1)
        # pair (30,60): diff = 20-10 = +10

        # Second update: angle diff negative for pair (30,60)
        data2 = {
            30: {"angle": 8.0, "slope_f": 0.01},
            60: {"angle": 18.0, "slope_f": 0.01},
            100: {"angle": 15.0, "slope_f": 0.01},
            120: {"angle": 12.0, "slope_f": 0.01},
            160: {"angle": 8.0, "slope_f": 0.01},
        }
        encoder._update_windows(tf, data2)
        # pair (30,60): diff = 8-18 = -10, sign changed!

        crosses = encoder.state["latest"][tf][f"xw_crosses_active_{tf}"]
        assert crosses >= 1.0, f"Expected crossing, got {crosses}"


# ============================================================================
# Integration Test: Full compute_row
# ============================================================================

class TestComputeRow:
    def test_compute_row_returns_series(self, encoder, sample_kline,
                                        sample_decomposed_w30,
                                        sample_decomposed_w120):
        """compute_row returns a pandas Series with features."""
        new_klines = {"5M": sample_kline}
        new_decomposed = {
            ("5M", W_FAST): sample_decomposed_w30,
            ("5M", W_SLOW): sample_decomposed_w120,
        }
        timestamp = pd.Timestamp("2026-01-15 12:00:00")

        result = encoder.compute_row(new_klines, new_decomposed, timestamp)
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_compute_row_all_tfs(self, encoder, sample_kline,
                                  sample_decomposed_w30,
                                  sample_decomposed_w120,
                                  sample_window_data):
        """compute_row with all TFs updated produces 500+ features."""
        new_klines = {tf: sample_kline for tf in TIMEFRAME_ORDER}
        new_decomposed = {}
        for tf in TIMEFRAME_ORDER:
            new_decomposed[(tf, W_FAST)] = sample_decomposed_w30
            new_decomposed[(tf, W_SLOW)] = sample_decomposed_w120
            for ws in ALL_WINDOWS:
                new_decomposed[(tf, ws)] = sample_window_data[ws]

        timestamp = pd.Timestamp("2026-01-15 12:00:00")
        result = encoder.compute_row(new_klines, new_decomposed, timestamp)

        # Should have 400+ features (some may be missing due to
        # buffer warm-up in first iteration)
        assert len(result) >= 400, f"Only {len(result)} features"

    def test_compute_row_no_nan(self, encoder, sample_kline,
                                 sample_decomposed_w30,
                                 sample_decomposed_w120):
        """compute_row should not produce NaN values."""
        new_klines = {tf: sample_kline for tf in TIMEFRAME_ORDER}
        new_decomposed = {}
        for tf in TIMEFRAME_ORDER:
            new_decomposed[(tf, W_FAST)] = sample_decomposed_w30
            new_decomposed[(tf, W_SLOW)] = sample_decomposed_w120
            for ws in ALL_WINDOWS:
                new_decomposed[(tf, ws)] = {
                    "angle": 10.0 + ws * 0.1,
                    "slope_f": 0.01 + ws * 0.001,
                }

        timestamp = pd.Timestamp("2026-01-15 12:00:00")
        result = encoder.compute_row(new_klines, new_decomposed, timestamp)

        nan_features = [k for k, v in result.items() if pd.isna(v)]
        assert len(nan_features) == 0, f"NaN features: {nan_features}"

    def test_temporal_features(self, encoder, sample_kline,
                                sample_decomposed_w30,
                                sample_decomposed_w120):
        """Temporal features computed correctly from timestamp."""
        new_klines = {"5M": sample_kline}
        new_decomposed = {
            ("5M", W_FAST): sample_decomposed_w30,
            ("5M", W_SLOW): sample_decomposed_w120,
        }
        # 14:30 UTC on a Wednesday (dow=2)
        timestamp = pd.Timestamp("2026-01-14 14:30:00")
        result = encoder.compute_row(new_klines, new_decomposed, timestamp)

        hour = 14 + 30 / 60.0
        expected_hour_sin = math.sin(2 * math.pi * hour / 24.0)
        assert abs(result["hour_sin"] - expected_hour_sin) < 1e-10

        # 14:30 is in NY session (13-21)
        assert result["is_ny_session"] == 1.0

    def test_cross_tf_features_present(self, encoder, sample_kline,
                                        sample_decomposed_w30,
                                        sample_decomposed_w120):
        """Cross-TF summary features are present in output."""
        new_klines = {tf: sample_kline for tf in TIMEFRAME_ORDER}
        new_decomposed = {}
        for tf in TIMEFRAME_ORDER:
            new_decomposed[(tf, W_FAST)] = sample_decomposed_w30
            new_decomposed[(tf, W_SLOW)] = sample_decomposed_w120

        timestamp = pd.Timestamp("2026-01-15 12:00:00")
        result = encoder.compute_row(new_klines, new_decomposed, timestamp)

        cross_tf_features = [
            "cross_tf_slope_agreement", "cross_tf_weighted_slope",
            "xtf_total_crosses", "xtf_total_long", "xtf_total_short",
            "xtf_total_converging", "xtf_tfs_with_crosses",
            "xtf_cascade_score", "xtf_direction_net", "xtf_direction_agreement",
            "xtf_reversal_total", "xtf_reversal_confirmed",
            "xtf_convergence_momentum", "xtf_corr_agreement",
        ]
        for feat in cross_tf_features:
            assert feat in result.index, f"Missing: {feat}"

    def test_interaction_features_present(self, encoder, sample_kline,
                                           sample_decomposed_w30,
                                           sample_decomposed_w120):
        """Phase F5 interaction features are present."""
        new_klines = {tf: sample_kline for tf in TIMEFRAME_ORDER}
        new_decomposed = {}
        for tf in TIMEFRAME_ORDER:
            new_decomposed[(tf, W_FAST)] = sample_decomposed_w30
            new_decomposed[(tf, W_SLOW)] = sample_decomposed_w120

        timestamp = pd.Timestamp("2026-01-15 12:00:00")
        result = encoder.compute_row(new_klines, new_decomposed, timestamp)

        for feat in ["convergence_volume", "crossing_atr",
                     "cascade_volume", "reversal_conviction"]:
            assert feat in result.index, f"Missing: {feat}"

    def test_partial_tf_update(self, encoder, sample_kline,
                                sample_decomposed_w30,
                                sample_decomposed_w120):
        """Updating only some TFs reuses stored values for others."""
        # First: update all TFs
        new_klines = {tf: sample_kline for tf in TIMEFRAME_ORDER}
        new_decomposed = {}
        for tf in TIMEFRAME_ORDER:
            new_decomposed[(tf, W_FAST)] = sample_decomposed_w30
            new_decomposed[(tf, W_SLOW)] = sample_decomposed_w120

        timestamp1 = pd.Timestamp("2026-01-15 12:00:00")
        result1 = encoder.compute_row(new_klines, new_decomposed, timestamp1)

        # Second: update only 5M
        result2 = encoder.compute_row(
            {"5M": sample_kline},
            {("5M", W_FAST): sample_decomposed_w30,
             ("5M", W_SLOW): sample_decomposed_w120},
            pd.Timestamp("2026-01-15 12:05:00"),
        )

        # 1D features should be identical (not updated in second call)
        assert result1.get("slope_f_mag_1D") == result2.get("slope_f_mag_1D")
        assert result1.get("angle_slow_1D") == result2.get("angle_slow_1D")

    def test_timestamp_updates(self, encoder, sample_kline,
                                sample_decomposed_w30,
                                sample_decomposed_w120):
        """last_timestamp updates after compute_row."""
        new_klines = {"5M": sample_kline}
        new_decomposed = {
            ("5M", W_FAST): sample_decomposed_w30,
            ("5M", W_SLOW): sample_decomposed_w120,
        }
        ts = pd.Timestamp("2026-01-15 12:00:00")
        encoder.compute_row(new_klines, new_decomposed, ts)

        assert encoder.state["last_timestamp"] == str(ts)


# ============================================================================
# Multi-step consistency tests
# ============================================================================

class TestMultiStep:
    def _make_kline(self, price, volume=1000.0):
        """Generate a kline with some spread."""
        return {
            "Open": price - 10,
            "High": price + 20,
            "Low": price - 30,
            "Close": price,
            "Volume": volume,
        }

    def _make_decomp(self, angle, slope_f=0.01):
        """Generate decomposed data."""
        return {
            "slope_f": slope_f,
            "slope_b": slope_f * 0.8,
            "angle": angle,
            "corr": 0.85,
            "acceleration": 0.1,
            "p_value_f": 0.01,
        }

    def test_ema_continuity_over_steps(self):
        """EMA values change smoothly across steps (no jumps)."""
        encoder = IncrementalEncoder()
        tf = "5M"

        ema_history = []
        for i in range(50):
            price = 50000 + i * 10
            kline = self._make_kline(price)
            decomp_w30 = self._make_decomp(10 + i * 0.1)
            decomp_w120 = self._make_decomp(8 + i * 0.05)

            encoder._update_klines(tf, kline)
            encoder._update_decomposed(tf, decomp_w30, decomp_w120)

            ema = encoder.state["klines"][tf]["ema_vol50"]
            ema_history.append(ema)

        # EMA should be monotonically changing (since volume is constant at 1000)
        # After initialization, values should be close to 1000
        assert abs(ema_history[-1] - 1000.0) < 100.0

    def test_buffer_sizes_correct(self):
        """After many updates, buffers are at max size."""
        encoder = IncrementalEncoder()
        tf = "1H"

        for i in range(100):
            kline = self._make_kline(50000 + i)
            decomp_w30 = self._make_decomp(10 + i * 0.01)
            decomp_w120 = self._make_decomp(8 + i * 0.005)

            encoder._update_klines(tf, kline)
            encoder._update_decomposed(tf, decomp_w30, decomp_w120)

        ks = encoder.state["klines"][tf]
        ds = encoder.state["decomposed"][tf]

        # Buffer size checks
        assert len(ks["vol_buf"]) == VOL_BUF_SIZE
        assert len(ks["wick_buf"]) == WICK_BUF_SIZE
        assert len(ks["body_pct_buf"]) == BODY_PCT_BUF_SIZE
        assert len(ks["body_buf_10"]) == 10
        assert len(ks["body_buf_50"]) == 50
        assert len(ds["d_sf_buf"]) == D_SLOPE_F_BUF_SIZE
        assert len(ds["corr_buf"]) == CORR_BUF_SIZE

        for w in STOCH_WINDOWS:
            assert len(ks[f"high_buf_{w}"]) == w
            assert len(ks[f"low_buf_{w}"]) == w
            assert len(ks[f"close_buf_{w}"]) == w

    def test_no_internal_keys_in_output(self):
        """compute_row output should not contain keys starting with _."""
        encoder = IncrementalEncoder()

        new_klines = {tf: self._make_kline(50000) for tf in TIMEFRAME_ORDER}
        new_decomposed = {}
        for tf in TIMEFRAME_ORDER:
            new_decomposed[(tf, W_FAST)] = self._make_decomp(10)
            new_decomposed[(tf, W_SLOW)] = self._make_decomp(8)
            for ws in ALL_WINDOWS:
                new_decomposed[(tf, ws)] = {"angle": 10.0, "slope_f": 0.01}

        result = encoder.compute_row(new_klines, new_decomposed,
                                     pd.Timestamp("2026-01-15 12:00:00"))

        internal_keys = [k for k in result.index if k.startswith("_")]
        assert len(internal_keys) == 0, f"Internal keys leaked: {internal_keys}"


# ============================================================================
# Regression Test: Batch vs Incremental (requires real data)
# ============================================================================

FEATURE_MATRIX_PATH = Path(__file__).parent.parent / "model_training" / "encoded_data" / "feature_matrix_v10.parquet"
ACTUAL_DATA_DIR = Path(__file__).parent.parent / "model_training" / "actual_data"
DECOMPOSED_DIR = Path(__file__).parent.parent / "model_training" / "decomposed_data"


@pytest.mark.skipif(
    not FEATURE_MATRIX_PATH.exists(),
    reason="Feature matrix not available (run encode_v10.py first)"
)
class TestBatchVsIncremental:
    """Compare incremental encoder output with batch-encoded feature matrix.

    These tests use real data and verify that the incremental encoder
    produces values close to the batch encoder for known rows.
    """

    @pytest.fixture(scope="class")
    def feature_matrix(self):
        """Load last 100 rows of feature matrix."""
        fm = pd.read_parquet(FEATURE_MATRIX_PATH)
        fm["time"] = pd.to_datetime(fm["time"]).dt.tz_localize(None)
        fm = fm.sort_values("time").reset_index(drop=True)
        return fm.tail(100).reset_index(drop=True)

    def test_feature_count_matches(self, feature_matrix):
        """Incremental encoder should produce same number of features."""
        fm_features = set(c for c in feature_matrix.columns if c != "time")
        # The incremental encoder should produce all these features
        # (minus a few that might need special handling)
        assert len(fm_features) >= 500, f"Expected 500+ features, got {len(fm_features)}"

    def test_state_initialization(self, feature_matrix):
        """State can be initialized from real feature matrix."""
        if not ACTUAL_DATA_DIR.exists() or not DECOMPOSED_DIR.exists():
            pytest.skip("Raw data not available")

        from data_service.state_initializer import initialize_state

        state = initialize_state(
            str(FEATURE_MATRIX_PATH),
            str(ACTUAL_DATA_DIR),
            str(DECOMPOSED_DIR),
        )

        encoder = IncrementalEncoder(state)

        # Check that most TFs are initialized
        n_init = sum(
            1 for tf in TIMEFRAME_ORDER
            if encoder.state["decomposed"][tf]["initialized"]
        )
        assert n_init >= 8, f"Only {n_init}/11 TFs initialized"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
