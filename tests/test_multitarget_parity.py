"""Multi-target stacking parity contract test.

This test is the deploy gate for the Contabo multi-target stack. It
verifies that the live ``MultiTargetPredictor`` reconstructs the
258-column meta-feature vector bit-identically to a fresh forward pass
through the shared training-side builder.

Why this matters
----------------
Two code paths write the meta vector:

* **Training**: ``train_multitarget_stacking.py`` ->
  ``build_interaction_features`` (now imported from
  ``core.multitarget_feature_builder``) on a base-prob DataFrame
  computed once across the OOS slice + raw V10 features.
* **Live**: ``MultiTargetPredictor.predict`` pushes per-cycle base probs
  + raw projections into 289-row ring buffers, then ``_build_live_meta_row``
  reconstructs DataFrames from those buffers and feeds them to the SAME
  ``build_stacking_meta_features``.

If the ring-buffer round-trip drops/reorders/corrupts a column, the
live stacking head sees out-of-distribution input and produces silently
wrong predictions. This test asserts the round-trip is lossless to
1e-6 absolute tolerance.

The test is skipped (xfail-strict=False) if the multi-target artifacts
or the feature_matrix_v10.parquet bootstrap source are not present on
the test machine -- the goal is a deploy gate, not a CI hard fail on a
fresh checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
MULTITARGET_ROOT = REPO_ROOT / "model_training" / "results_v10" / "multitarget"
FEATURE_MATRIX_PATH = REPO_ROOT / "model_training" / "encoded_data" / "feature_matrix_v10.parquet"
TOP_RAW_UNION_PATH = MULTITARGET_ROOT / "stacking" / "top_raw_features_union.json"

ATOL = 1e-6
RING = 289  # 288 history + 1 current


def _have_artifacts() -> bool:
    if not FEATURE_MATRIX_PATH.exists():
        return False
    if not TOP_RAW_UNION_PATH.exists():
        return False
    if not (MULTITARGET_ROOT / "base_models").exists():
        return False
    if not (MULTITARGET_ROOT / "stacking").exists():
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _have_artifacts(),
    reason="multi-target artifacts or feature_matrix_v10.parquet not present",
)


@pytest.fixture(scope="module")
def v10_slice() -> pd.DataFrame:
    """Load the first ``RING`` rows of the V10 feature matrix."""
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df = df.sort_values("time").head(RING).reset_index(drop=True)
    assert len(df) == RING, f"need {RING} rows, got {len(df)}"
    return df


@pytest.fixture(scope="module")
def v10_feature_names(v10_slice) -> list[str]:
    drop = {"time", "label", "Open", "High", "Low", "Close", "Volume",
            "Open Time", "Close Time"}
    return [c for c in v10_slice.columns if c not in drop]


@pytest.fixture(scope="module")
def base_models(v10_feature_names):
    """Load all 24 base CatBoost models keyed by (target, seed)."""
    from model_training.multitarget_config import TARGET_NAMES, SEEDS
    out = {}
    for t in TARGET_NAMES:
        for s in SEEDS:
            p = MULTITARGET_ROOT / "base_models" / f"base_model_{t}_s{s}.cbm"
            assert p.exists(), f"missing {p}"
            m = CatBoostClassifier()
            m.load_model(str(p))
            out[(t, s)] = m
    return out


def _run_base_independent(base_models, feature_arr, v10_feature_names):
    """Run base models on each row independently and average across seeds."""
    from model_training.multitarget_config import TARGET_NAMES, SEEDS
    n_rows = feature_arr.shape[0]
    out = {t: np.zeros((n_rows, 3), dtype=np.float64) for t in TARGET_NAMES}
    for i in range(n_rows):
        x = feature_arr[i].reshape(1, -1)
        for t in TARGET_NAMES:
            stacked = np.zeros(3, dtype=np.float64)
            for s in SEEDS:
                stacked += base_models[(t, s)].predict_proba(x)[0]
            out[t][i] = stacked / len(SEEDS)
    return out


def test_meta_vector_parity(v10_slice, v10_feature_names, base_models):
    """Live ring-buffer round-trip == direct forward pass to 1e-6."""
    from core.multitarget_feature_builder import (
        EXPECTED_META_FEATURE_COUNT,
        build_stacking_meta_features,
        load_top_raw_features_union,
    )
    from core.multitarget_predictor import MultiTargetPredictor
    from model_training.multitarget_config import TARGET_NAMES

    # ----- Path A: live predictor full flow with cold-start backfill -----
    predictor = MultiTargetPredictor(
        models_root=MULTITARGET_ROOT,
        top_raw_features_path=TOP_RAW_UNION_PATH,
        v10_feature_names=v10_feature_names,
        state_path=None,
        backfill_features_path=FEATURE_MATRIX_PATH,
    )
    # The cold-start backfill loads the LAST 289 rows of the parquet,
    # not the first 289. To make Path A and Path B comparable we replay
    # the same slice via the public predict() interface so both paths
    # see the same data.
    predictor._timestamps.clear()
    for t in predictor.targets:
        predictor._base_buffers[t].clear()
    predictor._raw_buffer.clear()

    feature_arr = v10_slice[v10_feature_names].to_numpy(dtype=np.float64)
    last_pred = None
    for i in range(RING):
        row_series = pd.Series(feature_arr[i], index=v10_feature_names)
        last_pred = predictor.predict(row_series, v10_slice["time"].iloc[i])

    # Predictor warms during the first RING-1 cycles; the final call returns
    # the live stacking outputs but we want the meta vector itself.
    live_meta_vec = predictor._build_live_meta_row()

    # ----- Path B: independent forward pass -----
    top_raw = load_top_raw_features_union(TOP_RAW_UNION_PATH)
    base_arrays = _run_base_independent(base_models, feature_arr, v10_feature_names)

    times = pd.to_datetime(v10_slice["time"].astype(str)).values
    base_data = {"time": times}
    for t in TARGET_NAMES:
        tl = t.lower()
        base_data[f"{tl}_prob_nt"] = base_arrays[t][:, 0]
        base_data[f"{tl}_prob_long"] = base_arrays[t][:, 1]
        base_data[f"{tl}_prob_short"] = base_arrays[t][:, 2]
    base_df = pd.DataFrame(base_data)

    raw_df = v10_slice[["time"] + list(top_raw)].copy()

    meta_df = build_stacking_meta_features(base_df, raw_df, list(TARGET_NAMES), top_raw)
    feature_cols = [c for c in meta_df.columns if c != "time"]
    direct_meta_vec = meta_df[feature_cols].iloc[-1].to_numpy(dtype=np.float64)

    # ----- Assertions -----
    assert live_meta_vec.shape == direct_meta_vec.shape, (
        f"shape drift: live={live_meta_vec.shape} direct={direct_meta_vec.shape}"
    )
    assert live_meta_vec.shape[0] == EXPECTED_META_FEATURE_COUNT, (
        f"meta vec is {live_meta_vec.shape[0]} cols, expected {EXPECTED_META_FEATURE_COUNT}"
    )

    diff = np.abs(live_meta_vec - direct_meta_vec)
    max_idx = int(np.argmax(diff))
    max_diff = float(diff[max_idx])
    assert max_diff <= ATOL, (
        f"meta vector parity FAILED: max abs diff = {max_diff:.3e} "
        f"at column index {max_idx} (>{ATOL})"
    )

    assert last_pred is not None
    assert not last_pred.warming, (
        f"final predict() should be ready after {RING} rows, got warming"
    )
