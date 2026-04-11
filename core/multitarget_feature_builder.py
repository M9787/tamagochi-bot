"""Shared meta-feature builder for the multi-target stacking pipeline.

Both training (`model_training/train_multitarget_stacking.py`) and live
inference (`core/multitarget_predictor.py`) must produce bit-identical
258-column meta-feature rows from 24 base-model probability outputs +
top-50-union raw V10 features. This module is the single source of truth.

Two public entry points:

* ``build_interaction_features(meta_df, target_names)`` -- verbatim
  extraction of the legacy training builder. Same signature as the old
  in-file copy so the training script can swap in via ``import ... as``.
  Operates only on the 24 base-prob columns; raw-feature merging is left
  to the caller (matches the historical training flow).

* ``build_stacking_meta_features(base_probs_df, raw_features_df,
  target_names, top_raw_features)`` -- richer wrapper for live use.
  Calls ``build_interaction_features`` then merges the top-50 raw V10
  feature union, returning a single combined DataFrame in the same column
  order training produced after its own merge step.

* ``load_top_raw_features_union(path)`` -- reads the persisted top-50
  union list dumped from the training script (Phase A.4).

Module constants:

* ``EXPECTED_META_FEATURE_COUNT = 258`` -- asserted at predict time as a
  fail-fast parity check.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from model_training.multitarget_config import SCALE_GROUPS

logger = logging.getLogger(__name__)

EXPECTED_META_FEATURE_COUNT = 258


def build_interaction_features(meta_df: pd.DataFrame,
                               target_names: list) -> pd.DataFrame:
    """Build interaction features from 8 base model probabilities.

    Verbatim extraction from train_multitarget_stacking.py (legacy
    location: lines 113-264). Mutates ``meta_df`` in place AND returns it
    so existing call sites continue to work.

    Inputs
    ------
    meta_df : DataFrame with columns ``time`` and, for each target T in
        ``target_names``, the three columns ``{t.lower()}_prob_nt``,
        ``{t.lower()}_prob_long``, ``{t.lower()}_prob_short``.
    target_names : list of target identifiers, e.g. ``["T1",...,"T8"]``.

    Output
    ------
    Same DataFrame with ~150 additional interaction columns appended.
    """
    logger.info("  Engineering interaction features...")
    t0 = time.time()

    prefixes = [t.lower() for t in target_names]

    long_cols = [f'{p}_prob_long' for p in prefixes]
    short_cols = [f'{p}_prob_short' for p in prefixes]
    nt_cols = [f'{p}_prob_nt' for p in prefixes]

    # --- Cross-model averages ---
    meta_df['avg_prob_nt'] = meta_df[nt_cols].mean(axis=1)
    meta_df['avg_prob_long'] = meta_df[long_cols].mean(axis=1)
    meta_df['avg_prob_short'] = meta_df[short_cols].mean(axis=1)

    # --- Spread (disagreement) ---
    meta_df['spread_long'] = meta_df[long_cols].max(axis=1) - meta_df[long_cols].min(axis=1)
    meta_df['spread_short'] = meta_df[short_cols].max(axis=1) - meta_df[short_cols].min(axis=1)
    meta_df['spread_nt'] = meta_df[nt_cols].max(axis=1) - meta_df[nt_cols].min(axis=1)

    # --- Std (dispersion) ---
    meta_df['std_long'] = meta_df[long_cols].std(axis=1)
    meta_df['std_short'] = meta_df[short_cols].std(axis=1)

    # --- Entropy per model ---
    for p in prefixes:
        probs = meta_df[[f'{p}_prob_nt', f'{p}_prob_long', f'{p}_prob_short']].values
        probs = np.clip(probs, 1e-10, 1.0)
        meta_df[f'{p}_entropy'] = -np.sum(probs * np.log(probs), axis=1)

    # --- Consensus strength ---
    all_trade_cols = long_cols + short_cols
    meta_df['max_trade_prob'] = meta_df[all_trade_cols].max(axis=1)
    meta_df['max_nt_prob'] = meta_df[nt_cols].max(axis=1)
    meta_df['trade_vs_nt'] = meta_df['max_trade_prob'] - meta_df['max_nt_prob']

    # --- Direction agreement ---
    dir_cols = []
    for p in prefixes:
        col = f'{p}_direction'
        meta_df[col] = meta_df[f'{p}_prob_long'] - meta_df[f'{p}_prob_short']
        dir_cols.append(col)

    meta_df['direction_agreement'] = meta_df[dir_cols].apply(
        lambda row: np.sum(np.sign(row.values) == np.sign(row.values[0])) / len(row),
        axis=1,
    )
    meta_df['avg_direction'] = meta_df[dir_cols].mean(axis=1)

    # --- Confidence ratio per model ---
    for p in prefixes:
        max_trade = np.maximum(meta_df[f'{p}_prob_long'], meta_df[f'{p}_prob_short'])
        meta_df[f'{p}_conf_ratio'] = max_trade / (meta_df[f'{p}_prob_nt'] + 1e-10)

    # --- Scale-group features ---
    for group_name, group_targets in SCALE_GROUPS.items():
        group_prefixes = [t.lower() for t in group_targets]
        for direction in ['long', 'short']:
            cols = [f'{p}_prob_{direction}' for p in group_prefixes]
            meta_df[f'{group_name}_avg_{direction}'] = meta_df[cols].mean(axis=1)
            meta_df[f'{group_name}_std_{direction}'] = meta_df[cols].std(axis=1)

    # Cross-scale divergence
    for direction in ['long', 'short']:
        meta_df[f'scalp_vs_position_{direction}'] = (
            meta_df[f'scalp_avg_{direction}'] - meta_df[f'position_avg_{direction}']
        )
        meta_df[f'intra_vs_position_{direction}'] = (
            meta_df[f'intra_avg_{direction}'] - meta_df[f'position_avg_{direction}']
        )
        meta_df[f'scalp_vs_intra_{direction}'] = (
            meta_df[f'scalp_avg_{direction}'] - meta_df[f'intra_avg_{direction}']
        )

    # --- Lagged probabilities (6h, 24h) ---
    lag_windows = {'lag72': 72, 'lag288': 288}
    for p in prefixes:
        for lag_name, lag_val in lag_windows.items():
            meta_df[f'{p}_long_{lag_name}'] = meta_df[f'{p}_prob_long'].shift(lag_val).fillna(0)
            meta_df[f'{p}_short_{lag_name}'] = meta_df[f'{p}_prob_short'].shift(lag_val).fillna(0)

    # --- Temporal momentum ---
    for p in prefixes:
        p_long = meta_df[f'{p}_prob_long']
        p_short = meta_df[f'{p}_prob_short']
        meta_df[f'{p}_long_mom_72'] = (p_long - p_long.shift(72)).fillna(0)
        meta_df[f'{p}_long_mom_288'] = (p_long - p_long.shift(288)).fillna(0)
        meta_df[f'{p}_short_mom_72'] = (p_short - p_short.shift(72)).fillna(0)
        meta_df[f'{p}_short_mom_288'] = (p_short - p_short.shift(288)).fillna(0)

    # --- Rolling stability ---
    for p in prefixes:
        meta_df[f'{p}_long_std_72'] = meta_df[f'{p}_prob_long'].rolling(72, min_periods=1).std().fillna(0)
        meta_df[f'{p}_long_std_288'] = meta_df[f'{p}_prob_long'].rolling(288, min_periods=1).std().fillna(0)

    # --- Voting features ---
    vote_long = sum(
        (meta_df[f'{p}_prob_long'] > meta_df[f'{p}_prob_nt']).astype(int)
        for p in prefixes
    )
    vote_short = sum(
        (meta_df[f'{p}_prob_short'] > meta_df[f'{p}_prob_nt']).astype(int)
        for p in prefixes
    )
    meta_df['vote_long'] = vote_long
    meta_df['vote_short'] = vote_short
    meta_df['vote_strength'] = (vote_long - vote_short).astype(float)

    # --- KL divergence for uncorrelated pairs (T1-T7, T1-T8, T3-T8) ---
    kl_pairs = [('t1_t7', 't1', 't7'), ('t1_t8', 't1', 't8'), ('t3_t8', 't3', 't8')]
    for pair_name, m1, m2 in kl_pairs:
        p = meta_df[[f'{m1}_prob_nt', f'{m1}_prob_long', f'{m1}_prob_short']].values + 1e-8
        q = meta_df[[f'{m2}_prob_nt', f'{m2}_prob_long', f'{m2}_prob_short']].values + 1e-8
        kl = (p * np.log(p / q)).sum(axis=1)
        meta_df[f'kl_div_{pair_name}'] = kl

    # --- Max disagreement ---
    meta_df['max_disagreement'] = pd.concat([
        meta_df[long_cols].std(axis=1),
        meta_df[short_cols].std(axis=1),
        meta_df[nt_cols].std(axis=1),
    ], axis=1).max(axis=1)

    # --- Conviction vs consensus ---
    meta_df['conviction_vs_consensus'] = (
        meta_df['max_trade_prob'] - meta_df[['avg_prob_long', 'avg_prob_short']].max(axis=1)
    )

    # --- Regime proxy ---
    meta_df['regime_proxy'] = meta_df['avg_prob_nt'].rolling(288, min_periods=1).std()

    # --- Percentile ranks ---
    for direction in ['long', 'short']:
        col = f'avg_prob_{direction}'
        meta_df[f'prank_{direction}'] = meta_df[col].rolling(288, min_periods=1).rank(pct=True)

    # Drop helper direction columns
    for p in prefixes:
        col = f'{p}_direction'
        if col in meta_df.columns:
            meta_df.drop(columns=[col], inplace=True)

    elapsed = time.time() - t0
    n_new = len([c for c in meta_df.columns if c != 'time' and not c.endswith(('_prob_nt', '_prob_long', '_prob_short'))])
    logger.info(f"  Interaction features: {n_new} new columns ({elapsed:.1f}s)")

    return meta_df


def build_stacking_meta_features(base_probs_df: pd.DataFrame,
                                 raw_features_df: pd.DataFrame,
                                 target_names: list,
                                 top_raw_features: Iterable[str]) -> pd.DataFrame:
    """Build the full meta-feature matrix used by stacking models.

    Wraps ``build_interaction_features`` and merges the top-50-union raw
    V10 features by ``time`` -- producing the same combined DataFrame the
    training script builds in its [2/5] + [3/5] steps.

    Inputs
    ------
    base_probs_df : 24 base-prob columns + ``time``. For live use this is
        the 289-row ring buffer (288 history + current). For training
        parity tests it can be any historical slice.
    raw_features_df : raw V10 features keyed by ``time``. Must contain at
        least every column in ``top_raw_features``. Extra columns are
        ignored after the projection.
    target_names : list of target identifiers, e.g. ``["T1",...,"T8"]``.
    top_raw_features : iterable of raw V10 column names to include
        (sorted union across all base-model importances).

    Output
    ------
    DataFrame with ``time`` + interaction columns + raw feature columns
    in the same order as the training pipeline produces. Caller takes
    the last row for live inference.
    """
    meta_df = build_interaction_features(base_probs_df.copy(), target_names)
    raw_cols_sorted = sorted(top_raw_features)
    cols_to_keep = ['time'] + raw_cols_sorted
    raw_proj = raw_features_df[cols_to_keep]
    meta_df = meta_df.merge(raw_proj, on='time', how='left')
    return meta_df


def load_top_raw_features_union(path: Path | str) -> list[str]:
    """Read the persisted top-50 raw features union (Phase A.4 dump).

    The training script computes this union once across all 8 base
    models' feature-importance CSVs and dumps it next to the stacking
    artifacts. The live predictor needs the same list at __init__ to
    project raw V10 rows down to the stacking input columns.
    """
    path = Path(path)
    with path.open('r') as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"top_raw_features_union must be a JSON list, got {type(data).__name__}")
    return sorted(data)
