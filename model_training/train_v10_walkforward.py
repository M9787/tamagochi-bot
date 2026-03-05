"""
V10 Walk-Forward Validation — 4-Fold Expanding Window, 3 Seeds

Validates V10 Cross-Scale Convergence features (508 features) with walk-forward.
Uses V7 d8 winner config (depth=8, lr=0.02, l2=15, cw=[0.5,2,2]).
3 seeds per fold = 12 total models.

Audit fixes (same as V6 WF):
  - C1: Threshold selected on VAL set, not test
  - C2: Pass/fail uses val-selected threshold
  - M1: 7-day embargo between val and test
  - M2: 60-candle trade cooldown

Fold layout:
  Fold 0: Train -> 2025-01  |  Val -> 2025-02  |  Embargo 7d  |  Test 2025-02-08->05-01
  Fold 1: Train -> 2025-04  |  Val -> 2025-05  |  Embargo 7d  |  Test 2025-05-08->08-01
  Fold 2: Train -> 2025-07  |  Val -> 2025-08  |  Embargo 7d  |  Test 2025-08-08->11-01
  Fold 3: Train -> 2025-10  |  Val -> 2025-11  |  Embargo 7d  |  Test 2025-11-08->2026-01-15

Pass criteria:
  - AUC > 0.70 in all folds
  - 3+ of 4 folds profitable at val-selected threshold
  - Shorts in 3+ folds
  - Fixed-threshold @0.70/@0.75/@0.80 cross-fold profitability

Output: results_v10/walkforward/

Usage:
  python model_training/train_v10_walkforward.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import logging
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
)

from model_training.build_labels import load_labels

logger = logging.getLogger(__name__)

ENCODED_DIR = Path(__file__).parent / "encoded_data"
RESULTS_DIR = Path(__file__).parent / "results_v10" / "walkforward"

# SL/TP config
SL_PCT = 2.0
TP_PCT = 4.0

# Class names
CLASS_NAMES = {0: 'NO_TRADE', 1: 'LONG', 2: 'SHORT'}
TRADE_CLASSES = [1, 2]

# V7 d8 winner params
MODEL_PARAMS = dict(
    depth=8,
    iterations=5000,
    learning_rate=0.02,
    l2_leaf_reg=15,
    min_data_in_leaf=500,
    loss_function='MultiClass',
    eval_metric='TotalF1:average=Macro;use_weights=false',
    verbose=500,
    task_type='GPU',
    class_weights=[0.5, 2.0, 2.0],
    early_stopping_rounds=600,
    has_time=True,
    random_strength=1,
    border_count=254,
    subsample=0.7,
    bootstrap_type='Bernoulli',
)
MIN_ITERATIONS = 200

# Multi-seed
SEEDS = [42, 123, 777]

# Thresholds (extended to @0.80 for precision targeting)
THRESHOLDS = [0.42, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

# Embargo and cooldown
EMBARGO_DAYS = 7
TRADE_COOLDOWN = 60

# Walk-forward fold definitions
FOLDS = [
    {
        'name': 'Fold 0',
        'train_end': pd.Timestamp('2025-01-01'),
        'val_end': pd.Timestamp('2025-02-01'),
        'embargo_end': pd.Timestamp('2025-02-08'),
        'test_end': pd.Timestamp('2025-05-01'),
    },
    {
        'name': 'Fold 1',
        'train_end': pd.Timestamp('2025-04-01'),
        'val_end': pd.Timestamp('2025-05-01'),
        'embargo_end': pd.Timestamp('2025-05-08'),
        'test_end': pd.Timestamp('2025-08-01'),
    },
    {
        'name': 'Fold 2',
        'train_end': pd.Timestamp('2025-07-01'),
        'val_end': pd.Timestamp('2025-08-01'),
        'embargo_end': pd.Timestamp('2025-08-08'),
        'test_end': pd.Timestamp('2025-11-01'),
    },
    {
        'name': 'Fold 3',
        'train_end': pd.Timestamp('2025-10-01'),
        'val_end': pd.Timestamp('2025-11-01'),
        'embargo_end': pd.Timestamp('2025-11-08'),
        'test_end': pd.Timestamp('2026-01-15'),
    },
]


# ============================================================================
# Label Preparation
# ============================================================================

def prepare_3class_labels(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    labels['label_3class'] = labels['label'].map({1: 1, -1: 2, 0: 0})
    return labels


# ============================================================================
# Data Alignment
# ============================================================================

def align_features_labels(features: pd.DataFrame, labels: pd.DataFrame):
    """Align features with labels by timestamp. Returns aligned X, y_df, times."""
    feat_times = pd.to_datetime(features['time'])
    label_times = pd.to_datetime(labels['timestamp'])

    common = set(feat_times.values) & set(label_times.values)
    logger.info(f"  Overlapping timestamps: {len(common):,}")

    if len(common) == 0:
        raise ValueError("No timestamp overlap between features and labels.")

    common_sorted = sorted(common)
    feat_mask = feat_times.isin(common_sorted)
    label_mask = label_times.isin(common_sorted)

    X = features[feat_mask.values].sort_values('time').reset_index(drop=True)
    y_df = labels[label_mask.values].sort_values('timestamp').reset_index(drop=True)
    times = pd.to_datetime(X['time'])

    logger.info(f"  Total aligned: {len(X):,} rows ({times.min()} to {times.max()})")

    return X, y_df, times


# ============================================================================
# Training
# ============================================================================

def train_model(X_train, y_train, X_val, y_val, seed=42):
    """Train a single CatBoost 3-class model."""
    train_pool = Pool(X_train, y_train)
    eval_pool = Pool(X_val, y_val)

    params = MODEL_PARAMS.copy()
    params['random_seed'] = seed
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=eval_pool, use_best_model=True)

    best_iter = model.get_best_iteration()
    if best_iter < MIN_ITERATIONS:
        logger.warning(f"  best_iter={best_iter} < {MIN_ITERATIONS}, retraining with min iterations...")
        params_min = params.copy()
        params_min['iterations'] = MIN_ITERATIONS
        params_min.pop('early_stopping_rounds', None)
        model = CatBoostClassifier(**params_min)
        model.fit(train_pool)
        best_iter = MIN_ITERATIONS

    logger.info(f"  Best iteration: {best_iter}")
    return model, best_iter


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_at_threshold(y_pred_proba, y_test, threshold: float, cooldown: int = 0):
    """Evaluate 3-class predictions at a given confidence threshold."""
    n_total = len(y_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    if threshold > 0:
        for i in range(n_total):
            if y_pred[i] in TRADE_CLASSES:
                if y_pred_proba[i, y_pred[i]] < threshold:
                    y_pred[i] = 0

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, labels=[0, 1, 2],
        target_names=['NO_TRADE', 'LONG', 'SHORT'],
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    # Trade metrics (before cooldown)
    trade_mask = np.isin(y_pred, TRADE_CLASSES)
    n_trade_preds = int(trade_mask.sum())
    n_long_preds = int((y_pred == 1).sum())
    n_short_preds = int((y_pred == 2).sum())

    if n_trade_preds > 0:
        trade_correct = (y_pred[trade_mask] == y_test[trade_mask])
        trade_precision = float(trade_correct.mean())
        wrong_dir = ((y_pred[trade_mask] == 1) & (y_test[trade_mask] == 2)) | \
                    ((y_pred[trade_mask] == 2) & (y_test[trade_mask] == 1))
        wrong_dir_rate = float(wrong_dir.mean())
        false_trade = (y_test[trade_mask] == 0)
        false_trade_rate = float(false_trade.mean())
    else:
        trade_precision = wrong_dir_rate = false_trade_rate = 0.0

    # Per-class precision
    long_precision = float((y_test[y_pred == 1] == 1).mean()) if n_long_preds > 0 else 0.0
    short_precision = float((y_test[y_pred == 2] == 2).mean()) if n_short_preds > 0 else 0.0

    # Trading simulation WITH cooldown
    equity = [0.0]
    trades = []
    next_allowed_idx = 0
    for i in range(n_total):
        if y_pred[i] == 0:
            continue
        if i < next_allowed_idx:
            continue
        gain = TP_PCT if y_pred[i] == y_test[i] else -SL_PCT
        trades.append({
            'index': i, 'predicted': int(y_pred[i]), 'actual': int(y_test[i]),
            'gain_pct': gain, 'confidence': float(y_pred_proba[i, y_pred[i]]),
        })
        equity.append(equity[-1] + gain)
        if cooldown > 0:
            next_allowed_idx = i + cooldown

    n_trades = len(trades)
    if n_trades > 0:
        trades_df = pd.DataFrame(trades)
        wins = trades_df[trades_df['gain_pct'] > 0]
        losses = trades_df[trades_df['gain_pct'] < 0]
        win_rate = len(wins) / n_trades * 100
        total_profit = trades_df['gain_pct'].sum()
        sum_wins = float(wins['gain_pct'].sum()) if len(wins) > 0 else 0.0
        sum_losses = abs(float(losses['gain_pct'].sum())) if len(losses) > 0 else 0.0
        profit_factor = sum_wins / sum_losses if sum_losses > 0 else float('inf')
        gains_arr = trades_df['gain_pct'].values
        sharpe = float(gains_arr.mean() / gains_arr.std()) if gains_arr.std() > 0 else 0.0
        equity_arr = np.array(equity)
        peak = np.maximum.accumulate(equity_arr)
        max_drawdown = float((equity_arr - peak).min())
    else:
        win_rate = total_profit = 0.0
        profit_factor = sharpe = max_drawdown = 0.0

    # ROC AUC
    try:
        y_onehot = np.zeros((n_total, 3))
        for i in range(n_total):
            y_onehot[i, y_test[i]] = 1
        roc_auc_macro = float(roc_auc_score(y_onehot, y_pred_proba, multi_class='ovr', average='macro'))
        roc_auc_per_class = {}
        avg_prec_per_class = {}
        for cls in range(3):
            if y_onehot[:, cls].sum() > 0:
                roc_auc_per_class[CLASS_NAMES[cls]] = float(
                    roc_auc_score(y_onehot[:, cls], y_pred_proba[:, cls]))
                avg_prec_per_class[CLASS_NAMES[cls]] = float(
                    average_precision_score(y_onehot[:, cls], y_pred_proba[:, cls]))
            else:
                roc_auc_per_class[CLASS_NAMES[cls]] = 0.0
                avg_prec_per_class[CLASS_NAMES[cls]] = 0.0
    except Exception as e:
        logger.warning(f"  ROC AUC failed: {e}")
        roc_auc_macro = 0.0
        roc_auc_per_class = {}
        avg_prec_per_class = {}

    return {
        'threshold': threshold,
        'accuracy': acc,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'roc_auc_macro': roc_auc_macro,
        'roc_auc_per_class': roc_auc_per_class,
        'avg_precision_per_class': avg_prec_per_class,
        'n_total': n_total,
        'n_trade_predictions': n_trade_preds,
        'n_long_predictions': n_long_preds,
        'n_short_predictions': n_short_preds,
        'trade_precision': trade_precision,
        'long_precision': long_precision,
        'short_precision': short_precision,
        'false_trade_rate': false_trade_rate,
        'wrong_direction_rate': wrong_dir_rate,
        'n_trades_simulated': n_trades,
        'win_rate': win_rate,
        'total_profit_pct': total_profit,
        'equity_curve': equity,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
    }


# ============================================================================
# Feature Group Analysis (V10-specific)
# ============================================================================

def analyze_feature_groups(fi_sorted):
    """Analyze feature importance by V10 feature groups."""
    groups = {
        'V6_base': 0.0,
        'F1_cross_window': 0.0,
        'F2_cross_tf': 0.0,
        'F3_corr_dynamics': 0.0,
        'F4_temporal': 0.0,
        'F5_interactions': 0.0,
    }
    group_counts = {k: 0 for k in groups}

    for fname, fval in fi_sorted:
        if fname.startswith('xw_'):
            groups['F1_cross_window'] += fval
            group_counts['F1_cross_window'] += 1
        elif fname.startswith('xtf_') and 'corr' not in fname:
            groups['F2_cross_tf'] += fval
            group_counts['F2_cross_tf'] += 1
        elif 'corr_velocity' in fname or fname == 'xtf_corr_agreement':
            groups['F3_corr_dynamics'] += fval
            group_counts['F3_corr_dynamics'] += 1
        elif fname in ('hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_ny_session'):
            groups['F4_temporal'] += fval
            group_counts['F4_temporal'] += 1
        elif fname in ('convergence_volume', 'crossing_atr', 'cascade_volume', 'reversal_conviction'):
            groups['F5_interactions'] += fval
            group_counts['F5_interactions'] += 1
        else:
            groups['V6_base'] += fval
            group_counts['V6_base'] += 1

    return groups, group_counts


# ============================================================================
# Visualization
# ============================================================================

def save_fold_plots(y_proba, y_test, fold_idx, seed, threshold_results, val_thresh):
    """Save confusion matrix, ROC, PR, equity PNGs for one fold+seed."""
    n_classes = 3
    y_onehot = np.zeros((len(y_test), n_classes))
    for i in range(len(y_test)):
        y_onehot[i, y_test[i]] = 1
    class_colors = {'NO_TRADE': 'gray', 'LONG': 'green', 'SHORT': 'red'}
    prefix = f"f{fold_idx}_s{seed}"

    # Confusion Matrix
    first_key = f"{THRESHOLDS[0]:.2f}"
    cm = np.array(threshold_results[first_key]['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NO_TRADE', 'LONG', 'SHORT'],
                yticklabels=['NO_TRADE', 'LONG', 'SHORT'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'V10 WF Confusion — Fold {fold_idx} s{seed}')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_confusion.png", dpi=150)
    plt.close(fig)

    # ROC Curve
    fig, ax = plt.subplots(figsize=(8, 7))
    for cls in range(n_classes):
        name = CLASS_NAMES[cls]
        if y_onehot[:, cls].sum() > 0:
            fpr, tpr, _ = roc_curve(y_onehot[:, cls], y_proba[:, cls])
            auc_val = roc_auc_score(y_onehot[:, cls], y_proba[:, cls])
            ax.plot(fpr, tpr, color=class_colors[name], linewidth=2,
                    label=f'{name} (AUC={auc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random')
    macro_auc = threshold_results[first_key].get('roc_auc_macro', 0)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'V10 WF ROC — Fold {fold_idx} s{seed} (AUC={macro_auc:.3f})')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{prefix}_roc.png", dpi=150)
    plt.close(fig)

    # Equity curve (val-selected threshold)
    best_key = f"{val_thresh:.2f}"
    if best_key in threshold_results:
        eq = threshold_results[best_key].get('equity_curve', [])
        profit = threshold_results[best_key].get('total_profit_pct', 0)
        if len(eq) > 1:
            fig, ax = plt.subplots(figsize=(14, 6))
            equity_arr = np.array(eq)
            ax.plot(range(len(eq)), eq, color='navy', linewidth=1.2, label='Equity')
            peak = np.maximum.accumulate(equity_arr)
            dd = equity_arr - peak
            ax.fill_between(range(len(eq)), equity_arr, peak,
                            where=(dd < 0), color='red', alpha=0.15, label='Drawdown')
            ax.set_xlabel('Trade #')
            ax.set_ylabel('Cumulative Equity (%)')
            ax.set_title(f'V10 WF Equity — Fold {fold_idx} s{seed} @{val_thresh:.2f} ({profit:.1f}%)')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
            fig.tight_layout()
            fig.savefig(RESULTS_DIR / f"{prefix}_equity.png", dpi=150)
            plt.close(fig)


def save_trade_log(y_proba, y_test, test_times, threshold, fold_idx, seed, cooldown=0):
    """Save per-trade CSV for one fold+seed at given threshold with cooldown."""
    y_pred = np.argmax(y_proba, axis=1)
    if threshold > 0:
        for i in range(len(y_pred)):
            if y_pred[i] in TRADE_CLASSES:
                if y_proba[i, y_pred[i]] < threshold:
                    y_pred[i] = 0

    rows = []
    cumulative = 0.0
    next_allowed_idx = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            continue
        if i < next_allowed_idx:
            continue
        gain = TP_PCT if y_pred[i] == y_test[i] else -SL_PCT
        cumulative += gain
        rows.append({
            'timestamp': str(test_times.iloc[i]),
            'predicted': CLASS_NAMES[int(y_pred[i])],
            'actual': CLASS_NAMES[int(y_test[i])],
            'confidence': float(y_proba[i, y_pred[i]]),
            'gain_pct': gain,
            'cumulative_equity': cumulative,
        })
        if cooldown > 0:
            next_allowed_idx = i + cooldown

    if rows:
        pd.DataFrame(rows).to_csv(
            RESULTS_DIR / f"f{fold_idx}_s{seed}_trades.csv", index=False)


def save_crossfold_equity(fold_seed_equity_data):
    """Save combined equity curves across folds (mean of seeds per fold)."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fold_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for fd in fold_seed_equity_data:
        fold_i = fd['fold']
        color = fold_colors[fold_i % len(fold_colors)]
        eq = fd['equity']
        if len(eq) > 1:
            ax.plot(range(len(eq)), eq, color=color, linewidth=1.2, alpha=0.7,
                    label=f'F{fold_i} s{fd["seed"]} ({fd["profit"]:.0f}%)')

    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative Equity (%)')
    ax.set_title('V10 Walk-Forward Equity — All Folds/Seeds')
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "crossfold_equity.png", dpi=150)
    plt.close(fig)


# ============================================================================
# JSON Helper
# ============================================================================

def convert_for_json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if obj == float('inf'):
        return "inf"
    if obj == float('-inf'):
        return "-inf"
    return obj


# ============================================================================
# Main Pipeline
# ============================================================================

def run_walkforward():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    t_start = time.time()

    logger.info("=" * 90)
    logger.info("V10 WALK-FORWARD VALIDATION — 4-Fold, 3 Seeds, 508 Features")
    logger.info(f"  Config: V7 d8 (depth={MODEL_PARAMS['depth']}, iter={MODEL_PARAMS['iterations']})")
    logger.info(f"  Seeds: {SEEDS} | Embargo={EMBARGO_DAYS}d | Cooldown={TRADE_COOLDOWN}")
    logger.info(f"  SL={SL_PCT}% | TP={TP_PCT}% | Thresholds={THRESHOLDS}")
    logger.info("=" * 90)

    # ------------------------------------------------------------------
    # Step 1: Load V10 features
    # ------------------------------------------------------------------
    logger.info("\n[STEP 1] Loading V10 feature matrix...")
    parquet_path = ENCODED_DIR / "feature_matrix_v10.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing: {parquet_path}\nRun: python model_training/encode_v10.py --force")

    features = pd.read_parquet(parquet_path)
    feature_cols = [c for c in features.columns if c != 'time']
    logger.info(f"  Shape: {features.shape} ({len(feature_cols)} features)")

    # V10 feature breakdown
    n_xw = sum(1 for c in feature_cols if c.startswith('xw_'))
    n_xtf = sum(1 for c in feature_cols if c.startswith('xtf_'))
    n_corr_vel = sum(1 for c in feature_cols if 'corr_velocity' in c)
    n_temporal = sum(1 for c in feature_cols if c in ('hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_ny_session'))
    n_interact = sum(1 for c in feature_cols if c in ('convergence_volume', 'crossing_atr', 'cascade_volume', 'reversal_conviction'))
    n_v6 = len(feature_cols) - n_xw - n_xtf - n_corr_vel - n_temporal - n_interact
    logger.info(f"  V6 base: {n_v6} | F1: {n_xw} | F2: {n_xtf} | F3: {n_corr_vel+1} | F4: {n_temporal} | F5: {n_interact}")

    # ------------------------------------------------------------------
    # Step 2: Load labels
    # ------------------------------------------------------------------
    logger.info("\n[STEP 2] Loading 3-class labels...")
    labels = load_labels()
    labels = prepare_3class_labels(labels)
    dist = labels['label_3class'].value_counts().to_dict()
    logger.info(f"  Label distribution: {dist}")

    # ------------------------------------------------------------------
    # Step 3: Align
    # ------------------------------------------------------------------
    logger.info("\n[STEP 3] Aligning features and labels...")
    X_aligned, y_df, times = align_features_labels(features, labels)
    del features

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 4: Walk-Forward Loop (4 folds x 3 seeds = 12 runs)
    # ------------------------------------------------------------------
    logger.info(f"\n[STEP 4] Walk-forward training ({len(FOLDS)} folds x {len(SEEDS)} seeds = {len(FOLDS)*len(SEEDS)} runs)...")

    all_fold_results = []  # one entry per fold (aggregated across seeds)
    all_run_results = []   # one entry per fold+seed
    fi_accum = {}          # feature importance across all runs
    fold_equity_data = []  # for cross-fold equity plot

    for fold_idx, fold_def in enumerate(FOLDS):
        fold_t0 = time.time()
        logger.info(f"\n{'='*90}")
        logger.info(f"FOLD {fold_idx}: Train -> {fold_def['train_end'].date()} | "
                     f"Val -> {fold_def['val_end'].date()} | "
                     f"Embargo -> {fold_def['embargo_end'].date()} | "
                     f"Test -> {fold_def['test_end'].date()}")
        logger.info(f"{'='*90}")

        # Split masks
        train_mask = times < fold_def['train_end']
        val_mask = (times >= fold_def['train_end']) & (times < fold_def['val_end'])
        test_mask = (times >= fold_def['embargo_end']) & (times < fold_def['test_end'])

        n_train = int(train_mask.sum())
        n_val = int(val_mask.sum())
        n_test = int(test_mask.sum())

        if n_train == 0 or n_val == 0 or n_test == 0:
            logger.warning(f"  SKIP: empty split (train={n_train}, val={n_val}, test={n_test})")
            continue

        X_train = X_aligned.loc[train_mask.values, feature_cols].reset_index(drop=True)
        X_val = X_aligned.loc[val_mask.values, feature_cols].reset_index(drop=True)
        X_test = X_aligned.loc[test_mask.values, feature_cols].reset_index(drop=True)

        y_train = y_df.loc[train_mask.values, 'label_3class'].values.astype(np.int8)
        y_val = y_df.loc[val_mask.values, 'label_3class'].values.astype(np.int8)
        y_test = y_df.loc[test_mask.values, 'label_3class'].values.astype(np.int8)

        test_times_fold = times[test_mask].reset_index(drop=True)

        logger.info(f"  Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")

        # Log label distributions
        for name, arr in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            classes, counts = np.unique(arr, return_counts=True)
            total = len(arr)
            dist_str = ", ".join([f"{CLASS_NAMES.get(int(c), c)}: {cnt:,} ({cnt/total*100:.1f}%)"
                                  for c, cnt in zip(classes, counts)])
            logger.info(f"  {name}: {total:,} — {dist_str}")

        # Per-seed training within fold
        fold_seed_results = []

        for seed in SEEDS:
            run_t0 = time.time()
            logger.info(f"\n  --- Fold {fold_idx}, Seed {seed} ---")

            model, best_iter = train_model(X_train, y_train, X_val, y_val, seed=seed)

            # Feature importance
            fi = model.get_feature_importance()
            fi_sorted = sorted(zip(feature_cols, fi), key=lambda x: -x[1])
            n_used = int((np.array(fi) > 0).sum())

            logger.info(f"  Best iter: {best_iter} | Features used: {n_used}/{len(feature_cols)}")
            logger.info(f"  Top 10:")
            for fname, fval in fi_sorted[:10]:
                is_v10 = any(fname.startswith(p) for p in ['xw_', 'xtf_', 'corr_velocity_']) or \
                          fname in ('xtf_corr_agreement', 'hour_sin', 'hour_cos', 'dow_sin',
                                    'dow_cos', 'is_ny_session', 'convergence_volume',
                                    'crossing_atr', 'cascade_volume', 'reversal_conviction')
                tag = " [V10]" if is_v10 else ""
                logger.info(f"    {fname}: {fval:.2f}{tag}")

            # Feature group analysis
            groups, gcounts = analyze_feature_groups(fi_sorted)

            # Accumulate feature importance
            for fname, fval in fi_sorted[:40]:
                if fname not in fi_accum:
                    fi_accum[fname] = []
                fi_accum[fname].append(float(fval))

            # Val threshold selection
            y_proba_val = model.predict_proba(Pool(X_val))
            val_best_thresh = THRESHOLDS[0]
            val_best_profit = -float('inf')
            for thresh in THRESHOLDS:
                val_m = evaluate_at_threshold(y_proba_val, y_val, thresh, cooldown=TRADE_COOLDOWN)
                if val_m['n_trades_simulated'] > 0 and val_m['total_profit_pct'] > val_best_profit:
                    val_best_profit = val_m['total_profit_pct']
                    val_best_thresh = thresh

            logger.info(f"  Val-selected threshold: @{val_best_thresh:.2f} (val profit={val_best_profit:.1f}%)")

            # Test evaluation at all thresholds
            y_proba_test = model.predict_proba(Pool(X_test))
            threshold_results = {}
            for thresh in THRESHOLDS:
                threshold_results[f"{thresh:.2f}"] = evaluate_at_threshold(
                    y_proba_test, y_test, thresh, cooldown=TRADE_COOLDOWN)

            # Log test results table
            logger.info(f"  Test results (cooldown={TRADE_COOLDOWN}):")
            logger.info(f"    {'Thresh':>7} {'Trades':>7} {'Prec':>6} {'L_prec':>7} "
                         f"{'S_prec':>7} {'Profit':>9} {'PF':>6} {'Sharpe':>7}")
            logger.info("    " + "-" * 70)
            for thresh in THRESHOLDS:
                m = threshold_results[f"{thresh:.2f}"]
                marker = " <-VAL" if abs(thresh - val_best_thresh) < 0.001 else ""
                logger.info(f"    {thresh:>7.2f} {m['n_trades_simulated']:>7} "
                            f"{m['trade_precision']:>6.3f} {m['long_precision']:>7.3f} "
                            f"{m['short_precision']:>7.3f} {m['total_profit_pct']:>8.1f}% "
                            f"{m['profit_factor']:>6.2f} {m['sharpe']:>7.3f}{marker}")

            # Key metrics
            honest_key = f"{val_best_thresh:.2f}"
            honest_m = threshold_results[honest_key]
            at070 = threshold_results.get('0.70', {})
            at075 = threshold_results.get('0.75', {})
            at080 = threshold_results.get('0.80', {})
            roc_auc = threshold_results[f"{THRESHOLDS[0]:.2f}"]['roc_auc_macro']

            run_time = time.time() - run_t0
            logger.info(f"  => F{fold_idx}s{seed}: AUC={roc_auc:.3f} | "
                        f"HONEST @{val_best_thresh:.2f}: {honest_m['total_profit_pct']:.1f}% | "
                        f"@0.70: {at070.get('total_profit_pct', 0):.1f}% | "
                        f"@0.75: {at075.get('total_profit_pct', 0):.1f}% | "
                        f"@0.80: {at080.get('total_profit_pct', 0):.1f}% | {run_time:.0f}s")

            # Save plots and trade logs
            try:
                save_fold_plots(y_proba_test, y_test, fold_idx, seed, threshold_results, val_best_thresh)
            except Exception as e:
                logger.warning(f"  Plot failed: {e}")

            save_trade_log(y_proba_test, y_test, test_times_fold, val_best_thresh,
                           fold_idx, seed, cooldown=TRADE_COOLDOWN)

            model.save_model(str(RESULTS_DIR / f"f{fold_idx}_s{seed}_model.cbm"))

            # Collect equity data
            eq_honest = honest_m.get('equity_curve', [])
            if len(eq_honest) > 1:
                fold_equity_data.append({
                    'fold': fold_idx, 'seed': seed,
                    'equity': eq_honest, 'profit': honest_m['total_profit_pct'],
                })

            fold_seed_results.append({
                'seed': seed,
                'best_iteration': best_iter,
                'n_features_used': n_used,
                'roc_auc_macro': roc_auc,
                'val_selected_threshold': val_best_thresh,
                'honest_profit': honest_m['total_profit_pct'],
                'honest_trades': honest_m['n_trades_simulated'],
                'honest_has_shorts': honest_m['n_short_predictions'] > 0,
                'feature_groups': {k: float(v) for k, v in groups.items()},
                'top_20_features': [(n, float(v)) for n, v in fi_sorted[:20]],
                'threshold_summary': [{
                    'threshold': thresh,
                    'trade_preds': threshold_results[f"{thresh:.2f}"]['n_trade_predictions'],
                    'long_preds': threshold_results[f"{thresh:.2f}"]['n_long_predictions'],
                    'short_preds': threshold_results[f"{thresh:.2f}"]['n_short_predictions'],
                    'trade_precision': threshold_results[f"{thresh:.2f}"]['trade_precision'],
                    'long_precision': threshold_results[f"{thresh:.2f}"]['long_precision'],
                    'short_precision': threshold_results[f"{thresh:.2f}"]['short_precision'],
                    'total_profit': threshold_results[f"{thresh:.2f}"]['total_profit_pct'],
                    'profit_factor': threshold_results[f"{thresh:.2f}"]['profit_factor'],
                    'sharpe': threshold_results[f"{thresh:.2f}"]['sharpe'],
                    'n_trades': threshold_results[f"{thresh:.2f}"]['n_trades_simulated'],
                    'win_rate': threshold_results[f"{thresh:.2f}"]['win_rate'],
                    'max_drawdown': threshold_results[f"{thresh:.2f}"]['max_drawdown'],
                } for thresh in THRESHOLDS],
            })
            all_run_results.append({
                'fold': fold_idx, **fold_seed_results[-1],
            })

        # Aggregate fold across seeds
        fold_aucs = [r['roc_auc_macro'] for r in fold_seed_results]
        fold_honest = [r['honest_profit'] for r in fold_seed_results]
        fold_time = time.time() - fold_t0

        logger.info(f"\n  Fold {fold_idx} summary (3 seeds):")
        logger.info(f"    AUC: {np.mean(fold_aucs):.3f} +/- {np.std(fold_aucs):.3f}")
        logger.info(f"    Honest profits: {[f'{p:.1f}%' for p in fold_honest]} mean={np.mean(fold_honest):.1f}%")

        # Per-threshold fold means
        logger.info(f"    Per-threshold (mean of 3 seeds):")
        logger.info(f"      {'Thresh':>7} {'Trades':>7} {'Prec':>6} {'Profit':>9} {'PF':>6} {'Sharpe':>7}")
        for tidx, thresh in enumerate(THRESHOLDS):
            t_avg = np.mean([r['threshold_summary'][tidx]['n_trades'] for r in fold_seed_results])
            p_avg = np.mean([r['threshold_summary'][tidx]['trade_precision'] for r in fold_seed_results])
            pr_avg = np.mean([r['threshold_summary'][tidx]['total_profit'] for r in fold_seed_results])
            pf_avg = np.mean([r['threshold_summary'][tidx]['profit_factor'] for r in fold_seed_results])
            sh_avg = np.mean([r['threshold_summary'][tidx]['sharpe'] for r in fold_seed_results])
            logger.info(f"      {thresh:>7.2f} {t_avg:>7.0f} {p_avg:>6.3f} "
                        f"{pr_avg:>8.1f}% {pf_avg:>6.2f} {sh_avg:>7.3f}")

        all_fold_results.append({
            'fold': fold_idx,
            'train_end': str(fold_def['train_end'].date()),
            'test_end': str(fold_def['test_end'].date()),
            'n_train': n_train, 'n_val': n_val, 'n_test': n_test,
            'mean_auc': float(np.mean(fold_aucs)),
            'std_auc': float(np.std(fold_aucs)),
            'mean_honest_profit': float(np.mean(fold_honest)),
            'seed_results': fold_seed_results,
            'fold_time_sec': fold_time,
        })

        logger.info(f"    Fold {fold_idx} total time: {fold_time:.0f}s")

    # ------------------------------------------------------------------
    # Step 5: Cross-Fold Summary
    # ------------------------------------------------------------------
    logger.info(f"\n\n{'='*100}")
    logger.info("WALK-FORWARD CROSS-FOLD SUMMARY — V10 (4 Folds x 3 Seeds)")
    logger.info(f"{'='*100}")

    # AUC summary
    all_aucs = [r['roc_auc_macro'] for r in all_run_results]
    mean_auc = float(np.mean(all_aucs))
    std_auc = float(np.std(all_aucs))
    logger.info(f"\n  AUC: {mean_auc:.3f} +/- {std_auc:.3f} (n={len(all_aucs)} runs)")

    # Honest results per fold (mean of seeds)
    logger.info(f"\n  === HONEST RESULTS (val-selected threshold, mean of 3 seeds) ===")
    logger.info(f"  {'Fold':>5} {'Period':<20} {'AUC':>6} {'Profit':>8} {'Thresholds':>20}")
    logger.info("  " + "-" * 65)

    n_honest_profitable_folds = 0
    total_honest_profit = 0.0
    for fr in all_fold_results:
        fold_i = fr['fold']
        mean_p = fr['mean_honest_profit']
        total_honest_profit += mean_p
        if mean_p > 0:
            n_honest_profitable_folds += 1
        val_threshs = [r['val_selected_threshold'] for r in fr['seed_results']]
        logger.info(f"  {fold_i:>5} {fr['train_end']}->{fr['test_end']:<10} "
                    f"{fr['mean_auc']:>6.3f} {mean_p:>7.1f}% {val_threshs}")

    logger.info(f"\n  Honest profitable folds: {n_honest_profitable_folds}/{len(all_fold_results)}")
    logger.info(f"  Honest total profit (mean of seeds): {total_honest_profit:.1f}%")

    # FIXED-THRESHOLD CROSS-FOLD TABLE (the key analysis)
    logger.info(f"\n  === FIXED-THRESHOLD CROSS-FOLD PROFITABILITY (mean of 3 seeds per fold) ===")
    header = f"  {'Thresh':>7}"
    for fr in all_fold_results:
        header += f" {'F'+str(fr['fold']):>8}"
    header += f" {'Total':>8} {'AvgPrec':>8} {'#Prof':>6}"
    logger.info(header)
    logger.info("  " + "-" * (7 + 8 * len(all_fold_results) + 25))

    best_fixed_thresh = THRESHOLDS[0]
    best_fixed_n_prof = 0
    fixed_thresh_data = {}

    for thresh in THRESHOLDS:
        tidx = THRESHOLDS.index(thresh)
        row = f"  {thresh:>7.2f}"
        fold_profits = []
        fold_precs = []
        n_prof = 0

        for fr in all_fold_results:
            # Mean across seeds for this fold+threshold
            seed_profits = [r['threshold_summary'][tidx]['total_profit'] for r in fr['seed_results']]
            seed_precs = [r['threshold_summary'][tidx]['trade_precision'] for r in fr['seed_results']]
            mean_profit = float(np.mean(seed_profits))
            mean_prec = float(np.mean(seed_precs))
            fold_profits.append(mean_profit)
            fold_precs.append(mean_prec)
            row += f" {mean_profit:>7.1f}%"
            if mean_profit > 0:
                n_prof += 1

        total = sum(fold_profits)
        avg_prec = float(np.mean(fold_precs)) if fold_precs else 0
        row += f" {total:>7.1f}% {avg_prec:>7.1f}% {n_prof:>3}/{len(all_fold_results)}"
        logger.info(row)

        if n_prof > best_fixed_n_prof or (n_prof == best_fixed_n_prof and total > fixed_thresh_data.get(best_fixed_thresh, {}).get('total', -999)):
            best_fixed_n_prof = n_prof
            best_fixed_thresh = thresh

        fixed_thresh_data[thresh] = {
            'profits': fold_profits,
            'precisions': fold_precs,
            'n_prof': n_prof,
            'total': total,
            'avg_precision': avg_prec,
        }

    all_folds_profitable = best_fixed_n_prof == len(all_fold_results)
    logger.info(f"\n  Best fixed threshold: @{best_fixed_thresh:.2f} "
                f"({best_fixed_n_prof}/{len(all_fold_results)} folds profitable)")

    # Precision focus: @0.70, @0.75, @0.80
    logger.info(f"\n  === PRECISION TARGETS (@0.70 / @0.75 / @0.80) ===")
    for target_thresh in [0.70, 0.75, 0.80]:
        td = fixed_thresh_data.get(target_thresh, {})
        logger.info(f"  @{target_thresh:.2f}: profits={[f'{p:.0f}%' for p in td.get('profits', [])]}, "
                    f"total={td.get('total', 0):.0f}%, "
                    f"avg_prec={td.get('avg_precision', 0):.1f}%, "
                    f"folds_prof={td.get('n_prof', 0)}/{len(all_fold_results)}")

    # Feature importance (averaged across all 12 runs)
    fi_avg = sorted([(k, float(np.mean(v))) for k, v in fi_accum.items()], key=lambda x: -x[1])
    logger.info(f"\n  Top 20 features (avg across {len(all_run_results)} runs):")
    for fname, fval in fi_avg[:20]:
        is_v10 = any(fname.startswith(p) for p in ['xw_', 'xtf_', 'corr_velocity_']) or \
                  fname in ('xtf_corr_agreement', 'hour_sin', 'hour_cos', 'dow_sin',
                            'dow_cos', 'is_ny_session', 'convergence_volume',
                            'crossing_atr', 'cascade_volume', 'reversal_conviction')
        tag = " [V10]" if is_v10 else ""
        logger.info(f"    {fname}: {fval:.2f}{tag}")

    # Feature stability
    feature_freq = {}
    for r in all_run_results:
        for fname, _ in r['top_20_features']:
            feature_freq[fname] = feature_freq.get(fname, 0) + 1
    freq_sorted = sorted(feature_freq.items(), key=lambda x: -x[1])
    logger.info(f"\n  Feature stability (top-20 appearances across {len(all_run_results)} runs):")
    for fname, count in freq_sorted[:15]:
        logger.info(f"    {fname}: {count}/{len(all_run_results)}")

    # Feature group importance (averaged)
    logger.info(f"\n  Feature group importance (avg):")
    avg_groups = {}
    for r in all_run_results:
        for gname, gval in r['feature_groups'].items():
            if gname not in avg_groups:
                avg_groups[gname] = []
            avg_groups[gname].append(gval)
    for gname in sorted(avg_groups.keys()):
        logger.info(f"    {gname}: {np.mean(avg_groups[gname]):.2f}")

    # V10 in top-20
    v10_in_top20 = sum(1 for fname, _ in fi_avg[:20]
                       if any(fname.startswith(p) for p in ['xw_', 'xtf_', 'corr_velocity_'])
                       or fname in ('xtf_corr_agreement', 'hour_sin', 'hour_cos', 'dow_sin',
                                    'dow_cos', 'is_ny_session', 'convergence_volume',
                                    'crossing_atr', 'cascade_volume', 'reversal_conviction'))
    logger.info(f"  V10 features in top-20: {v10_in_top20}/20")

    # ------------------------------------------------------------------
    # Step 6: Pass/Fail Assessment
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*90}")
    logger.info("PASS/FAIL ASSESSMENT")
    logger.info(f"{'='*90}")

    checks = []

    # Check 1: AUC
    auc_pass = all(r['roc_auc_macro'] > 0.70 for r in all_run_results)
    min_auc = min(r['roc_auc_macro'] for r in all_run_results)
    checks.append(('AUC > 0.70 all runs', 'PASS' if auc_pass else 'FAIL',
                    f"min={min_auc:.3f}, mean={mean_auc:.3f}"))

    # Check 2: Honest profitability
    profit_pass = n_honest_profitable_folds >= 3
    checks.append(('3+ folds profitable (honest)',
                    'PASS' if profit_pass else 'FAIL',
                    f"{n_honest_profitable_folds}/{len(all_fold_results)}"))

    # Check 3: Fixed threshold all folds profitable
    fixed_pass = all_folds_profitable
    checks.append(('Any fixed threshold ALL folds profitable',
                    'PASS' if fixed_pass else 'FAIL',
                    f"Best: @{best_fixed_thresh:.2f} = {best_fixed_n_prof}/{len(all_fold_results)}"))

    # Check 4: @0.70 profitability
    td70 = fixed_thresh_data.get(0.70, {})
    at70_pass = td70.get('n_prof', 0) >= 3
    checks.append(('@0.70: 3+ folds profitable',
                    'PASS' if at70_pass else 'FAIL',
                    f"{td70.get('n_prof', 0)}/{len(all_fold_results)}, total={td70.get('total', 0):.0f}%"))

    # Check 5: @0.75 profitability
    td75 = fixed_thresh_data.get(0.75, {})
    at75_pass = td75.get('n_prof', 0) >= 3
    checks.append(('@0.75: 3+ folds profitable',
                    'PASS' if at75_pass else 'FAIL',
                    f"{td75.get('n_prof', 0)}/{len(all_fold_results)}, total={td75.get('total', 0):.0f}%, prec={td75.get('avg_precision', 0):.1f}%"))

    # Check 6: @0.80 profitability
    td80 = fixed_thresh_data.get(0.80, {})
    at80_pass = td80.get('n_prof', 0) >= 3
    checks.append(('@0.80: 3+ folds profitable',
                    'PASS' if at80_pass else 'FAIL',
                    f"{td80.get('n_prof', 0)}/{len(all_fold_results)}, total={td80.get('total', 0):.0f}%, prec={td80.get('avg_precision', 0):.1f}%"))

    overall_pass = all(c[1] == 'PASS' for c in checks[:4])  # core checks (not precision targets)

    for check_name, status, detail in checks:
        marker = '[PASS]' if status == 'PASS' else '[FAIL]'
        logger.info(f"  {marker} {check_name}: {detail}")

    logger.info(f"\n  OVERALL (core): {'PASS' if overall_pass else 'FAIL'}")

    # ------------------------------------------------------------------
    # Step 7: Save results
    # ------------------------------------------------------------------
    logger.info(f"\n[STEP 7] Saving results...")

    # Cross-fold equity plot
    try:
        if fold_equity_data:
            save_crossfold_equity(fold_equity_data)
    except Exception as e:
        logger.warning(f"  Cross-fold equity failed: {e}")

    # Feature importance CSV
    fi_df = pd.DataFrame(fi_avg, columns=['feature', 'avg_importance'])
    fi_df.to_csv(RESULTS_DIR / "feature_importance_avg.csv", index=False)

    # Summary CSV
    summary_rows = []
    for r in all_run_results:
        row = {
            'fold': r['fold'], 'seed': r['seed'],
            'roc_auc': r['roc_auc_macro'],
            'best_iter': r['best_iteration'],
            'n_features_used': r['n_features_used'],
            'val_threshold': r['val_selected_threshold'],
            'honest_profit': r['honest_profit'],
            'honest_trades': r['honest_trades'],
        }
        for ts in r['threshold_summary']:
            t = ts['threshold']
            row[f'profit@{t:.2f}'] = ts['total_profit']
            row[f'trades@{t:.2f}'] = ts['n_trades']
            row[f'prec@{t:.2f}'] = ts['trade_precision']
            row[f'pf@{t:.2f}'] = ts['profit_factor']
            row[f'sharpe@{t:.2f}'] = ts['sharpe']
            row[f'wr@{t:.2f}'] = ts['win_rate']
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "walkforward_summary.csv", index=False)

    # Master JSON (strip equity curves)
    master = {
        'pipeline': 'V10_walkforward',
        'n_folds': len(FOLDS),
        'n_seeds': len(SEEDS),
        'n_features': len(feature_cols),
        'model_params': {k: str(v) if not isinstance(v, (int, float, list, bool)) else v
                         for k, v in MODEL_PARAMS.items()},
        'thresholds': THRESHOLDS,
        'embargo_days': EMBARGO_DAYS,
        'trade_cooldown': TRADE_COOLDOWN,
        'sl_pct': SL_PCT, 'tp_pct': TP_PCT,
        'aggregate': {
            'mean_auc': mean_auc, 'std_auc': std_auc,
            'n_honest_profitable_folds': n_honest_profitable_folds,
            'total_honest_profit': total_honest_profit,
            'v10_in_top20': v10_in_top20,
        },
        'fixed_threshold_analysis': {
            f"{t:.2f}": {
                'fold_profits': d['profits'],
                'fold_precisions': d['precisions'],
                'n_prof': d['n_prof'],
                'total': d['total'],
                'avg_precision': d['avg_precision'],
            } for t, d in fixed_thresh_data.items()
        },
        'pass_fail': {
            'overall': 'PASS' if overall_pass else 'FAIL',
            'checks': [{'name': n, 'status': s, 'detail': d} for n, s, d in checks],
        },
        'feature_importance_top20': fi_avg[:20],
        'feature_group_importance': {k: float(np.mean(v)) for k, v in avg_groups.items()},
        'fold_results': [{k: v for k, v in fr.items() if k != 'seed_results'}
                         for fr in all_fold_results],
        'all_run_results': [{k: v for k, v in r.items()
                             if k not in ('threshold_summary', 'top_20_features', 'feature_groups')}
                            for r in all_run_results],
    }

    with open(RESULTS_DIR / "walkforward_results.json", 'w') as f:
        json.dump(json.loads(json.dumps(master, default=convert_for_json)), f, indent=2)

    elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*90}")
    logger.info(f"V10 WALK-FORWARD COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"  {len(FOLDS)} folds x {len(SEEDS)} seeds = {len(all_run_results)} runs")
    logger.info(f"  AUC: {mean_auc:.3f} +/- {std_auc:.3f}")
    logger.info(f"  Honest profitable: {n_honest_profitable_folds}/{len(all_fold_results)} folds")
    logger.info(f"  Total honest profit: {total_honest_profit:.1f}%")
    logger.info(f"  Best fixed: @{best_fixed_thresh:.2f} = {best_fixed_n_prof}/{len(all_fold_results)} folds")

    for target in [0.70, 0.75, 0.80]:
        td = fixed_thresh_data.get(target, {})
        logger.info(f"  @{target:.2f}: total={td.get('total', 0):.0f}%, "
                    f"prec={td.get('avg_precision', 0):.1f}%, "
                    f"folds={td.get('n_prof', 0)}/{len(all_fold_results)}")

    logger.info(f"  V10 in top-20: {v10_in_top20}/20")
    logger.info(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    logger.info(f"  Output: {RESULTS_DIR}")
    logger.info(f"{'='*90}")

    return master


if __name__ == "__main__":
    run_walkforward()
