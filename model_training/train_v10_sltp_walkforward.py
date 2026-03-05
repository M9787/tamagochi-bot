"""
V10 SL/TP Target Sweep — Stage 2: Full Walk-Forward Validation

Runs full 4-fold x 3-seed walk-forward validation for winning configs from
Stage 1 screening. Parameterized version of train_v10_walkforward.py where
SL/TP/cooldown come from config rather than module constants.

Pass/fail criteria (per config, config-specific break-even):
  1. AUC > 0.70 in all 12 runs
  2. 3+ of 4 folds profitable (val-selected threshold)
  3. Any fixed threshold with ALL 4 folds profitable
  4. Avg precision > config break-even WR at best threshold
  5. At least one threshold with precision > 50%
  6. Total profit > 0

Output: results_v10/sltp_winners/C{N}/
Cross-config comparison: results_v10/sltp_winners/

Usage:
  python model_training/train_v10_sltp_walkforward.py --config C3
  python model_training/train_v10_sltp_walkforward.py --config C3 C6
  python model_training/train_v10_sltp_walkforward.py --config C3 C6 C7
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')

import argparse
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
    roc_auc_score, roc_curve, average_precision_score,
)

from data.target_labeling import create_sl_tp_labels

logger = logging.getLogger(__name__)

# ============================================================================
# Paths
# ============================================================================

ENCODED_DIR = Path(__file__).parent / "encoded_data"
ACTUAL_DATA_DIR = Path(__file__).parent / "actual_data"
WINNERS_DIR = Path(__file__).parent / "results_v10" / "sltp_winners"
LABEL_CACHE_DIR = Path(__file__).parent / "results_v10" / "sltp_screen" / "label_cache"

# ============================================================================
# Configs (same as Stage 1)
# ============================================================================

CONFIGS = {
    'C1':  {'sl': 0.5, 'tp': 1.0,  'max_hold': 72,  'cooldown': 15},
    'C2':  {'sl': 0.5, 'tp': 1.5,  'max_hold': 72,  'cooldown': 15},
    'C3':  {'sl': 1.0, 'tp': 2.0,  'max_hold': 144, 'cooldown': 30},
    'C4':  {'sl': 1.0, 'tp': 3.0,  'max_hold': 144, 'cooldown': 30},
    'C5':  {'sl': 1.0, 'tp': 4.0,  'max_hold': 144, 'cooldown': 30},
    'C6':  {'sl': 1.5, 'tp': 3.0,  'max_hold': 216, 'cooldown': 45},
    'C7':  {'sl': 1.5, 'tp': 4.5,  'max_hold': 216, 'cooldown': 45},
    'C8':  {'sl': 2.0, 'tp': 6.0,  'max_hold': 288, 'cooldown': 60},
    'C9':  {'sl': 1.0, 'tp': 5.0,  'max_hold': 144, 'cooldown': 30},
    'C10': {'sl': 0.5, 'tp': 2.0,  'max_hold': 72,  'cooldown': 15},
}

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

SEEDS = [42, 123, 777]
THRESHOLDS = [0.42, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
EMBARGO_DAYS = 7

# Walk-forward fold definitions (same as V10 WF)
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
# Label Generation with Disk Cache
# ============================================================================

def generate_labels_cached(config_name: str, config: dict, price_data: pd.DataFrame) -> pd.DataFrame:
    """Generate labels for a config, caching to disk. Reuses Stage 1 cache."""
    cache_path = LABEL_CACHE_DIR / f"labels_{config_name}.parquet"

    if cache_path.exists():
        logger.info(f"  Loading cached labels: {cache_path}")
        labels_df = pd.read_parquet(cache_path)
        labels_df['timestamp'] = pd.to_datetime(labels_df['timestamp'])
        return labels_df

    logger.info(f"  Generating labels: SL={config['sl']}%, TP={config['tp']}%, "
                f"max_hold={config['max_hold']}...")
    t0 = time.time()

    labels_df = create_sl_tp_labels(
        price_data,
        sl_pct=config['sl'],
        tp_pct=config['tp'],
        max_hold_periods=config['max_hold'],
        price_col='Close',
        high_col='High',
        low_col='Low',
        timestamp_col='Open Time',
    )

    elapsed = time.time() - t0
    logger.info(f"  Generated in {elapsed:.1f}s")

    LABEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    labels_df.to_parquet(cache_path, index=False)
    logger.info(f"  Cached: {cache_path}")

    return labels_df


# ============================================================================
# Label Preparation
# ============================================================================

def prepare_3class_labels(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    labels['label_3class'] = labels['label'].map({1: 1, -1: 2, 0: 0})
    return labels


def align_features_labels(features: pd.DataFrame, labels: pd.DataFrame):
    """Align features with labels by timestamp."""
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
# Evaluation — PARAMETERIZED sl_pct/tp_pct
# ============================================================================

def evaluate_at_threshold(y_pred_proba, y_test, threshold: float,
                          sl_pct: float, tp_pct: float, cooldown: int = 0):
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

    long_precision = float((y_test[y_pred == 1] == 1).mean()) if n_long_preds > 0 else 0.0
    short_precision = float((y_test[y_pred == 2] == 2).mean()) if n_short_preds > 0 else 0.0

    # Trading simulation WITH cooldown — parameterized sl_pct/tp_pct
    equity = [0.0]
    trades = []
    next_allowed_idx = 0
    for i in range(n_total):
        if y_pred[i] == 0:
            continue
        if i < next_allowed_idx:
            continue
        gain = tp_pct if y_pred[i] == y_test[i] else -sl_pct
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

def save_fold_plots(y_proba, y_test, fold_idx, seed, threshold_results,
                    val_thresh, results_dir: Path, config_name: str):
    """Save confusion matrix, ROC, equity PNGs for one fold+seed."""
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
    ax.set_title(f'{config_name} WF Confusion -- Fold {fold_idx} s{seed}')
    fig.tight_layout()
    fig.savefig(results_dir / f"{prefix}_confusion.png", dpi=150)
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
    ax.set_title(f'{config_name} WF ROC -- Fold {fold_idx} s{seed} (AUC={macro_auc:.3f})')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / f"{prefix}_roc.png", dpi=150)
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
            ax.set_title(f'{config_name} WF Equity -- Fold {fold_idx} s{seed} @{val_thresh:.2f} ({profit:.1f}%)')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
            fig.tight_layout()
            fig.savefig(results_dir / f"{prefix}_equity.png", dpi=150)
            plt.close(fig)


def save_trade_log(y_proba, y_test, test_times, threshold, fold_idx, seed,
                   sl_pct, tp_pct, cooldown, results_dir: Path):
    """Save per-trade CSV for one fold+seed."""
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
        gain = tp_pct if y_pred[i] == y_test[i] else -sl_pct
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
            results_dir / f"f{fold_idx}_s{seed}_trades.csv", index=False)


def save_crossfold_equity(fold_seed_equity_data, results_dir: Path, config_name: str):
    """Save combined equity curves across folds."""
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
    ax.set_title(f'{config_name} Walk-Forward Equity -- All Folds/Seeds')
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    fig.tight_layout()
    fig.savefig(results_dir / "crossfold_equity.png", dpi=150)
    plt.close(fig)


def save_cross_config_comparison(config_results: dict):
    """Save cross-config comparison CSV and PNG."""
    rows = []
    for cname, master in config_results.items():
        agg = master.get('aggregate', {})
        fta = master.get('fixed_threshold_analysis', {})
        pf = master.get('pass_fail', {})
        cfg = master.get('config', {})

        row = {
            'config': cname,
            'sl_pct': cfg.get('sl', 0),
            'tp_pct': cfg.get('tp', 0),
            'rr': cfg.get('tp', 0) / cfg.get('sl', 1),
            'break_even_wr': cfg.get('sl', 0) / (cfg.get('sl', 0) + cfg.get('tp', 1)) * 100,
            'mean_auc': agg.get('mean_auc', 0),
            'honest_profit': agg.get('total_honest_profit', 0),
            'overall': pf.get('overall', 'N/A'),
        }

        for thresh_key in ['0.70', '0.75', '0.80']:
            td = fta.get(thresh_key, {})
            row[f'profit@{thresh_key}'] = td.get('total', 0)
            row[f'prec@{thresh_key}'] = td.get('avg_precision', 0)
            row[f'folds_prof@{thresh_key}'] = td.get('n_prof', 0)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(WINNERS_DIR / "sltp_comparison.csv", index=False)

    # Comparison plot
    if len(rows) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    configs = [r['config'] for r in rows]
    n = len(configs)

    # Panel 1: Total profit at key thresholds
    ax = axes[0]
    x = np.arange(n)
    width = 0.25
    for i, thresh in enumerate(['0.70', '0.75', '0.80']):
        vals = [r.get(f'profit@{thresh}', 0) for r in rows]
        ax.bar(x + i * width, vals, width, label=f'@{thresh}', alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(configs)
    ax.set_ylabel('Total Profit (%)')
    ax.set_title('Profit by Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Precision at key thresholds
    ax = axes[1]
    for i, thresh in enumerate(['0.70', '0.75', '0.80']):
        vals = [r.get(f'prec@{thresh}', 0) * 100 for r in rows]
        ax.bar(x + i * width, vals, width, label=f'@{thresh}', alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(configs)
    ax.set_ylabel('Avg Precision (%)')
    ax.set_title('Precision by Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    # Draw break-even lines
    for j, r in enumerate(rows):
        ax.plot([j - 0.1, j + 0.6], [r['break_even_wr'], r['break_even_wr']],
                'r--', linewidth=1, alpha=0.5)

    # Panel 3: AUC comparison
    ax = axes[2]
    aucs = [r['mean_auc'] for r in rows]
    colors = ['#2ca02c' if r['overall'] == 'PASS' else '#d62728' for r in rows]
    ax.bar(configs, aucs, color=colors, alpha=0.8)
    ax.set_ylabel('Mean AUC')
    ax.set_title('Model AUC (green=PASS)')
    ax.axhline(y=0.70, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('V10 SL/TP Walk-Forward Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(WINNERS_DIR / "sltp_comparison.png", dpi=150)
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
# Walk-Forward for Single Config
# ============================================================================

def run_walkforward_config(config_name: str, config: dict,
                           features: pd.DataFrame, feature_cols: list,
                           price_data: pd.DataFrame) -> dict:
    """Run full 4-fold x 3-seed walk-forward for one config."""
    config_t0 = time.time()
    sl_pct = config['sl']
    tp_pct = config['tp']
    cooldown = config['cooldown']
    max_hold = config['max_hold']
    rr = tp_pct / sl_pct
    break_even = sl_pct / (sl_pct + tp_pct) * 100

    results_dir = WINNERS_DIR / config_name
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*90}")
    logger.info(f"WALK-FORWARD: {config_name} — SL={sl_pct}% TP={tp_pct}% RR={rr:.0f}:1 "
                f"MaxHold={max_hold} Cooldown={cooldown}")
    logger.info(f"  Break-even WR: {break_even:.1f}%")
    logger.info(f"  Output: {results_dir}")
    logger.info(f"{'='*90}")

    # Generate/load labels
    labels_df = generate_labels_cached(config_name, config, price_data)
    labels_df = prepare_3class_labels(labels_df)

    dist = labels_df['label_3class'].value_counts().to_dict()
    total = len(labels_df)
    logger.info(f"  Labels: {total:,} rows | "
                f"NT={dist.get(0,0)/total*100:.1f}% L={dist.get(1,0)/total*100:.1f}% "
                f"S={dist.get(2,0)/total*100:.1f}%")

    # Align
    X_aligned, y_df, times = align_features_labels(features, labels_df)

    # Walk-Forward Loop
    all_fold_results = []
    all_run_results = []
    fi_accum = {}
    fold_equity_data = []

    for fold_idx, fold_def in enumerate(FOLDS):
        fold_t0 = time.time()
        logger.info(f"\n{'='*80}")
        logger.info(f"FOLD {fold_idx}: Train -> {fold_def['train_end'].date()} | "
                     f"Val -> {fold_def['val_end'].date()} | "
                     f"Embargo -> {fold_def['embargo_end'].date()} | "
                     f"Test -> {fold_def['test_end'].date()}")

        # Split
        train_mask = times < fold_def['train_end']
        val_mask = (times >= fold_def['train_end']) & (times < fold_def['val_end'])
        test_mask = (times >= fold_def['embargo_end']) & (times < fold_def['test_end'])

        n_train = int(train_mask.sum())
        n_val = int(val_mask.sum())
        n_test = int(test_mask.sum())

        if n_train == 0 or n_val == 0 or n_test == 0:
            logger.warning(f"  SKIP: empty split")
            continue

        X_train = X_aligned.loc[train_mask.values, feature_cols].reset_index(drop=True)
        X_val = X_aligned.loc[val_mask.values, feature_cols].reset_index(drop=True)
        X_test = X_aligned.loc[test_mask.values, feature_cols].reset_index(drop=True)

        y_train = y_df.loc[train_mask.values, 'label_3class'].values.astype(np.int8)
        y_val = y_df.loc[val_mask.values, 'label_3class'].values.astype(np.int8)
        y_test = y_df.loc[test_mask.values, 'label_3class'].values.astype(np.int8)

        test_times_fold = times[test_mask].reset_index(drop=True)

        logger.info(f"  Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")

        for name, arr in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            classes, counts = np.unique(arr, return_counts=True)
            total_s = len(arr)
            dist_str = ", ".join([f"{CLASS_NAMES.get(int(c), c)}: {cnt:,} ({cnt/total_s*100:.1f}%)"
                                  for c, cnt in zip(classes, counts)])
            logger.info(f"  {name}: {total_s:,} -- {dist_str}")

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
            logger.info(f"  Top 5:")
            for fname, fval in fi_sorted[:5]:
                logger.info(f"    {fname}: {fval:.2f}")

            groups, gcounts = analyze_feature_groups(fi_sorted)

            for fname, fval in fi_sorted[:40]:
                if fname not in fi_accum:
                    fi_accum[fname] = []
                fi_accum[fname].append(float(fval))

            # Val threshold selection
            y_proba_val = model.predict_proba(Pool(X_val))
            val_best_thresh = THRESHOLDS[0]
            val_best_profit = -float('inf')
            for thresh in THRESHOLDS:
                val_m = evaluate_at_threshold(y_proba_val, y_val, thresh,
                                             sl_pct=sl_pct, tp_pct=tp_pct,
                                             cooldown=cooldown)
                if val_m['n_trades_simulated'] > 0 and val_m['total_profit_pct'] > val_best_profit:
                    val_best_profit = val_m['total_profit_pct']
                    val_best_thresh = thresh

            logger.info(f"  Val-selected threshold: @{val_best_thresh:.2f} (val profit={val_best_profit:.1f}%)")

            # Test evaluation
            y_proba_test = model.predict_proba(Pool(X_test))
            threshold_results = {}
            for thresh in THRESHOLDS:
                threshold_results[f"{thresh:.2f}"] = evaluate_at_threshold(
                    y_proba_test, y_test, thresh, sl_pct=sl_pct, tp_pct=tp_pct,
                    cooldown=cooldown)

            # Log results
            logger.info(f"  Test results (cooldown={cooldown}):")
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

            honest_key = f"{val_best_thresh:.2f}"
            honest_m = threshold_results[honest_key]
            roc_auc = threshold_results[f"{THRESHOLDS[0]:.2f}"]['roc_auc_macro']

            run_time = time.time() - run_t0
            logger.info(f"  => F{fold_idx}s{seed}: AUC={roc_auc:.3f} | "
                        f"HONEST @{val_best_thresh:.2f}: {honest_m['total_profit_pct']:.1f}% | {run_time:.0f}s")

            # Save plots and trade log
            try:
                save_fold_plots(y_proba_test, y_test, fold_idx, seed,
                                threshold_results, val_best_thresh,
                                results_dir, config_name)
            except Exception as e:
                logger.warning(f"  Plot failed: {e}")

            save_trade_log(y_proba_test, y_test, test_times_fold, val_best_thresh,
                           fold_idx, seed, sl_pct, tp_pct, cooldown, results_dir)

            model.save_model(str(results_dir / f"f{fold_idx}_s{seed}_model.cbm"))

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

        # Fold aggregate
        fold_aucs = [r['roc_auc_macro'] for r in fold_seed_results]
        fold_honest = [r['honest_profit'] for r in fold_seed_results]
        fold_time = time.time() - fold_t0

        logger.info(f"\n  Fold {fold_idx} summary (3 seeds):")
        logger.info(f"    AUC: {np.mean(fold_aucs):.3f} +/- {np.std(fold_aucs):.3f}")
        logger.info(f"    Honest profits: {[f'{p:.1f}%' for p in fold_honest]} mean={np.mean(fold_honest):.1f}%")

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

    # ============================================================
    # Cross-Fold Summary
    # ============================================================
    logger.info(f"\n\n{'='*100}")
    logger.info(f"{config_name} WALK-FORWARD CROSS-FOLD SUMMARY (4 Folds x 3 Seeds)")
    logger.info(f"{'='*100}")

    all_aucs = [r['roc_auc_macro'] for r in all_run_results]
    mean_auc = float(np.mean(all_aucs))
    std_auc = float(np.std(all_aucs))
    logger.info(f"\n  AUC: {mean_auc:.3f} +/- {std_auc:.3f} (n={len(all_aucs)} runs)")

    # Honest results
    logger.info(f"\n  === HONEST RESULTS (val-selected threshold, mean of 3 seeds) ===")
    n_honest_profitable_folds = 0
    total_honest_profit = 0.0
    for fr in all_fold_results:
        mean_p = fr['mean_honest_profit']
        total_honest_profit += mean_p
        if mean_p > 0:
            n_honest_profitable_folds += 1
        val_threshs = [r['val_selected_threshold'] for r in fr['seed_results']]
        logger.info(f"  F{fr['fold']}: {fr['train_end']}->{fr['test_end']} | "
                    f"AUC={fr['mean_auc']:.3f} | profit={mean_p:.1f}% | threshs={val_threshs}")

    logger.info(f"\n  Honest profitable: {n_honest_profitable_folds}/{len(all_fold_results)} | "
                f"Total: {total_honest_profit:.1f}%")

    # Fixed-threshold table
    logger.info(f"\n  === FIXED-THRESHOLD CROSS-FOLD PROFITABILITY ===")
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

    # Feature importance
    fi_avg = sorted([(k, float(np.mean(v))) for k, v in fi_accum.items()], key=lambda x: -x[1])
    logger.info(f"\n  Top 20 features (avg across {len(all_run_results)} runs):")
    for fname, fval in fi_avg[:20]:
        logger.info(f"    {fname}: {fval:.2f}")

    # ============================================================
    # Pass/Fail Assessment (config-specific break-even)
    # ============================================================
    logger.info(f"\n{'='*90}")
    logger.info(f"PASS/FAIL ASSESSMENT — {config_name} (break-even={break_even:.1f}%)")
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

    # Check 4: Avg precision > break-even at best threshold
    best_avg_prec = max(d.get('avg_precision', 0) for d in fixed_thresh_data.values())
    prec_pass = best_avg_prec * 100 > break_even
    checks.append((f'Avg precision > break-even ({break_even:.1f}%)',
                    'PASS' if prec_pass else 'FAIL',
                    f"best={best_avg_prec*100:.1f}%"))

    # Check 5: Any threshold with precision > 50%
    any_50 = any(d.get('avg_precision', 0) > 0.50 for d in fixed_thresh_data.values())
    checks.append(('Any threshold with precision > 50%',
                    'PASS' if any_50 else 'FAIL',
                    f"{'yes' if any_50 else 'no'}"))

    # Check 6: Total profit > 0
    best_total = max(d.get('total', 0) for d in fixed_thresh_data.values())
    total_pass = best_total > 0
    checks.append(('Total profit > 0 at any threshold',
                    'PASS' if total_pass else 'FAIL',
                    f"best={best_total:.1f}%"))

    overall_pass = all(c[1] == 'PASS' for c in checks)

    for check_name, status, detail in checks:
        marker = '[PASS]' if status == 'PASS' else '[FAIL]'
        logger.info(f"  {marker} {check_name}: {detail}")

    logger.info(f"\n  OVERALL: {'PASS' if overall_pass else 'FAIL'}")

    # ============================================================
    # Save Results
    # ============================================================
    try:
        if fold_equity_data:
            save_crossfold_equity(fold_equity_data, results_dir, config_name)
    except Exception as e:
        logger.warning(f"  Cross-fold equity failed: {e}")

    fi_df = pd.DataFrame(fi_avg, columns=['feature', 'avg_importance'])
    fi_df.to_csv(results_dir / "feature_importance_avg.csv", index=False)

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

    pd.DataFrame(summary_rows).to_csv(results_dir / "walkforward_summary.csv", index=False)

    # Master JSON
    master = {
        'pipeline': f'V10_sltp_walkforward_{config_name}',
        'config_name': config_name,
        'config': config,
        'n_folds': len(FOLDS),
        'n_seeds': len(SEEDS),
        'n_features': len(feature_cols),
        'model_params': {k: str(v) if not isinstance(v, (int, float, list, bool)) else v
                         for k, v in MODEL_PARAMS.items()},
        'thresholds': THRESHOLDS,
        'embargo_days': EMBARGO_DAYS,
        'sl_pct': sl_pct, 'tp_pct': tp_pct,
        'cooldown': cooldown,
        'break_even_wr': break_even,
        'aggregate': {
            'mean_auc': mean_auc, 'std_auc': std_auc,
            'n_honest_profitable_folds': n_honest_profitable_folds,
            'total_honest_profit': total_honest_profit,
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
        'fold_results': [{k: v for k, v in fr.items() if k != 'seed_results'}
                         for fr in all_fold_results],
        'all_run_results': [{k: v for k, v in r.items()
                             if k not in ('threshold_summary', 'top_20_features', 'feature_groups')}
                            for r in all_run_results],
    }

    with open(results_dir / "walkforward_results.json", 'w') as f:
        json.dump(json.loads(json.dumps(master, default=convert_for_json)), f, indent=2)

    config_time = time.time() - config_t0
    logger.info(f"\n  {config_name} complete in {config_time:.0f}s ({config_time/60:.1f}min)")
    logger.info(f"  Output: {results_dir}")

    return master


# ============================================================================
# Main
# ============================================================================

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="V10 SL/TP Walk-Forward Validation (Stage 2)")
    parser.add_argument('--config', nargs='+', required=True,
                        help='Config names to validate (e.g., C3 C6)')
    args = parser.parse_args()

    # Validate config names
    for c in args.config:
        if c not in CONFIGS:
            raise ValueError(f"Unknown config: {c}. Available: {list(CONFIGS.keys())}")

    t_start = time.time()

    logger.info("=" * 90)
    logger.info("V10 SL/TP TARGET SWEEP — STAGE 2: FULL WALK-FORWARD")
    logger.info(f"  Configs: {args.config}")
    logger.info(f"  {len(FOLDS)} folds x {len(SEEDS)} seeds = {len(FOLDS)*len(SEEDS)} runs per config")
    logger.info("=" * 90)

    # Load features (once)
    logger.info("\nLoading V10 feature matrix...")
    parquet_path = ENCODED_DIR / "feature_matrix_v10.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing: {parquet_path}\nRun: python model_training/encode_v10.py --force")

    features = pd.read_parquet(parquet_path)
    feature_cols = [c for c in features.columns if c != 'time']
    logger.info(f"  Shape: {features.shape} ({len(feature_cols)} features)")

    # Load price data (once)
    logger.info("Loading 5M price data...")
    ml_path = ACTUAL_DATA_DIR / "ml_data_5M.csv"
    if not ml_path.exists():
        raise FileNotFoundError(f"Missing: {ml_path}\nRun: python model_training/download_data.py")

    price_data = pd.read_csv(ml_path)
    if 'index' in price_data.columns:
        price_data = price_data.drop(columns=['index'])
    price_data['Open Time'] = pd.to_datetime(price_data['Open Time'])
    logger.info(f"  Price rows: {len(price_data):,}")

    WINNERS_DIR.mkdir(parents=True, exist_ok=True)

    # Run walk-forward for each config
    config_results = {}
    for config_name in args.config:
        config = CONFIGS[config_name]
        master = run_walkforward_config(config_name, config, features, feature_cols, price_data)
        config_results[config_name] = master

    # Cross-config comparison (if multiple)
    if len(config_results) > 1:
        logger.info(f"\n\n{'='*100}")
        logger.info("CROSS-CONFIG COMPARISON")
        logger.info(f"{'='*100}")

        logger.info(f"\n  {'Config':>6} {'SL%':>5} {'TP%':>5} {'RR':>4} {'BE%':>5} "
                     f"{'AUC':>6} {'Overall':>8} {'@0.70':>8} {'@0.75':>8} {'@0.80':>8}")
        logger.info("  " + "-" * 80)

        for cname, master in config_results.items():
            agg = master.get('aggregate', {})
            fta = master.get('fixed_threshold_analysis', {})
            pf_status = master.get('pass_fail', {}).get('overall', '?')
            cfg = master.get('config', {})
            be = cfg.get('sl', 0) / (cfg.get('sl', 0) + cfg.get('tp', 1)) * 100

            p70 = fta.get('0.70', {}).get('total', 0)
            p75 = fta.get('0.75', {}).get('total', 0)
            p80 = fta.get('0.80', {}).get('total', 0)

            logger.info(f"  {cname:>6} {cfg.get('sl',0):>5.1f} {cfg.get('tp',0):>5.1f} "
                        f"{cfg.get('tp',0)/cfg.get('sl',1):>3.0f}:1 {be:>4.1f}% "
                        f"{agg.get('mean_auc',0):>6.3f} {pf_status:>8} "
                        f"{p70:>7.0f}% {p75:>7.0f}% {p80:>7.0f}%")

        try:
            save_cross_config_comparison(config_results)
        except Exception as e:
            logger.warning(f"  Comparison plots failed: {e}")

    elapsed = time.time() - t_start

    logger.info(f"\n{'='*90}")
    logger.info(f"STAGE 2 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"  Configs validated: {list(config_results.keys())}")
    for cname, master in config_results.items():
        pf = master.get('pass_fail', {}).get('overall', '?')
        logger.info(f"  {cname}: {pf}")
    logger.info(f"  Output: {WINNERS_DIR}")
    logger.info(f"{'='*90}")


if __name__ == "__main__":
    main()
