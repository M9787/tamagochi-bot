"""
V10 SL/TP Target Sweep — Stage 1: Quick Screen

Tests 10 different SL/TP target definitions to find which labels the V10 model
(508 features) can predict most reliably. Features (X) stay constant — only
labels (y) change.

For each config: generate labels -> train 1 model (Fold 0, seed 42) ->
evaluate at 8 thresholds -> compute composite score -> rank.

10 Configurations:
  C1:  SL=0.5%, TP=1.0%,  RR=2:1, max_hold=72,  cooldown=15
  C2:  SL=0.5%, TP=1.5%,  RR=3:1, max_hold=72,  cooldown=15
  C3:  SL=1.0%, TP=2.0%,  RR=2:1, max_hold=144, cooldown=30
  C4:  SL=1.0%, TP=3.0%,  RR=3:1, max_hold=144, cooldown=30
  C5:  SL=1.0%, TP=4.0%,  RR=4:1, max_hold=144, cooldown=30
  C6:  SL=1.5%, TP=3.0%,  RR=2:1, max_hold=216, cooldown=45
  C7:  SL=1.5%, TP=4.5%,  RR=3:1, max_hold=216, cooldown=45
  C8:  SL=2.0%, TP=6.0%,  RR=3:1, max_hold=288, cooldown=60
  C9:  SL=1.0%, TP=5.0%,  RR=5:1, max_hold=144, cooldown=30
  C10: SL=0.5%, TP=2.0%,  RR=4:1, max_hold=72,  cooldown=15

Output: results_v10/sltp_screen/

Usage:
  python model_training/train_v10_sltp_screen.py
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
    roc_auc_score, roc_curve, average_precision_score,
)

from data.target_labeling import create_sl_tp_labels

logger = logging.getLogger(__name__)

# ============================================================================
# Paths
# ============================================================================

ENCODED_DIR = Path(__file__).parent / "encoded_data"
ACTUAL_DATA_DIR = Path(__file__).parent / "actual_data"
RESULTS_DIR = Path(__file__).parent / "results_v10" / "sltp_screen"
LABEL_CACHE_DIR = RESULTS_DIR / "label_cache"

# ============================================================================
# 10 SL/TP Configurations
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

# V7 d8 winner params (same as V10 WF)
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

SEED = 42
THRESHOLDS = [0.42, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
EMBARGO_DAYS = 7

# Fold 0 only for screening
FOLD_0 = {
    'train_end': pd.Timestamp('2025-01-01'),
    'val_end': pd.Timestamp('2025-02-01'),
    'embargo_end': pd.Timestamp('2025-02-08'),
    'test_end': pd.Timestamp('2025-05-01'),
}


# ============================================================================
# Label Generation with Disk Cache
# ============================================================================

def generate_labels_cached(config_name: str, config: dict, price_data: pd.DataFrame) -> pd.DataFrame:
    """Generate labels for a config, caching to disk."""
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
# Label Preparation (from train_v10_walkforward.py)
# ============================================================================

def prepare_3class_labels(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    labels['label_3class'] = labels['label'].map({1: 1, -1: 2, 0: 0})
    return labels


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
# Training (from train_v10_walkforward.py)
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
    """Evaluate 3-class predictions at a given confidence threshold.

    Unlike train_v10_walkforward.py, sl_pct and tp_pct are passed as parameters
    so different configs can use different SL/TP for profit simulation.
    """
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

    # Trading simulation WITH cooldown — uses parameterized sl_pct/tp_pct
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
        for cls in range(3):
            if y_onehot[:, cls].sum() > 0:
                roc_auc_per_class[CLASS_NAMES[cls]] = float(
                    roc_auc_score(y_onehot[:, cls], y_pred_proba[:, cls]))
            else:
                roc_auc_per_class[CLASS_NAMES[cls]] = 0.0
    except Exception as e:
        logger.warning(f"  ROC AUC failed: {e}")
        roc_auc_macro = 0.0
        roc_auc_per_class = {}

    return {
        'threshold': threshold,
        'accuracy': acc,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'roc_auc_macro': roc_auc_macro,
        'roc_auc_per_class': roc_auc_per_class,
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
# Scoring
# ============================================================================

def compute_composite_score(threshold_results: dict, thresholds: list,
                            min_trades: int = 10) -> dict:
    """Compute composite score = mean(profit * precision) across qualifying thresholds.

    Returns dict with composite_score and per-threshold details.
    """
    qualifying = []
    for thresh in thresholds:
        key = f"{thresh:.2f}"
        m = threshold_results[key]
        if m['n_trades_simulated'] >= min_trades:
            qualifying.append({
                'threshold': thresh,
                'profit': m['total_profit_pct'],
                'precision': m['trade_precision'],
                'product': m['total_profit_pct'] * m['trade_precision'],
                'n_trades': m['n_trades_simulated'],
                'win_rate': m['win_rate'],
                'profit_factor': m['profit_factor'],
                'sharpe': m['sharpe'],
            })

    if not qualifying:
        return {'composite_score': 0.0, 'qualifying_thresholds': 0, 'details': []}

    composite = float(np.mean([q['product'] for q in qualifying]))
    n_profitable = sum(1 for q in qualifying if q['profit'] > 0)

    return {
        'composite_score': composite,
        'qualifying_thresholds': len(qualifying),
        'profitable_thresholds': n_profitable,
        'details': qualifying,
    }


# ============================================================================
# Visualization
# ============================================================================

def save_equity_curve(equity: list, config_name: str, config: dict,
                      threshold: float, profit: float):
    """Save equity curve PNG for a config."""
    if len(equity) <= 1:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    equity_arr = np.array(equity)
    ax.plot(range(len(equity)), equity, color='navy', linewidth=1.2, label='Equity')

    peak = np.maximum.accumulate(equity_arr)
    dd = equity_arr - peak
    ax.fill_between(range(len(equity)), equity_arr, peak,
                    where=(dd < 0), color='red', alpha=0.15, label='Drawdown')

    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative Equity (%)')
    rr = config['tp'] / config['sl']
    ax.set_title(f'{config_name}: SL={config["sl"]}% TP={config["tp"]}% '
                 f'RR={rr:.0f}:1 @{threshold:.2f} ({profit:.1f}%)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"config_{config_name}_equity.png", dpi=150)
    plt.close(fig)


def save_comparison_plot(ranking: list, label_dists: dict):
    """Save 4-panel comparison across all configs."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    configs_sorted = [r['config'] for r in ranking]
    n = len(configs_sorted)

    # Panel 1: Composite Score
    ax = axes[0, 0]
    scores = [r['composite_score'] for r in ranking]
    colors = ['#2ca02c' if s > 0 else '#d62728' for s in scores]
    ax.barh(range(n), scores, color=colors, alpha=0.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(configs_sorted)
    ax.set_xlabel('Composite Score (mean profit x precision)')
    ax.set_title('Composite Score Ranking')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Panel 2: Label Distribution
    ax = axes[0, 1]
    for i, cname in enumerate(configs_sorted):
        dist = label_dists.get(cname, {})
        nt = dist.get('NO_TRADE_pct', 0)
        lo = dist.get('LONG_pct', 0)
        sh = dist.get('SHORT_pct', 0)
        ax.barh(i, nt, color='gray', alpha=0.7, label='NO_TRADE' if i == 0 else '')
        ax.barh(i, lo, left=nt, color='green', alpha=0.7, label='LONG' if i == 0 else '')
        ax.barh(i, sh, left=nt + lo, color='red', alpha=0.7, label='SHORT' if i == 0 else '')
    ax.set_yticks(range(n))
    ax.set_yticklabels(configs_sorted)
    ax.set_xlabel('Label Distribution (%)')
    ax.set_title('Label Balance per Config')
    ax.invert_yaxis()
    ax.legend(loc='lower right')

    # Panel 3: AUC
    ax = axes[1, 0]
    aucs = [r['auc_macro'] for r in ranking]
    ax.barh(range(n), aucs, color='#1f77b4', alpha=0.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(configs_sorted)
    ax.set_xlabel('ROC AUC (macro)')
    ax.set_title('Model Discrimination (AUC)')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(x=0.70, color='red', linestyle='--', alpha=0.5, label='Min=0.70')
    ax.legend()

    # Panel 4: Best Threshold Profit
    ax = axes[1, 1]
    profits = [r['best_profit'] for r in ranking]
    colors = ['#2ca02c' if p > 0 else '#d62728' for p in profits]
    ax.barh(range(n), profits, color=colors, alpha=0.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(configs_sorted)
    ax.set_xlabel('Best Threshold Total Profit (%)')
    ax.set_title('Best Profit per Config')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    fig.suptitle('V10 SL/TP Target Sweep — Stage 1 Screening', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "screening_comparison.png", dpi=150)
    plt.close(fig)


# ============================================================================
# Trade Log
# ============================================================================

def save_trade_log(y_proba, y_test, test_times, threshold, config_name,
                   sl_pct, tp_pct, cooldown=0):
    """Save per-trade CSV for one config at best threshold."""
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
            RESULTS_DIR / f"config_{config_name}_trades.csv", index=False)


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
# Screen Single Config
# ============================================================================

def screen_single_config(config_name: str, config: dict,
                         features: pd.DataFrame, feature_cols: list,
                         price_data: pd.DataFrame) -> dict:
    """Full screening pipeline for one config. Returns results dict."""
    config_t0 = time.time()
    sl_pct = config['sl']
    tp_pct = config['tp']
    cooldown = config['cooldown']
    max_hold = config['max_hold']
    rr = tp_pct / sl_pct
    break_even = sl_pct / (sl_pct + tp_pct) * 100

    logger.info(f"\n{'='*80}")
    logger.info(f"CONFIG {config_name}: SL={sl_pct}% TP={tp_pct}% RR={rr:.0f}:1 "
                f"MaxHold={max_hold} Cooldown={cooldown} BreakEven={break_even:.1f}%")
    logger.info(f"{'='*80}")

    # 1. Generate labels
    labels_df = generate_labels_cached(config_name, config, price_data)
    labels_df = prepare_3class_labels(labels_df)

    dist = labels_df['label_3class'].value_counts().to_dict()
    total = len(labels_df)
    dist_pct = {CLASS_NAMES.get(k, k): f"{v/total*100:.1f}%" for k, v in dist.items()}
    logger.info(f"  Label distribution: {dist_pct}")

    label_dist = {
        'total': total,
        'NO_TRADE': int(dist.get(0, 0)),
        'LONG': int(dist.get(1, 0)),
        'SHORT': int(dist.get(2, 0)),
        'NO_TRADE_pct': dist.get(0, 0) / total * 100,
        'LONG_pct': dist.get(1, 0) / total * 100,
        'SHORT_pct': dist.get(2, 0) / total * 100,
    }

    # 2. Align
    X_aligned, y_df, times = align_features_labels(features, labels_df)

    # 3. Split (Fold 0)
    train_mask = times < FOLD_0['train_end']
    val_mask = (times >= FOLD_0['train_end']) & (times < FOLD_0['val_end'])
    test_mask = (times >= FOLD_0['embargo_end']) & (times < FOLD_0['test_end'])

    n_train = int(train_mask.sum())
    n_val = int(val_mask.sum())
    n_test = int(test_mask.sum())

    if n_train == 0 or n_val == 0 or n_test == 0:
        logger.warning(f"  SKIP: empty split (train={n_train}, val={n_val}, test={n_test})")
        return {'config': config_name, 'status': 'SKIP', 'composite_score': 0.0}

    X_train = X_aligned.loc[train_mask.values, feature_cols].reset_index(drop=True)
    X_val = X_aligned.loc[val_mask.values, feature_cols].reset_index(drop=True)
    X_test = X_aligned.loc[test_mask.values, feature_cols].reset_index(drop=True)

    y_train = y_df.loc[train_mask.values, 'label_3class'].values.astype(np.int8)
    y_val = y_df.loc[val_mask.values, 'label_3class'].values.astype(np.int8)
    y_test = y_df.loc[test_mask.values, 'label_3class'].values.astype(np.int8)

    test_times = times[test_mask].reset_index(drop=True)

    logger.info(f"  Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")

    # Log label distributions per split
    for name, arr in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        classes, counts = np.unique(arr, return_counts=True)
        total_split = len(arr)
        dist_str = ", ".join([f"{CLASS_NAMES.get(int(c), c)}: {cnt:,} ({cnt/total_split*100:.1f}%)"
                              for c, cnt in zip(classes, counts)])
        logger.info(f"  {name}: {total_split:,} -- {dist_str}")

    # 4. Train
    model, best_iter = train_model(X_train, y_train, X_val, y_val, seed=SEED)

    # Feature importance
    fi = model.get_feature_importance()
    fi_sorted = sorted(zip(feature_cols, fi), key=lambda x: -x[1])
    n_used = int((np.array(fi) > 0).sum())
    logger.info(f"  Features used: {n_used}/{len(feature_cols)} | Best iter: {best_iter}")
    logger.info(f"  Top 5:")
    for fname, fval in fi_sorted[:5]:
        logger.info(f"    {fname}: {fval:.2f}")

    # 5. Evaluate at all thresholds
    y_proba_test = model.predict_proba(Pool(X_test))
    threshold_results = {}
    for thresh in THRESHOLDS:
        threshold_results[f"{thresh:.2f}"] = evaluate_at_threshold(
            y_proba_test, y_test, thresh, sl_pct=sl_pct, tp_pct=tp_pct,
            cooldown=cooldown)

    # Log results table
    logger.info(f"  Test results (cooldown={cooldown}):")
    logger.info(f"    {'Thresh':>7} {'Trades':>7} {'Prec':>6} {'L_prec':>7} "
                f"{'S_prec':>7} {'Profit':>9} {'PF':>6} {'Sharpe':>7}")
    logger.info("    " + "-" * 70)

    best_profit = -float('inf')
    best_profit_thresh = THRESHOLDS[0]

    for thresh in THRESHOLDS:
        m = threshold_results[f"{thresh:.2f}"]
        logger.info(f"    {thresh:>7.2f} {m['n_trades_simulated']:>7} "
                    f"{m['trade_precision']:>6.3f} {m['long_precision']:>7.3f} "
                    f"{m['short_precision']:>7.3f} {m['total_profit_pct']:>8.1f}% "
                    f"{m['profit_factor']:>6.2f} {m['sharpe']:>7.3f}")
        if m['n_trades_simulated'] >= 10 and m['total_profit_pct'] > best_profit:
            best_profit = m['total_profit_pct']
            best_profit_thresh = thresh

    # 6. Composite score
    score_data = compute_composite_score(threshold_results, THRESHOLDS)
    auc_macro = threshold_results[f"{THRESHOLDS[0]:.2f}"]['roc_auc_macro']

    # Best precision across thresholds with >=10 trades
    best_prec = 0.0
    for thresh in THRESHOLDS:
        m = threshold_results[f"{thresh:.2f}"]
        if m['n_trades_simulated'] >= 10 and m['trade_precision'] > best_prec:
            best_prec = m['trade_precision']

    break_even_margin = best_prec * 100 - break_even

    logger.info(f"\n  => {config_name} SUMMARY:")
    logger.info(f"     AUC={auc_macro:.3f} | Composite={score_data['composite_score']:.1f}")
    logger.info(f"     Best profit: {best_profit:.1f}% @{best_profit_thresh:.2f}")
    logger.info(f"     Best precision: {best_prec:.1%} | Break-even margin: {break_even_margin:.1f}pp")
    logger.info(f"     Profitable thresholds: {score_data['profitable_thresholds']}/{score_data['qualifying_thresholds']}")

    # 7. Save equity curve and trade log (at best profit threshold)
    best_key = f"{best_profit_thresh:.2f}"
    if best_key in threshold_results:
        eq = threshold_results[best_key].get('equity_curve', [])
        save_equity_curve(eq, config_name, config, best_profit_thresh, best_profit)

    save_trade_log(y_proba_test, y_test, test_times, best_profit_thresh,
                   config_name, sl_pct, tp_pct, cooldown=cooldown)

    config_time = time.time() - config_t0

    return {
        'config': config_name,
        'status': 'OK',
        'sl_pct': sl_pct,
        'tp_pct': tp_pct,
        'rr': rr,
        'max_hold': max_hold,
        'cooldown': cooldown,
        'break_even_wr': break_even,
        'auc_macro': auc_macro,
        'composite_score': score_data['composite_score'],
        'qualifying_thresholds': score_data['qualifying_thresholds'],
        'profitable_thresholds': score_data['profitable_thresholds'],
        'best_profit': best_profit if best_profit > -float('inf') else 0.0,
        'best_profit_threshold': best_profit_thresh,
        'best_precision': best_prec,
        'break_even_margin': break_even_margin,
        'n_features_used': n_used,
        'best_iteration': best_iter,
        'label_distribution': label_dist,
        'threshold_details': score_data['details'],
        'threshold_results': {k: {kk: vv for kk, vv in v.items() if kk != 'equity_curve'}
                              for k, v in threshold_results.items()},
        'top_10_features': [(n, float(v)) for n, v in fi_sorted[:10]],
        'runtime_sec': config_time,
    }


# ============================================================================
# Main
# ============================================================================

def run_screening():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    t_start = time.time()

    logger.info("=" * 90)
    logger.info("V10 SL/TP TARGET SWEEP — STAGE 1: QUICK SCREEN")
    logger.info(f"  10 Configs | Fold 0 | Seed {SEED} | {len(THRESHOLDS)} thresholds")
    logger.info("=" * 90)

    # ------------------------------------------------------------------
    # Step 1: Load V10 features (once)
    # ------------------------------------------------------------------
    logger.info("\n[STEP 1] Loading V10 feature matrix...")
    parquet_path = ENCODED_DIR / "feature_matrix_v10.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing: {parquet_path}\nRun: python model_training/encode_v10.py --force")

    features = pd.read_parquet(parquet_path)
    feature_cols = [c for c in features.columns if c != 'time']
    logger.info(f"  Shape: {features.shape} ({len(feature_cols)} features)")

    # ------------------------------------------------------------------
    # Step 2: Load 5M price data (once, for label generation)
    # ------------------------------------------------------------------
    logger.info("\n[STEP 2] Loading 5M price data...")
    ml_path = ACTUAL_DATA_DIR / "ml_data_5M.csv"
    if not ml_path.exists():
        raise FileNotFoundError(f"Missing: {ml_path}\nRun: python model_training/download_data.py")

    price_data = pd.read_csv(ml_path)
    if 'index' in price_data.columns:
        price_data = price_data.drop(columns=['index'])
    price_data['Open Time'] = pd.to_datetime(price_data['Open Time'])
    logger.info(f"  Price rows: {len(price_data):,} ({price_data['Open Time'].min()} to {price_data['Open Time'].max()})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 3: Screen each config
    # ------------------------------------------------------------------
    logger.info(f"\n[STEP 3] Screening {len(CONFIGS)} configs...")

    all_results = []
    label_dists = {}

    for config_name in sorted(CONFIGS.keys(), key=lambda x: int(x[1:])):
        config = CONFIGS[config_name]
        result = screen_single_config(config_name, config, features, feature_cols, price_data)
        all_results.append(result)
        if 'label_distribution' in result:
            label_dists[config_name] = result['label_distribution']

    # ------------------------------------------------------------------
    # Step 4: Rank and Report
    # ------------------------------------------------------------------
    logger.info(f"\n\n{'='*100}")
    logger.info("SCREENING RESULTS — RANKED BY COMPOSITE SCORE")
    logger.info(f"{'='*100}")

    ranking = sorted(all_results, key=lambda x: x.get('composite_score', 0), reverse=True)

    logger.info(f"\n  {'Rank':>4} {'Config':>6} {'SL%':>5} {'TP%':>5} {'RR':>4} "
                f"{'AUC':>6} {'Composite':>10} {'BestProfit':>10} {'BestPrec':>9} "
                f"{'BE_Margin':>10} {'#Prof':>6}")
    logger.info("  " + "-" * 95)

    for i, r in enumerate(ranking):
        if r.get('status') == 'SKIP':
            logger.info(f"  {i+1:>4} {r['config']:>6} SKIPPED")
            continue
        logger.info(f"  {i+1:>4} {r['config']:>6} {r['sl_pct']:>5.1f} {r['tp_pct']:>5.1f} "
                    f"{r['rr']:>3.0f}:1 {r['auc_macro']:>6.3f} "
                    f"{r['composite_score']:>10.1f} {r['best_profit']:>9.1f}% "
                    f"{r['best_precision']:>8.1%} {r['break_even_margin']:>9.1f}pp "
                    f"{r['profitable_thresholds']:>3}/{r['qualifying_thresholds']}")

    # Recommend top 2-3 for Stage 2
    winners = [r for r in ranking if r.get('status') == 'OK' and r.get('composite_score', 0) > 0][:3]

    logger.info(f"\n  RECOMMENDED FOR STAGE 2 (full walk-forward):")
    for r in winners:
        logger.info(f"    {r['config']}: SL={r['sl_pct']}% TP={r['tp_pct']}% "
                    f"| score={r['composite_score']:.1f} | profit={r['best_profit']:.1f}% "
                    f"| prec={r['best_precision']:.1%}")

    if winners:
        cmd_configs = " ".join([r['config'] for r in winners])
        logger.info(f"\n  Run: python model_training/train_v10_sltp_walkforward.py --config {cmd_configs}")

    # ------------------------------------------------------------------
    # Step 5: Save outputs
    # ------------------------------------------------------------------
    logger.info(f"\n[STEP 5] Saving outputs...")

    # Comparison plot
    try:
        save_comparison_plot(ranking, label_dists)
    except Exception as e:
        logger.warning(f"  Comparison plot failed: {e}")

    # Ranking CSV
    ranking_rows = []
    for i, r in enumerate(ranking):
        if r.get('status') == 'SKIP':
            continue
        ranking_rows.append({
            'rank': i + 1,
            'config': r['config'],
            'sl_pct': r['sl_pct'],
            'tp_pct': r['tp_pct'],
            'rr': r['rr'],
            'max_hold': r['max_hold'],
            'cooldown': r['cooldown'],
            'break_even_wr': r['break_even_wr'],
            'auc_macro': r['auc_macro'],
            'composite_score': r['composite_score'],
            'best_profit': r['best_profit'],
            'best_profit_threshold': r['best_profit_threshold'],
            'best_precision': r['best_precision'],
            'break_even_margin': r['break_even_margin'],
            'qualifying_thresholds': r['qualifying_thresholds'],
            'profitable_thresholds': r['profitable_thresholds'],
            'n_features_used': r['n_features_used'],
            'runtime_sec': r['runtime_sec'],
        })
    pd.DataFrame(ranking_rows).to_csv(RESULTS_DIR / "screening_ranking.csv", index=False)

    # Label distributions CSV
    dist_rows = []
    for cname, dist in label_dists.items():
        dist_rows.append({'config': cname, **dist})
    pd.DataFrame(dist_rows).to_csv(RESULTS_DIR / "label_distributions.csv", index=False)

    # Full results JSON (strip equity curves)
    results_json = []
    for r in all_results:
        r_clean = {k: v for k, v in r.items() if k != 'threshold_results'}
        results_json.append(r_clean)

    with open(RESULTS_DIR / "screening_results.json", 'w') as f:
        json.dump(json.loads(json.dumps(results_json, default=convert_for_json)), f, indent=2)

    elapsed = time.time() - t_start

    logger.info(f"\n{'='*90}")
    logger.info(f"SCREENING COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"  {len(all_results)} configs screened")
    logger.info(f"  Winners: {[r['config'] for r in winners]}")
    logger.info(f"  Output: {RESULTS_DIR}")
    logger.info(f"{'='*90}")

    return ranking


if __name__ == "__main__":
    run_screening()
