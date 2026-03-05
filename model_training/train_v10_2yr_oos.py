"""
V10 2-Year Out-of-Sample Final Exam — 4yr Train, 2yr Test

The production gate test. Train on 4 years of data (Feb 2020 -> Feb 2024),
test on 2 years truly unseen (Mar 2024 -> Feb 2026). The test period covers
ETF approval, halving, post-halving rally, ATH, and recent chop.

If this passes -> train production model on ALL 6yr -> deploy live.

Split:
  Train: 2020-02-17 -> 2024-02-01 (4 years)
  Val:   2024-02-01 -> 2024-03-01 (1 month, for early stopping + threshold selection)
  Embargo: 2024-03-01 -> 2024-03-08 (7 days)
  Test:  2024-03-08 -> 2026-02-15 (~2 years, truly unseen)

Config: V7 d8 (depth=8, iter=5000, lr=0.02, l2=15, cw=[0.5,2,2])
Seeds: 42, 123, 777 (3 runs)

Pass/Fail Criteria:
  C1: AUC >= 0.80 mean across 3 seeds
  C2: Profit @0.70 > 0% mean across seeds
  C3: Profit @0.75 > 0% mean across seeds
  C5: Precision @0.70 >= 45% mean across seeds
  C6: Both LONG and SHORT present
  PASS = C1+C2+C3+C5+C6 all pass

Output: results_v10/2yr_oos/

Usage:
  python model_training/train_v10_2yr_oos.py
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
RESULTS_DIR = Path(__file__).parent / "results_v10" / "2yr_oos"

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

# Seeds
SEEDS = [42, 123, 777]

# Thresholds to evaluate
THRESHOLDS = [0.42, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

# Trade cooldown: 60 candles = 5 hours at 5M
TRADE_COOLDOWN = 60

# Single split definition — 4yr train, 2yr test
SPLIT = {
    'train_start': pd.Timestamp('2020-02-17'),   # Actual 5M data start
    'train_end': pd.Timestamp('2024-02-01'),      # 4 years training
    'val_end': pd.Timestamp('2024-03-01'),        # 1 month val
    'embargo_end': pd.Timestamp('2024-03-08'),    # 7 day embargo
    'test_end': pd.Timestamp('2026-02-15'),       # ~2 years truly unseen test
}

# Regime periods for analysis
REGIME_PERIODS = [
    ('2024-Q1', '2024-03-08', '2024-07-01', 'ETF approval + pre-halving'),
    ('2024-Q3', '2024-07-01', '2025-01-01', 'Post-halving rally'),
    ('2025-H1', '2025-01-01', '2025-07-01', 'ATH + chop'),
    ('2025-H2+', '2025-07-01', '2026-02-15', 'Recent mixed'),
]

# V10-specific feature groups for logging
V10_FEATURE_GROUPS = {
    'xw_': 'F1: Per-TF cross-window',
    'xtf_': 'F2: Cross-TF composites',
    'corr_velocity_': 'F3: Correlation velocity',
    'xtf_corr_agreement': 'F3: Correlation agreement',
    'hour_': 'F4: Temporal (hour)',
    'dow_': 'F4: Temporal (dow)',
    'is_ny_session': 'F4: Temporal (session)',
    'convergence_volume': 'F5: Interaction',
    'crossing_atr': 'F5: Interaction',
    'cascade_volume': 'F5: Interaction',
    'reversal_conviction': 'F5: Interaction',
}


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
        'trades_detail': trades if n_trades > 0 else [],
    }


# ============================================================================
# Val Threshold Selection
# ============================================================================

def select_val_threshold(y_proba_val, y_val, thresholds, cooldown):
    """Select the best threshold based on val set profit."""
    best_thresh = thresholds[0]
    best_profit = -float('inf')
    val_results = {}

    for thresh in thresholds:
        val_metrics = evaluate_at_threshold(y_proba_val, y_val, thresh, cooldown=cooldown)
        val_results[f"{thresh:.2f}"] = val_metrics
        if val_metrics['n_trades_simulated'] > 0 and val_metrics['total_profit_pct'] > best_profit:
            best_profit = val_metrics['total_profit_pct']
            best_thresh = thresh

    return best_thresh, best_profit, val_results


# ============================================================================
# Regime Analysis
# ============================================================================

def analyze_regimes(trades_detail, test_times, threshold):
    """Analyze trading performance per market regime period."""
    if not trades_detail or len(test_times) == 0:
        return []

    regime_results = []
    for name, start_str, end_str, desc in REGIME_PERIODS:
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str)

        regime_trades = []
        for t in trades_detail:
            idx = t['index']
            if idx < len(test_times):
                ts = test_times.iloc[idx]
                if start <= ts < end:
                    regime_trades.append(t)

        n_trades = len(regime_trades)
        if n_trades > 0:
            trades_df = pd.DataFrame(regime_trades)
            wins = (trades_df['gain_pct'] > 0).sum()
            profit = float(trades_df['gain_pct'].sum())
            win_rate = wins / n_trades * 100
            n_long = int((trades_df['predicted'] == 1).sum())
            n_short = int((trades_df['predicted'] == 2).sum())
            correct = sum(1 for t in regime_trades if t['predicted'] == t['actual'])
            precision = correct / n_trades * 100
        else:
            profit = win_rate = precision = 0.0
            n_long = n_short = 0

        regime_results.append({
            'name': name,
            'description': desc,
            'start': start_str,
            'end': end_str,
            'n_trades': n_trades,
            'n_long': n_long,
            'n_short': n_short,
            'profit_pct': profit,
            'win_rate': win_rate,
            'precision': precision,
        })

    return regime_results


# ============================================================================
# Pass/Fail Criteria
# ============================================================================

def check_pass_fail(seed_results, all_threshold_results):
    """Check the 6 pass/fail criteria. Returns (passed, details)."""
    checks = {}

    # C1: AUC >= 0.80 mean across 3 seeds
    aucs = [sr['roc_auc_macro'] for sr in seed_results]
    mean_auc = float(np.mean(aucs))
    checks['C1_auc'] = {
        'criterion': 'AUC >= 0.80 mean',
        'value': mean_auc,
        'passed': mean_auc >= 0.80,
    }

    # C2: Profit @0.70 > 0% mean across seeds
    profits_070 = [sr['profit_at_070'] for sr in seed_results]
    mean_070 = float(np.mean(profits_070))
    checks['C2_profit_070'] = {
        'criterion': 'Profit @0.70 > 0% mean',
        'value': mean_070,
        'passed': mean_070 > 0,
    }

    # C3: Profit @0.75 > 0% mean across seeds
    profits_075 = [sr['profit_at_075'] for sr in seed_results]
    mean_075 = float(np.mean(profits_075))
    checks['C3_profit_075'] = {
        'criterion': 'Profit @0.75 > 0% mean',
        'value': mean_075,
        'passed': mean_075 > 0,
    }

    # C5: Precision @0.70 >= 45% mean across seeds
    prec_070 = [sr['precision_at_070'] for sr in seed_results]
    mean_prec = float(np.mean(prec_070))
    checks['C5_precision_070'] = {
        'criterion': 'Precision @0.70 >= 45% mean',
        'value': mean_prec,
        'passed': mean_prec >= 0.45,
    }

    # C6: Both LONG and SHORT present (across all seeds)
    total_long = sum(sr.get('total_long_070', 0) for sr in seed_results)
    total_short = sum(sr.get('total_short_070', 0) for sr in seed_results)
    checks['C6_directionality'] = {
        'criterion': 'Both LONG and SHORT present',
        'value': f"LONG={total_long}, SHORT={total_short}",
        'passed': total_long > 0 and total_short > 0,
    }

    all_passed = all(c['passed'] for c in checks.values())
    return all_passed, checks


# ============================================================================
# Visualization
# ============================================================================

def save_plots(y_proba, y_test, test_times, threshold_results, seed, output_dir,
               val_selected_threshold=None):
    """Save confusion matrix, ROC, PR, equity PNGs."""
    n_classes = 3
    y_onehot = np.zeros((len(y_test), n_classes))
    for i in range(len(y_test)):
        y_onehot[i, y_test[i]] = 1
    class_colors = {'NO_TRADE': 'gray', 'LONG': 'green', 'SHORT': 'red'}
    prefix = f"s{seed}"

    # Confusion Matrix (at lowest threshold)
    first_key = f"{THRESHOLDS[0]:.2f}"
    cm = np.array(threshold_results[first_key]['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NO_TRADE', 'LONG', 'SHORT'],
                yticklabels=['NO_TRADE', 'LONG', 'SHORT'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'V10 2yr OOS Confusion — s{seed}')
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion.png", dpi=150)
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
    ax.set_title(f'V10 2yr OOS ROC — s{seed} (Macro AUC={macro_auc:.3f})')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_roc.png", dpi=150)
    plt.close(fig)

    # PR Curve
    fig, ax = plt.subplots(figsize=(8, 7))
    for cls in range(n_classes):
        name = CLASS_NAMES[cls]
        if y_onehot[:, cls].sum() > 0:
            prec_arr, rec_arr, _ = precision_recall_curve(
                y_onehot[:, cls], y_proba[:, cls])
            ap = average_precision_score(y_onehot[:, cls], y_proba[:, cls])
            ax.plot(rec_arr, prec_arr, color=class_colors[name], linewidth=2,
                    label=f'{name} (AP={ap:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'V10 2yr OOS PR — s{seed}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_pr.png", dpi=150)
    plt.close(fig)

    # Equity curves for all thresholds on ONE plot
    fig, ax = plt.subplots(figsize=(16, 7))
    colors = ['#e41a1c', '#ff7f00', '#984ea3', '#377eb8', '#4daf4a', '#000000',
              '#a65628', '#f781bf']
    for idx, thresh in enumerate(THRESHOLDS):
        tkey = f"{thresh:.2f}"
        if tkey in threshold_results:
            eq = threshold_results[tkey].get('equity_curve', [])
            profit = threshold_results[tkey].get('total_profit_pct', 0)
            trades_n = threshold_results[tkey].get('n_trades_simulated', 0)
            if len(eq) > 1:
                lw = 2.5 if val_selected_threshold and abs(thresh - val_selected_threshold) < 0.001 else 1.2
                marker = " (VAL)" if val_selected_threshold and abs(thresh - val_selected_threshold) < 0.001 else ""
                ax.plot(range(len(eq)), eq, color=colors[idx % len(colors)],
                        linewidth=lw, alpha=0.85,
                        label=f'@{thresh:.2f}: {profit:+.1f}% ({trades_n}t){marker}')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative Equity (%)')
    ax.set_title(f'V10 2yr OOS Equity — s{seed} (all thresholds)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_equity_all.png", dpi=150)
    plt.close(fig)

    # Equity curve for val-selected threshold (detailed with drawdown)
    if val_selected_threshold is not None:
        best_key = f"{val_selected_threshold:.2f}"
        if best_key in threshold_results:
            eq = threshold_results[best_key].get('equity_curve', [])
            profit = threshold_results[best_key].get('total_profit_pct', 0)
            if len(eq) > 1:
                fig, ax = plt.subplots(figsize=(16, 7))
                equity_arr = np.array(eq)
                ax.plot(range(len(eq)), eq, color='navy', linewidth=1.2, label='Equity')
                peak = np.maximum.accumulate(equity_arr)
                dd = equity_arr - peak
                ax.fill_between(range(len(eq)), equity_arr, peak,
                                where=(dd < 0), color='red', alpha=0.15, label='Drawdown')
                ax.set_xlabel('Trade #')
                ax.set_ylabel('Cumulative Equity (%)')
                ax.set_title(f'V10 2yr OOS Equity — s{seed} @{val_selected_threshold:.2f} '
                             f'(profit={profit:.1f}%)')
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
                fig.tight_layout()
                fig.savefig(output_dir / f"{prefix}_equity_val.png", dpi=150)
                plt.close(fig)

    # Monthly profit breakdown
    if val_selected_threshold is not None:
        best_key = f"{val_selected_threshold:.2f}"
        if best_key in threshold_results:
            trades_detail = threshold_results[best_key].get('trades_detail', [])
            if trades_detail and len(test_times) > 0:
                _save_monthly_breakdown(trades_detail, test_times, seed, output_dir,
                                        val_selected_threshold)


def _save_monthly_breakdown(trades_detail, test_times, seed, output_dir, threshold):
    """Save monthly profit bar chart."""
    rows = []
    for t in trades_detail:
        idx = t['index']
        if idx < len(test_times):
            ts = test_times.iloc[idx]
            rows.append({
                'month': ts.strftime('%Y-%m'),
                'gain_pct': t['gain_pct'],
                'direction': CLASS_NAMES[t['predicted']],
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    monthly = df.groupby('month').agg(
        profit=('gain_pct', 'sum'),
        trades=('gain_pct', 'count'),
        wins=('gain_pct', lambda x: (x > 0).sum()),
    ).reset_index()
    monthly['wr'] = (monthly['wins'] / monthly['trades'] * 100).round(1)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['green' if p > 0 else 'red' for p in monthly['profit']]
    bars = ax.bar(range(len(monthly)), monthly['profit'], color=colors, alpha=0.7, edgecolor='black')

    for i, (_, row) in enumerate(monthly.iterrows()):
        ax.text(i, row['profit'] + (1 if row['profit'] >= 0 else -2),
                f"{row['trades']}t\n{row['wr']:.0f}%", ha='center', va='bottom' if row['profit'] >= 0 else 'top',
                fontsize=7)

    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly['month'], rotation=45, ha='right')
    ax.set_ylabel('Profit (%)')
    ax.set_title(f'V10 2yr OOS Monthly Profit — s{seed} @{threshold:.2f}')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(output_dir / f"s{seed}_monthly.png", dpi=150)
    plt.close(fig)

    # Save monthly CSV
    monthly.to_csv(output_dir / f"s{seed}_monthly.csv", index=False)


def save_trade_log(y_proba, y_test, test_times, threshold, seed, output_dir, cooldown=0):
    """Save per-trade CSV at given threshold with cooldown."""
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
            output_dir / f"s{seed}_trades.csv", index=False)
    return rows


def save_cross_seed_equity(seed_results, output_dir):
    """Save combined equity curves across all seeds on one plot."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    seed_colors = {42: '#1f77b4', 123: '#ff7f0e', 777: '#2ca02c'}

    # Left: val-selected threshold
    ax = axes[0]
    for sr in seed_results:
        seed = sr['seed']
        thresh = sr['val_selected_threshold']
        eq = sr['val_equity_curve']
        profit = sr['val_selected_profit']
        if len(eq) > 1:
            ax.plot(range(len(eq)), eq, color=seed_colors.get(seed, 'gray'),
                    linewidth=1.5, label=f's{seed} @{thresh:.2f}: {profit:+.1f}%')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative Equity (%)')
    ax.set_title('V10 2yr OOS — Val-Selected Threshold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')

    # Middle: fixed @0.70
    ax = axes[1]
    for sr in seed_results:
        seed = sr['seed']
        eq70 = sr.get('equity_at_070', [])
        profit70 = sr.get('profit_at_070', 0)
        if len(eq70) > 1:
            ax.plot(range(len(eq70)), eq70, color=seed_colors.get(seed, 'gray'),
                    linewidth=1.5, label=f's{seed} @0.70: {profit70:+.1f}%')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative Equity (%)')
    ax.set_title('V10 2yr OOS — Fixed @0.70')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')

    # Right: fixed @0.75 (default production threshold)
    ax = axes[2]
    for sr in seed_results:
        seed = sr['seed']
        eq75 = sr.get('equity_at_075', [])
        profit75 = sr.get('profit_at_075', 0)
        if len(eq75) > 1:
            ax.plot(range(len(eq75)), eq75, color=seed_colors.get(seed, 'gray'),
                    linewidth=1.5, label=f's{seed} @0.75: {profit75:+.1f}%')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative Equity (%)')
    ax.set_title('V10 2yr OOS — Fixed @0.75 (Production)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')

    fig.tight_layout()
    fig.savefig(output_dir / "cross_seed_equity.png", dpi=150)
    plt.close(fig)


# ============================================================================
# V10-specific: Feature Group Analysis
# ============================================================================

def analyze_feature_groups(fi_sorted, feature_cols):
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

def run_2yr_oos():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    t_start = time.time()

    logger.info("=" * 90)
    logger.info("V10 2-YEAR OUT-OF-SAMPLE FINAL EXAM — Production Gate Test")
    logger.info(f"  Config: V7 d8 (depth={MODEL_PARAMS['depth']}, iter={MODEL_PARAMS['iterations']})")
    logger.info(f"  Seeds: {SEEDS}")
    logger.info(f"  Train: {SPLIT['train_start'].date()} -> {SPLIT['train_end'].date()} (4yr)")
    logger.info(f"  Val:   {SPLIT['train_end'].date()} -> {SPLIT['val_end'].date()} (1mo)")
    logger.info(f"  Embargo: {SPLIT['val_end'].date()} -> {SPLIT['embargo_end'].date()} (7d)")
    logger.info(f"  Test:  {SPLIT['embargo_end'].date()} -> {SPLIT['test_end'].date()} (~2yr)")
    logger.info(f"  SL={SL_PCT}% | TP={TP_PCT}% | Cooldown={TRADE_COOLDOWN} | Thresholds={THRESHOLDS}")
    logger.info(f"  Regime periods: {[r[0] for r in REGIME_PERIODS]}")
    logger.info("=" * 90)

    # ------------------------------------------------------------------
    # Step 1: Load features
    # ------------------------------------------------------------------
    logger.info("\n[STEP 1] Loading V10 feature matrix...")
    parquet_path = ENCODED_DIR / "feature_matrix_v10.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing: {parquet_path}\nRun: python model_training/encode_v10.py --force")

    features = pd.read_parquet(parquet_path)
    feature_cols = [c for c in features.columns if c != 'time']
    logger.info(f"  Shape: {features.shape} ({len(feature_cols)} features)")

    # Log V10-specific feature counts
    n_xw = sum(1 for c in feature_cols if c.startswith('xw_'))
    n_xtf = sum(1 for c in feature_cols if c.startswith('xtf_'))
    n_corr_vel = sum(1 for c in feature_cols if 'corr_velocity' in c)
    n_temporal = sum(1 for c in feature_cols if c in ('hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_ny_session'))
    n_interact = sum(1 for c in feature_cols if c in ('convergence_volume', 'crossing_atr', 'cascade_volume', 'reversal_conviction'))
    n_v6 = len(feature_cols) - n_xw - n_xtf - n_corr_vel - n_temporal - n_interact
    logger.info(f"  Feature breakdown:")
    logger.info(f"    V6 base:          {n_v6}")
    logger.info(f"    F1 cross-window:  {n_xw}")
    logger.info(f"    F2 cross-TF:      {n_xtf}")
    logger.info(f"    F3 corr dynamics: {n_corr_vel + 1}")
    logger.info(f"    F4 temporal:      {n_temporal}")
    logger.info(f"    F5 interactions:  {n_interact}")

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
    del features  # free memory

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 4: Split data
    # ------------------------------------------------------------------
    logger.info("\n[STEP 4] Splitting data...")

    train_mask = (times >= SPLIT['train_start']) & (times < SPLIT['train_end'])
    val_mask = (times >= SPLIT['train_end']) & (times < SPLIT['val_end'])
    embargo_mask = (times >= SPLIT['val_end']) & (times < SPLIT['embargo_end'])
    test_mask = (times >= SPLIT['embargo_end']) & (times < SPLIT['test_end'])

    n_train = int(train_mask.sum())
    n_val = int(val_mask.sum())
    n_embargo = int(embargo_mask.sum())
    n_test = int(test_mask.sum())

    logger.info(f"  Train:   {n_train:>8,} rows ({SPLIT['train_start'].date()} -> {SPLIT['train_end'].date()})")
    logger.info(f"  Val:     {n_val:>8,} rows ({SPLIT['train_end'].date()} -> {SPLIT['val_end'].date()})")
    logger.info(f"  Embargo: {n_embargo:>8,} rows ({SPLIT['val_end'].date()} -> {SPLIT['embargo_end'].date()})")
    logger.info(f"  Test:    {n_test:>8,} rows ({SPLIT['embargo_end'].date()} -> {SPLIT['test_end'].date()})")

    if n_train == 0 or n_val == 0 or n_test == 0:
        raise ValueError(f"Empty split: train={n_train}, val={n_val}, test={n_test}")

    X_train = X_aligned.loc[train_mask.values, feature_cols].reset_index(drop=True)
    X_val = X_aligned.loc[val_mask.values, feature_cols].reset_index(drop=True)
    X_test = X_aligned.loc[test_mask.values, feature_cols].reset_index(drop=True)

    y_train = y_df.loc[train_mask.values, 'label_3class'].values.astype(np.int8)
    y_val = y_df.loc[val_mask.values, 'label_3class'].values.astype(np.int8)
    y_test = y_df.loc[test_mask.values, 'label_3class'].values.astype(np.int8)

    test_times = times[test_mask].reset_index(drop=True)

    # Label distributions
    for name, arr in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        unique, counts = np.unique(arr, return_counts=True)
        dist_str = ", ".join([f"{CLASS_NAMES[int(u)]}={c}" for u, c in zip(unique, counts)])
        logger.info(f"  {name} labels: {dist_str}")

    # ------------------------------------------------------------------
    # Step 5: Train and evaluate per seed
    # ------------------------------------------------------------------
    logger.info(f"\n[STEP 5] Training 3 seeds...")

    seed_results = []
    all_fi = {}

    for seed in SEEDS:
        logger.info(f"\n  {'='*70}")
        logger.info(f"  SEED {seed}")
        logger.info(f"  {'='*70}")

        seed_t0 = time.time()

        # Train
        train_pool = Pool(X_train, y_train)
        eval_pool = Pool(X_val, y_val)

        params = MODEL_PARAMS.copy()
        params['random_seed'] = seed

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=eval_pool, use_best_model=True)

        best_iter = model.get_best_iteration()
        if best_iter < MIN_ITERATIONS:
            logger.warning(f"  best_iter={best_iter} < {MIN_ITERATIONS}, retraining...")
            params_min = params.copy()
            params_min['iterations'] = MIN_ITERATIONS
            params_min.pop('early_stopping_rounds', None)
            model = CatBoostClassifier(**params_min)
            model.fit(train_pool)
            best_iter = MIN_ITERATIONS

        logger.info(f"  Best iteration: {best_iter}")

        # Feature importance
        fi = model.get_feature_importance()
        fi_sorted = sorted(zip(feature_cols, fi), key=lambda x: -x[1])
        n_used = int((np.array(fi) > 0).sum())
        logger.info(f"  Features used: {n_used}/{len(feature_cols)}")

        # Top 20 features
        logger.info(f"  Top 20 features:")
        for fname, fval in fi_sorted[:20]:
            is_v10 = any(fname.startswith(p) for p in ['xw_', 'xtf_', 'corr_velocity_']) or \
                      fname in ('xtf_corr_agreement', 'hour_sin', 'hour_cos', 'dow_sin',
                                'dow_cos', 'is_ny_session', 'convergence_volume',
                                'crossing_atr', 'cascade_volume', 'reversal_conviction')
            tag = " [V10]" if is_v10 else ""
            logger.info(f"    {fname}: {fval:.2f}{tag}")
            if fname not in all_fi:
                all_fi[fname] = []
            all_fi[fname].append(fval)

        # Also accumulate features outside top 20
        for fname, fval in fi_sorted[20:40]:
            if fname not in all_fi:
                all_fi[fname] = []
            all_fi[fname].append(fval)

        # Feature group analysis (V10-specific)
        groups, group_counts = analyze_feature_groups(fi_sorted, feature_cols)
        logger.info(f"\n  Feature group importance:")
        for gname, gval in sorted(groups.items(), key=lambda x: -x[1]):
            cnt = group_counts[gname]
            logger.info(f"    {gname}: {gval:.2f} ({cnt} features)")

        # Val threshold selection
        y_proba_val = model.predict_proba(Pool(X_val))
        val_best_thresh, val_best_profit, val_results = select_val_threshold(
            y_proba_val, y_val, THRESHOLDS, TRADE_COOLDOWN)

        logger.info(f"\n  Val threshold scan (cooldown={TRADE_COOLDOWN}):")
        for thresh in THRESHOLDS:
            vm = val_results[f"{thresh:.2f}"]
            marker = " <--" if abs(thresh - val_best_thresh) < 0.001 else ""
            logger.info(f"    @{thresh:.2f}: {vm['n_trades_simulated']:>5} trades, "
                        f"prec={vm['trade_precision']:.3f}, profit={vm['total_profit_pct']:.1f}%{marker}")

        # Test evaluation at all thresholds
        y_proba_test = model.predict_proba(Pool(X_test))
        threshold_results = {}
        for thresh in THRESHOLDS:
            metrics = evaluate_at_threshold(y_proba_test, y_test, thresh, cooldown=TRADE_COOLDOWN)
            threshold_results[f"{thresh:.2f}"] = metrics

        # Log test results
        logger.info(f"\n  Test results (cooldown={TRADE_COOLDOWN}):")
        logger.info(f"    {'Thresh':>7} {'Trades':>7} {'L/S':>8} {'Prec':>6} {'L_prec':>7} {'S_prec':>7} {'Profit%':>9} {'PF':>6} {'Sharpe':>7} {'WR':>6}")
        logger.info("    " + "-" * 85)
        for thresh in THRESHOLDS:
            tkey = f"{thresh:.2f}"
            m = threshold_results[tkey]
            marker = " <-- VAL" if abs(thresh - val_best_thresh) < 0.001 else ""
            logger.info(f"    {thresh:>7.2f} {m['n_trades_simulated']:>7} "
                        f"{m['n_long_predictions']:>3}/{m['n_short_predictions']:<4} "
                        f"{m['trade_precision']:>6.3f} "
                        f"{m['long_precision']:>7.3f} {m['short_precision']:>7.3f} "
                        f"{m['total_profit_pct']:>8.1f}% "
                        f"{m['profit_factor']:>6.2f} {m['sharpe']:>7.3f} "
                        f"{m['win_rate']:>5.1f}%{marker}")

        # Honest metrics
        honest_key = f"{val_best_thresh:.2f}"
        honest_metrics = threshold_results[honest_key]
        honest_profit = honest_metrics['total_profit_pct']
        honest_trades = honest_metrics['n_trades_simulated']

        # @0.70 metrics
        at070 = threshold_results.get('0.70', {})
        # @0.75 metrics (production threshold)
        at075 = threshold_results.get('0.75', {})

        seed_time = time.time() - seed_t0

        logger.info(f"\n  => s{seed}: AUC={threshold_results[f'{THRESHOLDS[0]:.2f}']['roc_auc_macro']:.3f} | "
                    f"HONEST @{val_best_thresh:.2f}: {honest_profit:.1f}% ({honest_trades} trades) | "
                    f"@0.70: {at070.get('total_profit_pct', 0):.1f}% | "
                    f"@0.75: {at075.get('total_profit_pct', 0):.1f}% | "
                    f"Time: {seed_time:.0f}s")

        # Regime analysis at @0.75
        regime_075 = analyze_regimes(
            at075.get('trades_detail', []), test_times, 0.75)
        if regime_075:
            logger.info(f"\n  Regime analysis @0.75:")
            for r in regime_075:
                logger.info(f"    {r['name']} ({r['description']}): "
                            f"{r['n_trades']}t, L={r['n_long']}/S={r['n_short']}, "
                            f"profit={r['profit_pct']:+.1f}%, "
                            f"prec={r['precision']:.1f}%, wr={r['win_rate']:.1f}%")

        # Save model
        model.save_model(str(RESULTS_DIR / f"s{seed}_model.cbm"))

        # Save plots
        try:
            save_plots(y_proba_test, y_test, test_times, threshold_results,
                       seed, RESULTS_DIR, val_best_thresh)
        except Exception as e:
            logger.warning(f"  Plot save failed: {e}")

        # Save trade logs (at val-selected, @0.70, and @0.75)
        save_trade_log(y_proba_test, y_test, test_times, val_best_thresh,
                       seed, RESULTS_DIR, cooldown=TRADE_COOLDOWN)

        for extra_thresh, suffix in [(0.70, 'at070'), (0.75, 'at075')]:
            if abs(val_best_thresh - extra_thresh) > 0.001:
                _save_extra_trade_log(y_proba_test, y_test, test_times,
                                      extra_thresh, seed, RESULTS_DIR, suffix)

        seed_results.append({
            'seed': seed,
            'best_iteration': best_iter,
            'n_features_used': n_used,
            'roc_auc_macro': threshold_results[f'{THRESHOLDS[0]:.2f}']['roc_auc_macro'],
            'roc_auc_per_class': threshold_results[f'{THRESHOLDS[0]:.2f}']['roc_auc_per_class'],
            'val_selected_threshold': val_best_thresh,
            'val_selected_profit': honest_profit,
            'val_selected_trades': honest_trades,
            'profit_at_070': at070.get('total_profit_pct', 0),
            'trades_at_070': at070.get('n_trades_simulated', 0),
            'precision_at_070': at070.get('trade_precision', 0),
            'total_long_070': at070.get('n_long_predictions', 0),
            'total_short_070': at070.get('n_short_predictions', 0),
            'equity_at_070': at070.get('equity_curve', []),
            'profit_at_075': at075.get('total_profit_pct', 0),
            'trades_at_075': at075.get('n_trades_simulated', 0),
            'precision_at_075': at075.get('trade_precision', 0),
            'equity_at_075': at075.get('equity_curve', []),
            'val_equity_curve': honest_metrics.get('equity_curve', []),
            'regime_analysis_075': regime_075,
            'threshold_summary': [],
            'top_20_features': [(n, float(v)) for n, v in fi_sorted[:20]],
            'feature_groups': {k: float(v) for k, v in groups.items()},
            'seed_time_sec': seed_time,
        })

        # Build threshold summary for this seed
        for thresh in THRESHOLDS:
            tkey = f"{thresh:.2f}"
            m = threshold_results[tkey]
            seed_results[-1]['threshold_summary'].append({
                'threshold': thresh,
                'n_trades': m['n_trades_simulated'],
                'n_long': m['n_long_predictions'],
                'n_short': m['n_short_predictions'],
                'trade_precision': m['trade_precision'],
                'long_precision': m['long_precision'],
                'short_precision': m['short_precision'],
                'profit': m['total_profit_pct'],
                'profit_factor': m['profit_factor'],
                'sharpe': m['sharpe'],
                'win_rate': m['win_rate'],
                'max_drawdown': m['max_drawdown'],
            })

    # ------------------------------------------------------------------
    # Step 6: Cross-seed aggregation
    # ------------------------------------------------------------------
    logger.info(f"\n\n{'='*90}")
    logger.info("CROSS-SEED AGGREGATION")
    logger.info(f"{'='*90}")

    aucs = [sr['roc_auc_macro'] for sr in seed_results]
    honest_profits = [sr['val_selected_profit'] for sr in seed_results]
    profits_070 = [sr['profit_at_070'] for sr in seed_results]
    profits_075 = [sr['profit_at_075'] for sr in seed_results]
    val_thresholds = [sr['val_selected_threshold'] for sr in seed_results]

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    mean_honest = float(np.mean(honest_profits))
    mean_070 = float(np.mean(profits_070))
    mean_075 = float(np.mean(profits_075))
    cv_honest = (float(np.std(honest_profits)) / abs(mean_honest) * 100) if abs(mean_honest) > 0.001 else float('inf')

    logger.info(f"\n  AUC: {mean_auc:.3f} +/- {std_auc:.3f} ({[f'{a:.3f}' for a in aucs]})")
    logger.info(f"  Val thresholds selected: {val_thresholds}")
    logger.info(f"  Honest profits: {[f'{p:.1f}%' for p in honest_profits]} (mean={mean_honest:.1f}%)")
    logger.info(f"  @0.70 profits: {[f'{p:.1f}%' for p in profits_070]} (mean={mean_070:.1f}%)")
    logger.info(f"  @0.75 profits: {[f'{p:.1f}%' for p in profits_075]} (mean={mean_075:.1f}%)")
    logger.info(f"  Seed CV (honest): {cv_honest:.1f}%")

    # Per-threshold cross-seed table
    logger.info(f"\n  Per-threshold (mean across seeds):")
    logger.info(f"    {'Thresh':>7} {'Trades':>7} {'Prec':>6} {'L_prec':>7} {'S_prec':>7} {'Profit':>9} {'PF':>6} {'Sharpe':>7}")
    logger.info("    " + "-" * 70)
    for tidx, thresh in enumerate(THRESHOLDS):
        trades_avg = np.mean([sr['threshold_summary'][tidx]['n_trades'] for sr in seed_results])
        prec_avg = np.mean([sr['threshold_summary'][tidx]['trade_precision'] for sr in seed_results])
        lprec_avg = np.mean([sr['threshold_summary'][tidx]['long_precision'] for sr in seed_results])
        sprec_avg = np.mean([sr['threshold_summary'][tidx]['short_precision'] for sr in seed_results])
        profit_avg = np.mean([sr['threshold_summary'][tidx]['profit'] for sr in seed_results])
        pf_avg = np.mean([sr['threshold_summary'][tidx]['profit_factor'] for sr in seed_results])
        sharpe_avg = np.mean([sr['threshold_summary'][tidx]['sharpe'] for sr in seed_results])
        logger.info(f"    {thresh:>7.2f} {trades_avg:>7.0f} {prec_avg:>6.3f} "
                    f"{lprec_avg:>7.3f} {sprec_avg:>7.3f} "
                    f"{profit_avg:>8.1f}% {pf_avg:>6.2f} {sharpe_avg:>7.3f}")

    # Cross-seed regime analysis
    logger.info(f"\n  Cross-seed regime analysis @0.75:")
    for ridx, (rname, _, _, rdesc) in enumerate(REGIME_PERIODS):
        r_profits = []
        r_trades = []
        r_prec = []
        for sr in seed_results:
            regime_list = sr.get('regime_analysis_075', [])
            if ridx < len(regime_list):
                r = regime_list[ridx]
                r_profits.append(r['profit_pct'])
                r_trades.append(r['n_trades'])
                r_prec.append(r['precision'])
        if r_profits:
            logger.info(f"    {rname} ({rdesc}):")
            logger.info(f"      profit: {[f'{p:.1f}' for p in r_profits]} (mean={np.mean(r_profits):.1f}%)")
            logger.info(f"      trades: {[str(t) for t in r_trades]} (mean={np.mean(r_trades):.0f})")
            logger.info(f"      precision: {[f'{p:.1f}' for p in r_prec]} (mean={np.mean(r_prec):.1f}%)")

    # Feature importance (averaged)
    fi_avg = sorted([(k, float(np.mean(v))) for k, v in all_fi.items()], key=lambda x: -x[1])
    logger.info(f"\n  Top 20 features (averaged across {len(SEEDS)} seeds):")
    for fname, fval in fi_avg[:20]:
        is_v10 = any(fname.startswith(p) for p in ['xw_', 'xtf_', 'corr_velocity_']) or \
                  fname in ('xtf_corr_agreement', 'hour_sin', 'hour_cos', 'dow_sin',
                            'dow_cos', 'is_ny_session', 'convergence_volume',
                            'crossing_atr', 'cascade_volume', 'reversal_conviction')
        tag = " [V10]" if is_v10 else ""
        logger.info(f"    {fname}: {fval:.2f}{tag}")

    # Feature group importance (averaged across seeds)
    logger.info(f"\n  Feature group importance (averaged):")
    avg_groups = {}
    for sr in seed_results:
        for gname, gval in sr['feature_groups'].items():
            if gname not in avg_groups:
                avg_groups[gname] = []
            avg_groups[gname].append(gval)
    for gname in sorted(avg_groups.keys()):
        avg_val = float(np.mean(avg_groups[gname]))
        logger.info(f"    {gname}: {avg_val:.2f}")

    v10_in_top20 = sum(1 for fname, _ in fi_avg[:20]
                       if any(fname.startswith(p) for p in ['xw_', 'xtf_', 'corr_velocity_'])
                       or fname in ('xtf_corr_agreement', 'hour_sin', 'hour_cos', 'dow_sin',
                                    'dow_cos', 'is_ny_session', 'convergence_volume',
                                    'crossing_atr', 'cascade_volume', 'reversal_conviction'))
    logger.info(f"\n  V10 features in top-20: {v10_in_top20}/20")

    fi_df = pd.DataFrame(fi_avg, columns=['feature', 'avg_importance'])
    fi_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    # Cross-seed equity plot
    try:
        save_cross_seed_equity(seed_results, RESULTS_DIR)
    except Exception as e:
        logger.warning(f"  Cross-seed equity plot failed: {e}")

    # ------------------------------------------------------------------
    # Step 7: Pass/Fail Criteria
    # ------------------------------------------------------------------
    logger.info(f"\n\n{'='*90}")
    logger.info("PASS/FAIL CRITERIA CHECK")
    logger.info(f"{'='*90}")

    all_passed, checks = check_pass_fail(seed_results, None)

    for cname, cinfo in checks.items():
        status = "PASS" if cinfo['passed'] else "FAIL"
        logger.info(f"  [{status}] {cname}: {cinfo['criterion']} = {cinfo['value']}")

    if all_passed:
        logger.info(f"\n  >>> OVERALL: PASS — Proceed to production model training <<<")
    else:
        logger.info(f"\n  >>> OVERALL: FAIL — Do NOT proceed to production <<<")
        failed = [k for k, v in checks.items() if not v['passed']]
        logger.info(f"  Failed checks: {failed}")

    # ------------------------------------------------------------------
    # Step 8: Save summary CSV
    # ------------------------------------------------------------------
    summary_rows = []
    for sr in seed_results:
        row = {
            'seed': sr['seed'],
            'roc_auc_macro': sr['roc_auc_macro'],
            'best_iteration': sr['best_iteration'],
            'n_features_used': sr['n_features_used'],
            'val_threshold': sr['val_selected_threshold'],
            'val_profit': sr['val_selected_profit'],
            'val_trades': sr['val_selected_trades'],
        }
        for ts in sr['threshold_summary']:
            t = ts['threshold']
            row[f'profit@{t:.2f}'] = ts['profit']
            row[f'trades@{t:.2f}'] = ts['n_trades']
            row[f'prec@{t:.2f}'] = ts['trade_precision']
            row[f'pf@{t:.2f}'] = ts['profit_factor']
            row[f'sharpe@{t:.2f}'] = ts['sharpe']
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "summary.csv", index=False)

    # ------------------------------------------------------------------
    # Step 9: Save master results JSON
    # ------------------------------------------------------------------
    logger.info(f"\n[STEP 9] Saving master results...")

    # Strip equity curves from JSON (too large)
    seed_results_for_json = []
    for sr in seed_results:
        sr_copy = {k: v for k, v in sr.items()
                   if k not in ('val_equity_curve', 'equity_at_070', 'equity_at_075')}
        seed_results_for_json.append(sr_copy)

    master_results = {
        'pipeline': 'V10_2yr_oos',
        'config': {k: str(v) if not isinstance(v, (int, float, list, bool)) else v
                   for k, v in MODEL_PARAMS.items()},
        'seeds': SEEDS,
        'split': {k: str(v.date()) for k, v in SPLIT.items()},
        'n_train': n_train,
        'n_val': n_val,
        'n_embargo': n_embargo,
        'n_test': n_test,
        'n_features': len(feature_cols),
        'sl_pct': SL_PCT,
        'tp_pct': TP_PCT,
        'trade_cooldown': TRADE_COOLDOWN,
        'thresholds': THRESHOLDS,
        'aggregate': {
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'per_seed_auc': aucs,
            'mean_honest_profit': mean_honest,
            'honest_profits': honest_profits,
            'val_thresholds': val_thresholds,
            'mean_profit_070': mean_070,
            'profits_070': profits_070,
            'mean_profit_075': mean_075,
            'profits_075': profits_075,
            'seed_cv_honest': cv_honest,
            'v10_features_in_top20': v10_in_top20,
        },
        'pass_fail': {
            'overall': 'PASS' if all_passed else 'FAIL',
            'checks': {k: {'criterion': v['criterion'], 'value': str(v['value']),
                           'passed': v['passed']} for k, v in checks.items()},
        },
        'regime_periods': [
            {'name': r[0], 'start': r[1], 'end': r[2], 'description': r[3]}
            for r in REGIME_PERIODS
        ],
        'seed_results': seed_results_for_json,
        'feature_importance_top20': fi_avg[:20],
        'feature_group_importance': {k: float(np.mean(v)) for k, v in avg_groups.items()},
    }

    results_path = RESULTS_DIR / "v10_2yr_oos_results.json"
    with open(results_path, 'w') as f:
        json.dump(json.loads(json.dumps(master_results, default=convert_for_json)),
                  f, indent=2)

    elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*90}")
    logger.info(f"V10 2-YEAR OOS FINAL EXAM COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"  VERDICT: {'PASS' if all_passed else 'FAIL'}")
    logger.info(f"  AUC: {mean_auc:.3f} +/- {std_auc:.3f}")
    logger.info(f"  Honest (val-selected): {mean_honest:.1f}% mean ({honest_profits})")
    logger.info(f"  @0.70 (fixed):    {mean_070:.1f}% mean ({profits_070})")
    logger.info(f"  @0.75 (production): {mean_075:.1f}% mean ({profits_075})")
    logger.info(f"  Seed CV: {cv_honest:.1f}%")
    logger.info(f"  V10 features in top-20: {v10_in_top20}/20")
    logger.info(f"  Output: {RESULTS_DIR}")
    if all_passed:
        logger.info(f"\n  >>> Next step: python model_training/train_v10_production.py <<<")
    logger.info(f"{'='*90}")

    return master_results


def _save_extra_trade_log(y_proba, y_test, test_times, threshold, seed, output_dir, suffix):
    """Save trade log at a specific threshold."""
    y_pred = np.argmax(y_proba, axis=1)
    for i in range(len(y_pred)):
        if y_pred[i] in TRADE_CLASSES:
            if y_proba[i, y_pred[i]] < threshold:
                y_pred[i] = 0

    rows = []
    cumulative = 0.0
    next_allowed = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 0 or i < next_allowed:
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
        next_allowed = i + TRADE_COOLDOWN
    if rows:
        pd.DataFrame(rows).to_csv(
            output_dir / f"s{seed}_trades_{suffix}.csv", index=False)


if __name__ == "__main__":
    run_2yr_oos()
