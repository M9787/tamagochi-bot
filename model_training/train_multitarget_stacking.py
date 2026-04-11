"""
Multi-Target Stacking Model Training — 8 stacking models from 8 base model OOS predictions.

For each target:
  1. Load OOS probabilities from all 8 base models (24 features)
  2. Build interaction features (~150 features)
  3. Include top-50 raw V10 features (~50-60 features)
  4. Train stacking CatBoost (depth=5, Optuna-tuned meta params)
  5. Evaluate with target-specific SL/TP/cooldown

Prerequisite: run train_multitarget_base.py first.

Usage:
  python model_training/train_multitarget_stacking.py                    # All 8
  python model_training/train_multitarget_stacking.py --targets T1 T3    # Specific
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
    roc_auc_score,
)

from multitarget_config import (
    TARGET_CONFIGS, TARGET_NAMES, SCALE_GROUPS,
    META_MODEL_PARAMS, MIN_ITERATIONS, THRESHOLDS,
    OOS_START, OOS_END, STACK_TRAIN_END, STACK_VAL_END, STACK_TEST_END,
    ENCODED_DIR, RESULTS_DIR, LABEL_CACHE_DIR, OOS_DIR, STACKING_DIR,
    CLASS_NAMES, TRADE_CLASSES, TOP_RAW_FEATURES,
)

# Single source of truth for stacking meta-features (live + training share this).
from core.multitarget_feature_builder import build_interaction_features

logger = logging.getLogger(__name__)


# ============================================================================
# Load Data
# ============================================================================

def load_oos_probabilities() -> pd.DataFrame:
    """Load combined OOS probabilities from all base models."""
    path = RESULTS_DIR / "oos_probabilities_all.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Combined OOS predictions not found: {path}\n"
            f"Run train_multitarget_base.py first."
        )
    df = pd.read_parquet(path)
    df['time'] = pd.to_datetime(df['time'])
    logger.info(f"  OOS probabilities: {len(df):,} rows x {len(df.columns)} cols")
    return df


def load_target_labels(target_name: str) -> pd.DataFrame:
    """Load cached labels for a target."""
    path = LABEL_CACHE_DIR / f"labels_{target_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Labels not found: {path}")
    df = pd.read_parquet(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['label_3class'] = df['label'].map({1: 1, -1: 2, 0: 0})
    return df


def load_top_raw_features(targets: list) -> tuple:
    """Load top-50 raw V10 features (union across all base models)."""
    all_features = set()
    for target_name in targets:
        fi_path = RESULTS_DIR / f"feature_importance_{target_name}.csv"
        if not fi_path.exists():
            logger.warning(f"  Missing FI for {target_name}: {fi_path}")
            continue
        fi_df = pd.read_csv(fi_path)
        top = fi_df.nlargest(TOP_RAW_FEATURES, 'importance')['feature'].tolist()
        all_features.update(top)

    logger.info(f"  Union of top-{TOP_RAW_FEATURES} raw features: {len(all_features)} unique")

    if not all_features:
        return None, []

    # Persist the union list so the live MultiTargetPredictor can project
    # raw V10 rows down to the same column set without rebuilding it from
    # the per-target feature_importance CSVs.
    STACKING_DIR.mkdir(parents=True, exist_ok=True)
    union_path = STACKING_DIR / "top_raw_features_union.json"
    with union_path.open('w') as fh:
        json.dump(sorted(all_features), fh, indent=2)
    logger.info(f"  Wrote top-raw union to {union_path}")

    # Load feature matrix (only selected columns + time)
    feat_path = ENCODED_DIR / "feature_matrix_v10.parquet"
    cols_to_load = ['time'] + sorted(all_features)
    features = pd.read_parquet(feat_path, columns=cols_to_load)
    features['time'] = pd.to_datetime(features['time'])
    raw_feature_cols = sorted(all_features)

    logger.info(f"  Raw features loaded: {len(features):,} rows x {len(raw_feature_cols)} cols")
    return features, raw_feature_cols


# ============================================================================
# Interaction Features
# ============================================================================
# build_interaction_features lives in core/multitarget_feature_builder.py so
# the live MultiTargetPredictor and this training script share one
# implementation. The import at the top of this file binds it into module
# scope; the legacy call site below is unchanged.


# ============================================================================
# Evaluation (reused from base, with equity curve plotting)
# ============================================================================

def evaluate_stacking(y_pred_proba, y_test, threshold: float,
                      sl_pct: float, tp_pct: float, cooldown: int,
                      times=None) -> dict:
    """Evaluate stacking model predictions."""
    n_total = len(y_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    if threshold > 0:
        for i in range(n_total):
            if y_pred[i] in TRADE_CLASSES:
                if y_pred_proba[i, y_pred[i]] < threshold:
                    y_pred[i] = 0

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    trade_mask = np.isin(y_pred, TRADE_CLASSES)
    n_trade_preds = int(trade_mask.sum())
    n_long_preds = int((y_pred == 1).sum())
    n_short_preds = int((y_pred == 2).sum())

    if n_trade_preds > 0:
        trade_correct = (y_pred[trade_mask] == y_test[trade_mask])
        trade_precision = float(trade_correct.mean())
    else:
        trade_precision = 0.0

    # Trading simulation with cooldown
    equity = [0.0]
    trades = []
    next_allowed_idx = 0
    for i in range(n_total):
        if y_pred[i] == 0 or i < next_allowed_idx:
            continue
        gain = tp_pct if y_pred[i] == y_test[i] else -sl_pct
        trades.append({
            'index': i, 'predicted': int(y_pred[i]), 'actual': int(y_test[i]),
            'gain_pct': gain, 'confidence': float(y_pred_proba[i, y_pred[i]]),
            'time': str(times[i]) if times is not None else None,
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
        roc_auc_macro = float(roc_auc_score(
            y_onehot, y_pred_proba, multi_class='ovr', average='macro'))
    except Exception:
        roc_auc_macro = 0.0

    return {
        'threshold': threshold,
        'accuracy': acc,
        'confusion_matrix': cm.tolist(),
        'roc_auc_macro': roc_auc_macro,
        'n_total': n_total,
        'n_trade_predictions': n_trade_preds,
        'n_long_predictions': n_long_preds,
        'n_short_predictions': n_short_preds,
        'trade_precision': trade_precision,
        'n_trades_simulated': n_trades,
        'win_rate': win_rate,
        'total_profit_pct': total_profit,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
        'equity_curve': equity,
        'trades': trades,
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_equity_curve(equity: list, target_name: str, config: dict,
                      threshold: float, output_dir: Path):
    """Plot and save equity curve for a stacking model."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity, linewidth=1.2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'{target_name} Stacking | SL={config["sl"]}% TP={config["tp"]}% '
                 f'@{threshold} | Final: {equity[-1]:+.1f}%')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative P&L (%)')
    ax.grid(True, alpha=0.3)
    path = output_dir / f"equity_curve_{target_name}.png"
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Equity curve: {path.name}")


def plot_signal_correlation(predictions: dict, output_dir: Path):
    """Plot 8x8 signal correlation matrix."""
    if len(predictions) < 2:
        return

    # Build signal matrix (1=LONG, -1=SHORT, 0=NT)
    signal_df = pd.DataFrame()
    for target_name, pred_class in predictions.items():
        signal_df[target_name] = pred_class

    corr = signal_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title('Stacking Signal Correlation (Test Period)')
    path = output_dir / "signal_correlation_matrix.png"
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Correlation matrix: {path.name}")


# ============================================================================
# Main Training Loop
# ============================================================================

def train_stacking_model(target_name: str, config: dict,
                         meta_features: pd.DataFrame,
                         meta_feature_cols: list,
                         labels: pd.DataFrame,
                         meta_times: pd.Series) -> dict:
    """Train a single stacking model for one target."""
    logger.info(f"\n--- Stacking: {target_name} (SL={config['sl']}% TP={config['tp']}%) ---")

    sl_pct = config['sl']
    tp_pct = config['tp']
    cooldown = config['cooldown']

    # Align meta features with labels
    label_times = pd.to_datetime(labels['timestamp'])
    common = set(meta_times.values) & set(label_times.values)
    if len(common) == 0:
        logger.error(f"  [{target_name}] No overlapping timestamps!")
        return {}

    common_sorted = sorted(common)
    meta_mask = meta_times.isin(common_sorted)
    label_mask = label_times.isin(common_sorted)

    X = meta_features[meta_mask.values].reset_index(drop=True)
    y_df = labels[label_mask.values].sort_values('timestamp').reset_index(drop=True)
    times = meta_times[meta_mask.values].reset_index(drop=True)
    y = y_df['label_3class'].values

    logger.info(f"  Aligned: {len(X):,} rows")

    # Temporal split
    train_mask = times < STACK_TRAIN_END
    val_mask = (times >= STACK_TRAIN_END) & (times < STACK_VAL_END)
    test_mask = (times >= STACK_VAL_END) & (times <= STACK_TEST_END)

    X_train = X.loc[train_mask, meta_feature_cols].values
    y_train = y[train_mask.values]
    X_val = X.loc[val_mask, meta_feature_cols].values
    y_val = y[val_mask.values]
    X_test = X.loc[test_mask, meta_feature_cols].values
    y_test = y[test_mask.values]
    test_times = times[test_mask].values

    logger.info(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        logger.error(f"  [{target_name}] Empty split!")
        return {}

    # Train
    train_pool = Pool(X_train, y_train)
    eval_pool = Pool(X_val, y_val)

    params = META_MODEL_PARAMS.copy()
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

    # Save model
    model_path = STACKING_DIR / f"stacking_model_{target_name}.cbm"
    model.save_model(str(model_path))
    logger.info(f"  Saved: {model_path.name}")

    # Predict on test
    y_pred_proba = model.predict_proba(X_test)

    # Evaluate at all thresholds
    results = {
        'target': target_name,
        'sl': sl_pct,
        'tp': tp_pct,
        'max_hold': config['max_hold'],
        'cooldown': cooldown,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'best_iter': best_iter,
        'n_features': len(meta_feature_cols),
        'thresholds': {},
        'best_threshold': None,
        'best_profit': -999,
    }

    for th in THRESHOLDS:
        r = evaluate_stacking(
            y_pred_proba, y_test, th,
            sl_pct=sl_pct, tp_pct=tp_pct, cooldown=cooldown,
            times=test_times,
        )
        results['thresholds'][str(th)] = {
            k: v for k, v in r.items()
            if k not in ('equity_curve', 'trades')
        }

        if r['total_profit_pct'] > results['best_profit']:
            results['best_profit'] = r['total_profit_pct']
            results['best_threshold'] = th
            results['best_equity'] = r['equity_curve']
            results['best_result'] = r

        logger.info(f"  @{th}: AUC={r['roc_auc_macro']:.3f} | "
                     f"Prec={r['trade_precision']:.1%} | "
                     f"Profit={r['total_profit_pct']:+.1f}% | "
                     f"Trades={r['n_trades_simulated']} | "
                     f"PF={r['profit_factor']:.2f}")

    # Plot equity at best threshold
    if results.get('best_equity'):
        plot_equity_curve(
            results['best_equity'], target_name, config,
            results['best_threshold'], STACKING_DIR,
        )

    # Feature importance
    fi = model.get_feature_importance()
    fi_sorted = sorted(zip(meta_feature_cols, fi), key=lambda x: -x[1])
    fi_df = pd.DataFrame(fi_sorted, columns=['feature', 'importance'])
    fi_path = STACKING_DIR / f"stacking_fi_{target_name}.csv"
    fi_df.to_csv(fi_path, index=False)

    top_5 = fi_sorted[:5]
    logger.info(f"  Top 5 stacking features:")
    for fname, fval in top_5:
        logger.info(f"    {fname}: {fval:.2f}")

    results['top_10_features'] = [(f, round(v, 3)) for f, v in fi_sorted[:10]]

    # Pass criteria
    best_r = results.get('best_result', {})
    results['pass_criteria'] = {
        'auc_gte_070': best_r.get('roc_auc_macro', 0) >= 0.70,
        'profit_positive': best_r.get('total_profit_pct', 0) > 0,
        'precision_gte_be': best_r.get('trade_precision', 0) > (sl_pct / (sl_pct + tp_pct)),
        'has_both_directions': best_r.get('n_long_predictions', 0) > 0 and best_r.get('n_short_predictions', 0) > 0,
        'trades_gte_5_per_month': best_r.get('n_trades_simulated', 0) >= 40,  # ~5/month * 8 months
    }
    all_pass = all(results['pass_criteria'].values())
    results['verdict'] = 'PASS' if all_pass else 'FAIL'

    logger.info(f"\n  VERDICT: {results['verdict']}")
    for check, passed in results['pass_criteria'].items():
        logger.info(f"    {'PASS' if passed else 'FAIL'}: {check}")

    # Clean up non-serializable data
    results.pop('best_equity', None)
    results.pop('best_result', None)

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    parser = argparse.ArgumentParser(description="Multi-Target Stacking Model Training")
    parser.add_argument("--targets", nargs='+', default=None,
                        help=f"Targets to train (default: all). Options: {TARGET_NAMES}")
    parser.add_argument("--no-raw-features", action="store_true",
                        help="Skip including raw V10 features in stacking input")
    args = parser.parse_args()

    targets = args.targets or TARGET_NAMES
    for t in targets:
        if t not in TARGET_CONFIGS:
            parser.error(f"Unknown target: {t}. Options: {TARGET_NAMES}")

    STACKING_DIR.mkdir(parents=True, exist_ok=True)

    t0_total = time.time()
    logger.info("=" * 70)
    logger.info(f"MULTI-TARGET STACKING TRAINING: {len(targets)} targets")
    logger.info(f"Stacking train: {OOS_START} to {STACK_TRAIN_END}")
    logger.info(f"Stacking val:   {STACK_TRAIN_END} to {STACK_VAL_END}")
    logger.info(f"Stacking test:  {STACK_VAL_END} to {STACK_TEST_END}")
    logger.info("=" * 70)

    # Load OOS probabilities
    logger.info("\n[1/5] Loading OOS probabilities...")
    oos_df = load_oos_probabilities()

    # Build interaction features
    logger.info("\n[2/5] Building interaction features...")
    meta_df = build_interaction_features(oos_df.copy(), TARGET_NAMES)

    # Load top raw features
    raw_features_df = None
    raw_feature_cols = []
    if not args.no_raw_features:
        logger.info("\n[3/5] Loading top raw V10 features...")
        raw_features_df, raw_feature_cols = load_top_raw_features(TARGET_NAMES)

        if raw_features_df is not None:
            # Merge raw features into meta_df by time
            meta_df = meta_df.merge(raw_features_df, on='time', how='left')
            logger.info(f"  Merged raw features: {len(raw_feature_cols)} columns")
    else:
        logger.info("\n[3/5] Skipping raw features (--no-raw-features)")

    # Determine feature columns
    meta_feature_cols = [c for c in meta_df.columns if c != 'time']
    meta_times = pd.to_datetime(meta_df['time'])
    logger.info(f"  Total stacking features: {len(meta_feature_cols)}")

    # Train stacking models
    logger.info("\n[4/5] Training stacking models...")
    all_results = {}
    signal_predictions = {}

    for target_name in targets:
        config = TARGET_CONFIGS[target_name]

        # Load target labels
        labels = load_target_labels(target_name)

        # Train stacking model
        result = train_stacking_model(
            target_name, config, meta_df, meta_feature_cols,
            labels, meta_times,
        )
        all_results[target_name] = result

        # Collect test predictions for correlation analysis
        if result and result.get('best_threshold'):
            model_path = STACKING_DIR / f"stacking_model_{target_name}.cbm"
            if model_path.exists():
                model = CatBoostClassifier()
                model.load_model(str(model_path))

                test_mask = (meta_times >= STACK_VAL_END) & (meta_times <= STACK_TEST_END)
                X_test = meta_df.loc[test_mask, meta_feature_cols].values
                proba = model.predict_proba(X_test)
                preds = np.argmax(proba, axis=1)

                # Apply threshold
                th = result['best_threshold']
                for i in range(len(preds)):
                    if preds[i] in TRADE_CLASSES and proba[i, preds[i]] < th:
                        preds[i] = 0

                # Map to signal: 0=NT, 1=LONG, 2=SHORT -> 0, 1, -1
                signals = np.where(preds == 1, 1, np.where(preds == 2, -1, 0))
                signal_predictions[target_name] = signals

    # Signal correlation matrix
    logger.info("\n[5/5] Cross-target analysis...")
    plot_signal_correlation(signal_predictions, STACKING_DIR)

    # Save results
    def _json_safe(obj):
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj

    summary = {
        'targets': targets,
        'n_stacking_features': len(meta_feature_cols),
        'n_raw_features_included': len(raw_feature_cols),
        'stack_train_end': str(STACK_TRAIN_END),
        'stack_val_end': str(STACK_VAL_END),
        'stack_test_end': str(STACK_TEST_END),
        'results': _json_safe(all_results),
    }
    summary_path = STACKING_DIR / "multitarget_stacking_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary table
    elapsed = time.time() - t0_total
    logger.info(f"\n{'='*70}")
    logger.info(f"MULTI-TARGET STACKING COMPLETE ({elapsed/60:.1f} min)")
    logger.info(f"{'='*70}")

    logger.info(f"\n{'Target':>6} | {'SL/TP':>6} | {'Best@':>6} | {'Profit':>8} | "
                f"{'AUC':>5} | {'Prec':>5} | {'Trades':>6} | {'PF':>5} | Verdict")
    logger.info("-" * 75)
    for t in targets:
        r = all_results.get(t, {})
        if not r:
            continue
        cfg = TARGET_CONFIGS[t]
        best_th = r.get('best_threshold', 0)
        best_th_str = str(best_th)
        th_r = r.get('thresholds', {}).get(best_th_str, {})
        logger.info(
            f"{t:>6} | {cfg['sl']}/{cfg['tp']:>4} | "
            f"@{best_th:.2f} | "
            f"{th_r.get('total_profit_pct', 0):>+7.1f}% | "
            f"{th_r.get('roc_auc_macro', 0):>5.3f} | "
            f"{th_r.get('trade_precision', 0):>4.1%} | "
            f"{th_r.get('n_trades_simulated', 0):>6} | "
            f"{th_r.get('profit_factor', 0):>5.2f} | "
            f"{r.get('verdict', '?')}"
        )

    logger.info(f"\nResults: {summary_path}")
    logger.info(f"Models:  {STACKING_DIR}/stacking_model_T*.cbm")


if __name__ == "__main__":
    main()
