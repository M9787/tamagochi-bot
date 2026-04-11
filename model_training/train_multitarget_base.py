"""
Multi-Target Base Model Training — 8 SL/TP targets.

For each target:
  1. Generate SL/TP labels (cached)
  2. Walk-forward validation (4 folds, seed=42)
  3. Train final model (3 seeds: 42, 123, 777) on all pre-2024 data
  4. Generate OOS predictions on 2024-01-02 to 2026-03-03

Usage:
  python model_training/train_multitarget_base.py                    # All 8 targets
  python model_training/train_multitarget_base.py --targets T1 T3    # Specific targets
  python model_training/train_multitarget_base.py --skip-walkforward # Skip WF, train final only
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
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
)

from data.target_labeling import create_sl_tp_labels
from multitarget_config import (
    TARGET_CONFIGS, TARGET_NAMES,
    BASE_MODEL_PARAMS, MIN_ITERATIONS, SEEDS, THRESHOLDS, EMBARGO_DAYS,
    BASE_TRAIN_CUTOFF, WALKFORWARD_FOLDS, OOS_START, OOS_END,
    BASE_FINAL_VAL_START,
    ENCODED_DIR, ACTUAL_DATA_DIR, RESULTS_DIR,
    BASE_MODELS_DIR, LABEL_CACHE_DIR, OOS_DIR,
    CLASS_NAMES, TRADE_CLASSES,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Label Generation
# ============================================================================

def generate_labels_cached(target_name: str, config: dict,
                           price_data: pd.DataFrame) -> pd.DataFrame:
    """Generate SL/TP labels with disk cache."""
    LABEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = LABEL_CACHE_DIR / f"labels_{target_name}.parquet"

    if cache_path.exists():
        logger.info(f"  [{target_name}] Loading cached labels: {cache_path.name}")
        df = pd.read_parquet(cache_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    logger.info(f"  [{target_name}] Generating labels: SL={config['sl']}%, "
                f"TP={config['tp']}%, max_hold={config['max_hold']}...")
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
    dist = labels_df['label'].value_counts().to_dict()
    logger.info(f"  [{target_name}] {len(labels_df):,} labels in {elapsed:.1f}s | {dist}")

    labels_df.to_parquet(cache_path, index=False)
    logger.info(f"  [{target_name}] Cached: {cache_path.name}")
    return labels_df


def prepare_3class_labels(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Map {1->1(LONG), -1->2(SHORT), 0->0(NO_TRADE)}."""
    labels = labels_df.copy()
    labels['label_3class'] = labels['label'].map({1: 1, -1: 2, 0: 0})
    return labels


# ============================================================================
# Feature-Label Alignment
# ============================================================================

def align_features_labels(features: pd.DataFrame,
                          labels: pd.DataFrame) -> tuple:
    """Align features with labels by timestamp. Returns (X, y_df, times)."""
    feat_times = pd.to_datetime(features['time'])
    label_times = pd.to_datetime(labels['timestamp'])

    common = set(feat_times.values) & set(label_times.values)
    if len(common) == 0:
        raise ValueError("No timestamp overlap between features and labels.")

    common_sorted = sorted(common)
    feat_mask = feat_times.isin(common_sorted)
    label_mask = label_times.isin(common_sorted)

    X = features[feat_mask.values].sort_values('time').reset_index(drop=True)
    y_df = labels[label_mask.values].sort_values('timestamp').reset_index(drop=True)
    times = pd.to_datetime(X['time'])

    logger.info(f"  Aligned: {len(X):,} rows ({times.min()} to {times.max()})")
    return X, y_df, times


# ============================================================================
# Training
# ============================================================================

def train_model(X_train, y_train, X_val, y_val, seed=42):
    """Train a single CatBoost 3-class model."""
    train_pool = Pool(X_train, y_train)
    eval_pool = Pool(X_val, y_val)

    params = BASE_MODEL_PARAMS.copy()
    params['random_seed'] = seed
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=eval_pool, use_best_model=True)

    best_iter = model.get_best_iteration()
    if best_iter < MIN_ITERATIONS:
        logger.warning(f"  best_iter={best_iter} < {MIN_ITERATIONS}, "
                       f"retraining with min iterations...")
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
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    trade_mask = np.isin(y_pred, TRADE_CLASSES)
    n_trade_preds = int(trade_mask.sum())
    n_long_preds = int((y_pred == 1).sum())
    n_short_preds = int((y_pred == 2).sum())

    if n_trade_preds > 0:
        trade_correct = (y_pred[trade_mask] == y_test[trade_mask])
        trade_precision = float(trade_correct.mean())
        false_trade = (y_test[trade_mask] == 0)
        false_trade_rate = float(false_trade.mean())
    else:
        trade_precision = false_trade_rate = 0.0

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
    except Exception as e:
        logger.warning(f"  ROC AUC failed: {e}")
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
        'false_trade_rate': false_trade_rate,
        'n_trades_simulated': n_trades,
        'win_rate': win_rate,
        'total_profit_pct': total_profit,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
    }


# ============================================================================
# Walk-Forward Validation (single target)
# ============================================================================

def run_walkforward(target_name: str, config: dict,
                    X_all: pd.DataFrame, y_df_all: pd.DataFrame,
                    times_all: pd.Series, feature_cols: list) -> dict:
    """Run 4-fold walk-forward validation for one target. Seed=42 only."""
    logger.info(f"\n{'='*60}")
    logger.info(f"WALK-FORWARD: {target_name} (SL={config['sl']}%, TP={config['tp']}%)")
    logger.info(f"{'='*60}")

    sl_pct = config['sl']
    tp_pct = config['tp']
    cooldown = config['cooldown']
    y_all = y_df_all['label_3class'].values
    all_fold_results = []

    for fold in WALKFORWARD_FOLDS:
        fold_name = fold['name']
        logger.info(f"\n--- {fold_name}: train -> {fold['train_end'].date()}, "
                     f"test {fold['embargo_end'].date()} -> {fold['test_end'].date()} ---")

        # Split data
        train_mask = times_all < fold['train_end']
        val_mask = (times_all >= fold['train_end']) & (times_all < fold['val_end'])
        test_mask = (times_all >= fold['embargo_end']) & (times_all < fold['test_end'])

        X_train = X_all.loc[train_mask, feature_cols].values
        y_train = y_all[train_mask.values]
        X_val = X_all.loc[val_mask, feature_cols].values
        y_val = y_all[val_mask.values]
        X_test = X_all.loc[test_mask, feature_cols].values
        y_test = y_all[test_mask.values]

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            logger.warning(f"  {fold_name}: empty split, skipping")
            continue

        logger.info(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

        # Train
        model, best_iter = train_model(X_train, y_train, X_val, y_val, seed=42)
        y_pred_proba = model.predict_proba(X_test)

        # Evaluate at all thresholds
        fold_results = {
            'fold': fold_name,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'best_iter': best_iter,
            'thresholds': {},
        }

        for th in THRESHOLDS:
            result = evaluate_at_threshold(
                y_pred_proba, y_test, th,
                sl_pct=sl_pct, tp_pct=tp_pct, cooldown=cooldown,
            )
            fold_results['thresholds'][str(th)] = result

            if th == 0.70:
                logger.info(f"  @{th}: AUC={result['roc_auc_macro']:.3f} | "
                            f"Prec={result['trade_precision']:.1%} | "
                            f"Profit={result['total_profit_pct']:+.1f}% | "
                            f"Trades={result['n_trades_simulated']} | "
                            f"PF={result['profit_factor']:.2f}")

        all_fold_results.append(fold_results)

    # Summary across folds
    summary = _summarize_walkforward(target_name, config, all_fold_results)
    return summary


def _summarize_walkforward(target_name: str, config: dict,
                           fold_results: list) -> dict:
    """Summarize walk-forward results across folds."""
    summary = {
        'target': target_name,
        'sl': config['sl'],
        'tp': config['tp'],
        'max_hold': config['max_hold'],
        'cooldown': config['cooldown'],
        'folds': fold_results,
        'best_threshold': None,
        'best_total_profit': -999,
    }

    # Find best threshold (max total profit across all folds)
    for th in THRESHOLDS:
        th_str = str(th)
        total_profit = sum(
            f['thresholds'].get(th_str, {}).get('total_profit_pct', 0)
            for f in fold_results
        )
        avg_auc = np.mean([
            f['thresholds'].get(th_str, {}).get('roc_auc_macro', 0)
            for f in fold_results
        ])
        all_profitable = all(
            f['thresholds'].get(th_str, {}).get('total_profit_pct', 0) > 0
            for f in fold_results
        )

        if total_profit > summary['best_total_profit']:
            summary['best_total_profit'] = total_profit
            summary['best_threshold'] = th
            summary['best_avg_auc'] = avg_auc
            summary['best_all_folds_profitable'] = all_profitable

    # Per-fold summary at best threshold
    best_th = str(summary['best_threshold'])
    per_fold = []
    for f in fold_results:
        r = f['thresholds'].get(best_th, {})
        per_fold.append({
            'fold': f['fold'],
            'auc': r.get('roc_auc_macro', 0),
            'precision': r.get('trade_precision', 0),
            'profit': r.get('total_profit_pct', 0),
            'trades': r.get('n_trades_simulated', 0),
            'pf': r.get('profit_factor', 0),
        })
    summary['per_fold_at_best'] = per_fold

    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY: {target_name} | Best @{summary['best_threshold']}")
    logger.info(f"  Total Profit: {summary['best_total_profit']:+.1f}%")
    logger.info(f"  Avg AUC: {summary['best_avg_auc']:.3f}")
    logger.info(f"  All Folds Profitable: {summary['best_all_folds_profitable']}")
    for pf in per_fold:
        logger.info(f"  {pf['fold']}: AUC={pf['auc']:.3f} Prec={pf['precision']:.1%} "
                     f"Profit={pf['profit']:+.1f}% Trades={pf['trades']} PF={pf['pf']:.2f}")
    logger.info(f"{'='*60}")

    return summary


# ============================================================================
# Final Model Training + OOS Prediction
# ============================================================================

def train_final_and_predict_oos(target_name: str, config: dict,
                                X_all: pd.DataFrame, y_df_all: pd.DataFrame,
                                times_all: pd.Series,
                                feature_cols: list) -> dict:
    """Train final models (3 seeds) on pre-2024 data, predict OOS."""
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL MODEL: {target_name} (3 seeds)")
    logger.info(f"{'='*60}")

    BASE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OOS_DIR.mkdir(parents=True, exist_ok=True)

    y_all = y_df_all['label_3class'].values

    # Train/val split for final model
    train_mask = times_all < BASE_FINAL_VAL_START
    val_mask = (times_all >= BASE_FINAL_VAL_START) & (times_all < BASE_TRAIN_CUTOFF)
    oos_mask = (times_all >= OOS_START) & (times_all <= OOS_END)

    X_train = X_all.loc[train_mask, feature_cols].values
    y_train = y_all[train_mask.values]
    X_val = X_all.loc[val_mask, feature_cols].values
    y_val = y_all[val_mask.values]
    X_oos = X_all.loc[oos_mask, feature_cols].values
    oos_times = times_all[oos_mask].values

    logger.info(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | OOS: {len(X_oos):,}")

    if len(X_train) == 0 or len(X_oos) == 0:
        logger.error(f"  [{target_name}] Empty train or OOS set!")
        return {}

    # Train 3 seeds and average OOS predictions
    oos_probas = []
    feature_importances = []
    model_info = {}

    for seed in SEEDS:
        logger.info(f"\n  --- Seed {seed} ---")
        model, best_iter = train_model(X_train, y_train, X_val, y_val, seed=seed)

        # Save model
        model_path = BASE_MODELS_DIR / f"base_model_{target_name}_s{seed}.cbm"
        model.save_model(str(model_path))
        logger.info(f"  Saved: {model_path.name}")

        # OOS predictions
        proba = model.predict_proba(X_oos)
        oos_probas.append(proba)

        # Feature importance
        fi = model.get_feature_importance()
        feature_importances.append(fi)

        model_info[seed] = {
            'best_iter': best_iter,
            'model_path': str(model_path),
        }

    # Average probabilities across seeds
    avg_proba = np.mean(oos_probas, axis=0)

    # Save OOS probabilities
    oos_df = pd.DataFrame({
        'time': oos_times,
        f'{target_name.lower()}_prob_nt': avg_proba[:, 0],
        f'{target_name.lower()}_prob_long': avg_proba[:, 1],
        f'{target_name.lower()}_prob_short': avg_proba[:, 2],
    })
    oos_path = OOS_DIR / f"oos_proba_{target_name}.parquet"
    oos_df.to_parquet(oos_path, index=False)
    logger.info(f"  OOS predictions: {len(oos_df):,} rows -> {oos_path.name}")

    # Average feature importance across seeds
    avg_fi = np.mean(feature_importances, axis=0)
    fi_sorted = sorted(zip(feature_cols, avg_fi), key=lambda x: -x[1])

    # Save feature importance
    fi_df = pd.DataFrame(fi_sorted, columns=['feature', 'importance'])
    fi_path = RESULTS_DIR / f"feature_importance_{target_name}.csv"
    fi_df.to_csv(fi_path, index=False)

    top_10 = fi_sorted[:10]
    logger.info(f"  Top 10 features:")
    for fname, fval in top_10:
        logger.info(f"    {fname}: {fval:.2f}")

    return {
        'target': target_name,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_oos': len(X_oos),
        'models': model_info,
        'top_10_features': [(f, round(v, 3)) for f, v in top_10],
        'oos_path': str(oos_path),
    }


# ============================================================================
# Combine OOS Predictions
# ============================================================================

def combine_oos_predictions(targets: list) -> pd.DataFrame:
    """Combine OOS predictions from all targets into single DataFrame."""
    dfs = []
    for target_name in targets:
        path = OOS_DIR / f"oos_proba_{target_name}.parquet"
        if not path.exists():
            logger.warning(f"  Missing OOS predictions: {path}")
            continue
        df = pd.read_parquet(path)
        df['time'] = pd.to_datetime(df['time'])
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No OOS prediction files found")

    # Merge all on time
    combined = dfs[0]
    for df in dfs[1:]:
        combined = combined.merge(df, on='time', how='inner')

    combined = combined.sort_values('time').reset_index(drop=True)

    # Save combined
    combined_path = RESULTS_DIR / "oos_probabilities_all.parquet"
    combined.to_parquet(combined_path, index=False)
    logger.info(f"  Combined OOS: {len(combined):,} rows x {len(combined.columns)} cols "
                f"-> oos_probabilities_all.parquet")
    return combined


# ============================================================================
# Main
# ============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    parser = argparse.ArgumentParser(description="Multi-Target Base Model Training")
    parser.add_argument("--targets", nargs='+', default=None,
                        help=f"Targets to train (default: all). Options: {TARGET_NAMES}")
    parser.add_argument("--skip-walkforward", action="store_true",
                        help="Skip walk-forward, train final models only")
    args = parser.parse_args()

    targets = args.targets or TARGET_NAMES
    for t in targets:
        if t not in TARGET_CONFIGS:
            parser.error(f"Unknown target: {t}. Options: {TARGET_NAMES}")

    # Create output dirs
    for d in [RESULTS_DIR, BASE_MODELS_DIR, LABEL_CACHE_DIR, OOS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    t0_total = time.time()
    logger.info("=" * 70)
    logger.info(f"MULTI-TARGET BASE MODEL TRAINING: {len(targets)} targets")
    logger.info(f"Targets: {targets}")
    logger.info(f"Base train cutoff: {BASE_TRAIN_CUTOFF}")
    logger.info(f"OOS period: {OOS_START} to {OOS_END}")
    logger.info("=" * 70)

    # Load features (once)
    logger.info("\n[1/4] Loading features...")
    feat_path = ENCODED_DIR / "feature_matrix_v10.parquet"
    features = pd.read_parquet(feat_path)
    features['time'] = pd.to_datetime(features['time'])
    feature_cols = [c for c in features.columns if c != 'time']
    logger.info(f"  Features: {len(features):,} rows x {len(feature_cols)} features")

    # Load price data (once, for label generation)
    logger.info("\n[2/4] Loading price data...")
    price_path = ACTUAL_DATA_DIR / "ml_data_5M.csv"
    price_data = pd.read_csv(price_path)
    price_data['Open Time'] = pd.to_datetime(price_data['Open Time'])
    for c in ('Open', 'High', 'Low', 'Close', 'Volume'):
        price_data[c] = pd.to_numeric(price_data[c], errors='coerce')
    logger.info(f"  Price data: {len(price_data):,} rows")

    # Process each target
    wf_results = {}
    final_results = {}

    for target_name in targets:
        config = TARGET_CONFIGS[target_name]
        logger.info(f"\n{'#'*70}")
        logger.info(f"# TARGET: {target_name} | SL={config['sl']}% TP={config['tp']}% "
                     f"max_hold={config['max_hold']} ({config['category']})")
        logger.info(f"{'#'*70}")

        # Generate labels
        labels_df = generate_labels_cached(target_name, config, price_data)
        labels_df = prepare_3class_labels(labels_df)

        # Sanity check label distribution
        dist = labels_df['label'].value_counts(normalize=True)
        nt_pct = dist.get(0, 0) * 100
        logger.info(f"  Label distribution: NT={nt_pct:.1f}% | "
                     f"LONG={dist.get(1, 0)*100:.1f}% | SHORT={dist.get(-1, 0)*100:.1f}%")

        # Align features and labels
        X_all, y_df_all, times_all = align_features_labels(features, labels_df)

        # Walk-forward validation
        if not args.skip_walkforward:
            wf = run_walkforward(
                target_name, config, X_all, y_df_all, times_all, feature_cols,
            )
            wf_results[target_name] = wf

        # Train final model + OOS predictions
        final = train_final_and_predict_oos(
            target_name, config, X_all, y_df_all, times_all, feature_cols,
        )
        final_results[target_name] = final

    # Combine OOS predictions
    logger.info("\n[3/4] Combining OOS predictions...")
    combine_oos_predictions(targets)

    # Save results
    logger.info("\n[4/4] Saving results...")
    results = {
        'targets': targets,
        'base_train_cutoff': str(BASE_TRAIN_CUTOFF),
        'oos_start': str(OOS_START),
        'oos_end': str(OOS_END),
        'walkforward': {k: _make_json_safe(v) for k, v in wf_results.items()},
        'final_models': {k: _make_json_safe(v) for k, v in final_results.items()},
    }
    results_path = RESULTS_DIR / "multitarget_base_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"  Results: {results_path}")

    # Print summary table
    elapsed = time.time() - t0_total
    logger.info(f"\n{'='*70}")
    logger.info(f"MULTI-TARGET BASE TRAINING COMPLETE ({elapsed/60:.1f} min)")
    logger.info(f"{'='*70}")

    if wf_results:
        logger.info(f"\n{'Target':>6} | {'SL/TP':>6} | {'Best@':>6} | {'Profit':>8} | "
                     f"{'AUC':>5} | {'All+':>4} | Category")
        logger.info("-" * 65)
        for t in targets:
            if t not in wf_results:
                continue
            wf = wf_results[t]
            cfg = TARGET_CONFIGS[t]
            logger.info(f"{t:>6} | {cfg['sl']}/{cfg['tp']:>4} | "
                         f"@{wf.get('best_threshold', 0):.2f} | "
                         f"{wf.get('best_total_profit', 0):>+7.1f}% | "
                         f"{wf.get('best_avg_auc', 0):>5.3f} | "
                         f"{'YES' if wf.get('best_all_folds_profitable') else 'NO':>4} | "
                         f"{cfg['category']}")


def _make_json_safe(obj):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    return obj


if __name__ == "__main__":
    main()
