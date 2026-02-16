"""
XGBoost Training Pipeline: Train 3 model versions (V1/V2/V3) with fixed SL/TP labels
and confidence threshold evaluation.

Usage: python model_training/train.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import logging
import json
import pickle
from pathlib import Path
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

from model_training.etl import run_etl_features, build_labels, align_features_labels

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

# Fixed SL/TP config
SL_PCT = 2.0          # 2% Stop Loss
TP_PCT = 6.0          # 6% Take Profit
MAX_HOLD = 288        # 1 day of 5M candles

# Confidence thresholds to evaluate
CONFIDENCE_THRESHOLDS = [0.0, 0.6, 0.7]


def split_data_temporal(features: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """
    Split into train/test by time: last 13 months total.
    Train: first 12 months (~105,120 rows of 5M)
    Test: last 1 month (~8,640 rows of 5M)
    """
    y = labels['label'].values

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

    n = len(features)
    # 12/13 train, 1/13 test
    train_end = int(n * (12.0 / 13.0))

    logger.info(f"  Temporal split: {train_end} train / {n - train_end} test "
                f"(total {n})")

    return {
        'X_train': features.iloc[:train_end],
        'X_test': features.iloc[train_end:],
        'y_train': y[:train_end],
        'y_test': y[train_end:],
        'train_end': train_end,
        'labels_test': labels.iloc[train_end:].reset_index(drop=True),
    }


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute inverse frequency sample weights for class balancing."""
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    total = len(y)

    weight_map = {}
    for cls, cnt in zip(classes, counts):
        weight_map[cls] = total / (n_classes * cnt)

    weights = np.array([weight_map[val] for val in y])
    logger.info(f"  Class weights: {dict(zip(classes, [f'{weight_map[c]:.3f}' for c in classes]))}")
    return weights


def train_xgboost(X_train, y_train, version_name: str) -> XGBClassifier:
    """Train XGBoost classifier with inverse frequency class weights."""
    y_mapped = y_train + 1  # -1->0, 0->1, 1->2

    unique_classes = np.unique(y_mapped)
    n_classes = len(unique_classes)

    sample_weights = compute_sample_weights(y_mapped)

    if n_classes < 2:
        logger.warning(f"  {version_name}: Only {n_classes} class(es)")

    if n_classes <= 2:
        model = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            objective='binary:logistic' if n_classes == 2 else 'multi:softprob',
            eval_metric='logloss' if n_classes == 2 else 'mlogloss',
            random_state=42, use_label_encoder=False, verbosity=0,
        )
    else:
        model = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            objective='multi:softprob', num_class=3, eval_metric='mlogloss',
            random_state=42, use_label_encoder=False, verbosity=0,
        )

    model.fit(X_train, y_mapped, sample_weight=sample_weights)
    logger.info(f"  {version_name}: {len(X_train)} samples, "
                f"{X_train.shape[1]} features, {n_classes} classes")

    return model


def evaluate_with_confidence(model, X_test, y_test, labels_test,
                             version_name: str, threshold: float) -> dict:
    """
    Evaluate model at a given confidence threshold.

    threshold=0.0 means no filter (baseline).
    threshold=0.6/0.7 means only trade when max(proba) >= threshold.
    Uses ATR-aware trading: SL is dynamic per trade (from labels_test).
    """
    y_pred_mapped = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    y_pred = y_pred_mapped - 1

    # Apply confidence filter
    max_proba = y_pred_proba.max(axis=1)
    if threshold > 0:
        conf_mask = max_proba >= threshold
    else:
        conf_mask = np.ones(len(y_pred), dtype=bool)

    n_total = len(y_pred)
    n_confident = conf_mask.sum()

    # Metrics on confident predictions only
    y_pred_conf = y_pred[conf_mask]
    y_test_conf = y_test[conf_mask]
    y_test_mapped_conf = y_test_conf + 1

    if len(y_pred_conf) == 0:
        logger.info(f"  {version_name} @{threshold}: No predictions above threshold")
        return _empty_metrics(version_name, threshold, n_total)

    acc = accuracy_score(y_test_conf, y_pred_conf)
    cm = confusion_matrix(y_test_conf, y_pred_conf, labels=[-1, 0, 1])
    report = classification_report(y_test_conf, y_pred_conf, labels=[-1, 0, 1],
                                   target_names=['SHORT', 'NO_TRADE', 'LONG'],
                                   output_dict=True, zero_division=0)

    try:
        y_pred_proba_conf = y_pred_proba[conf_mask]
        roc_auc = roc_auc_score(y_test_mapped_conf, y_pred_proba_conf,
                                multi_class='ovr', average='weighted')
    except Exception:
        roc_auc = 0.0

    # Trading simulation
    trades = []
    equity = [0.0]
    conf_indices = np.where(conf_mask)[0]

    for idx in conf_indices:
        pred = y_pred[idx]
        actual_label = y_test[idx]

        if pred == 0:
            continue

        if pred == actual_label:
            gain = TP_PCT
        elif actual_label == -pred:
            gain = -SL_PCT
        else:
            gain = 0.0

        trades.append({
            'index': int(idx),
            'prediction': int(pred),
            'actual': int(actual_label),
            'gain_pct': gain,
            'confidence': float(max_proba[idx]),
        })
        equity.append(equity[-1] + gain)

    trades_df = pd.DataFrame(trades)
    n_trades = len(trades_df)

    if n_trades > 0:
        wins = trades_df[trades_df['gain_pct'] > 0]
        win_rate = len(wins) / n_trades * 100
        total_profit = trades_df['gain_pct'].sum()
        avg_gain = trades_df['gain_pct'].mean()
        returns = trades_df['gain_pct'].values
        sharpe = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(288) if len(returns) > 1 else 0
        equity_arr = np.array(equity)
        peak = np.maximum.accumulate(equity_arr)
        max_drawdown = (equity_arr - peak).min()
    else:
        win_rate = total_profit = avg_gain = sharpe = max_drawdown = 0.0

    metrics = {
        'version': version_name,
        'threshold': threshold,
        'accuracy': acc,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'n_total_test': n_total,
        'n_confident': int(n_confident),
        'n_trades': n_trades,
        'win_rate': win_rate,
        'total_profit_pct': total_profit,
        'avg_gain_pct': avg_gain,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_drawdown,
        'equity_curve': equity,
        'predictions': y_pred_conf.tolist(),
        'actuals': y_test_conf.tolist(),
    }

    if n_trades > 0:
        metrics['trades'] = trades_df.to_dict('records')

    logger.info(f"  {version_name} @{threshold:.1f}: Acc={acc:.3f} | AUC={roc_auc:.3f} | "
                f"Confident={n_confident}/{n_total} | "
                f"Trades={n_trades} | WR={win_rate:.1f}% | "
                f"Profit={total_profit:.1f}% | Sharpe={sharpe:.2f} | "
                f"MaxDD={max_drawdown:.1f}%")

    return metrics


def _empty_metrics(version_name, threshold, n_total):
    return {
        'version': version_name, 'threshold': threshold,
        'accuracy': 0, 'roc_auc': 0,
        'confusion_matrix': [[0]*3]*3, 'classification_report': {},
        'n_total_test': n_total, 'n_confident': 0,
        'n_trades': 0, 'win_rate': 0, 'total_profit_pct': 0,
        'avg_gain_pct': 0, 'sharpe_ratio': 0, 'max_drawdown_pct': 0,
        'equity_curve': [0], 'predictions': [], 'actuals': [],
    }


def get_feature_importance(model, feature_names, top_n: int = 30) -> list:
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    return [(feature_names[i], float(importance[i])) for i in indices]


def run_training():
    """
    Full pipeline: ETL features (13mo trim) -> Fixed SL/TP labels -> Train V1/V2/V3 ->
    Evaluate at confidence thresholds 0.0/0.6/0.7.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("=" * 70)
    logger.info("ML TRAINING PIPELINE — FIXED SL/TP + CONFIDENCE THRESHOLDS")
    logger.info(f"SL={SL_PCT}% | TP={TP_PCT}% | Max Hold={MAX_HOLD}")
    logger.info(f"Confidence thresholds: {CONFIDENCE_THRESHOLDS}")
    logger.info("=" * 70)

    # Step 1: Build features ONCE (trimmed to 13 months)
    logger.info("\n[STEP 1] Building features (13-month trim)...")
    feat_data = run_etl_features(trim_months=13)

    version_keys = ['v1', 'v2', 'v3']
    version_names = {
        'v1': 'V1_Flat', 'v2': 'V2_Score',
        'v3': 'V3_Binary'
    }

    feat_times = feat_data['feat_times']

    version_features = {k: feat_data[k] for k in version_keys if k in feat_data}

    RESULTS_DIR.mkdir(exist_ok=True)

    # Step 2: Build fixed SL/TP labels
    logger.info(f"\n[STEP 2] Building labels (SL={SL_PCT}%, TP={TP_PCT}%)...")
    labels = build_labels(
        sl_pct=SL_PCT, tp_pct=TP_PCT, max_hold=MAX_HOLD
    )
    label_dist = labels['label'].value_counts().to_dict()
    logger.info(f"  Labels: {labels.shape} | Dist: {label_dist}")

    # Align features with labels
    aligned, aligned_labels = align_features_labels(
        version_features, feat_times, labels
    )
    n_rows = len(aligned_labels)
    logger.info(f"  Aligned to {n_rows} rows")

    # Step 3: Train and evaluate
    master_results = {}  # {version: {threshold: metrics}}
    summary_rows = []

    for vkey in version_keys:
        if vkey not in aligned:
            logger.warning(f"  Skipping {version_names.get(vkey, vkey)}: not in aligned features")
            continue

        vname = version_names[vkey]
        features_df = aligned[vkey]

        if features_df.empty:
            logger.warning(f"  Skipping {vname}: empty")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"[MODEL] {vname} ({features_df.shape[1]} features)")
        logger.info(f"{'='*60}")

        # Split: 12mo train / 1mo test
        split = split_data_temporal(features_df, aligned_labels)
        logger.info(f"  Train: {len(split['X_train'])} | Test: {len(split['X_test'])}")

        # Train
        model = train_xgboost(split['X_train'], split['y_train'], vname)

        # Feature importance
        fi = get_feature_importance(model, list(features_df.columns))

        # Save model
        model_path = RESULTS_DIR / f"model_{vname}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Evaluate at each confidence threshold
        version_results = {}
        for thresh in CONFIDENCE_THRESHOLDS:
            metrics = evaluate_with_confidence(
                model, split['X_test'], split['y_test'],
                split['labels_test'], vname, thresh
            )
            metrics['feature_importance'] = fi
            version_results[str(thresh)] = metrics

            summary_rows.append({
                'version': vname,
                'threshold': thresh,
                'accuracy': metrics['accuracy'],
                'roc_auc': metrics['roc_auc'],
                'n_confident': metrics['n_confident'],
                'n_trades': metrics['n_trades'],
                'win_rate': metrics['win_rate'],
                'total_profit_pct': metrics['total_profit_pct'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown_pct': metrics['max_drawdown_pct'],
            })

        master_results[vname] = version_results

    # Save all results
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_path = RESULTS_DIR / "training_results.json"
    serializable = json.loads(json.dumps(master_results, default=convert))
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    # Save summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)

    # Print master comparison
    logger.info("\n" + "=" * 100)
    logger.info("MASTER COMPARISON — ALL MODELS x CONFIDENCE THRESHOLDS")
    logger.info("=" * 100)
    logger.info(f"{'Version':<15} {'Thresh':>6} {'Acc':>6} {'AUC':>6} {'Conf':>6} "
                f"{'Trades':>7} {'WR%':>6} {'Profit%':>9} {'Sharpe':>7} {'MaxDD%':>8}")
    logger.info("-" * 100)

    for row in sorted(summary_rows, key=lambda x: (x['version'], x['threshold'])):
        logger.info(f"{row['version']:<15} {row['threshold']:>6.1f} {row['accuracy']:>6.3f} "
                    f"{row['roc_auc']:>6.3f} {row['n_confident']:>6} "
                    f"{row['n_trades']:>7} {row['win_rate']:>5.1f}% "
                    f"{row['total_profit_pct']:>8.1f}% {row['sharpe_ratio']:>7.2f} "
                    f"{row['max_drawdown_pct']:>7.1f}%")

    logger.info(f"\nResults: {results_path}")
    logger.info(f"Summary: {summary_path}")

    return master_results


if __name__ == "__main__":
    run_training()
