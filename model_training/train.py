"""
CatBoost Training Pipeline: 5-Matrix Augmented Features with fixed SL/TP labels
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

from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
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


def train_catboost(X_train, y_train, cat_indices: list,
                   X_eval=None, y_eval=None) -> CatBoostClassifier:
    """Train CatBoost classifier with categorical feature support and early stopping."""
    # Map labels: -1->0, 0->1, 1->2 for multiclass
    y_mapped = y_train + 1

    unique_classes = np.unique(y_mapped)
    n_classes = len(unique_classes)

    logger.info(f"  CatBoost: {len(X_train)} samples, {X_train.shape[1]} features, "
                f"{n_classes} classes, {len(cat_indices)} categorical features")

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        random_seed=42,
        verbose=50,
        task_type='GPU',
        cat_features=cat_indices if cat_indices else None,
        auto_class_weights='Balanced',
        early_stopping_rounds=50,
    )

    # F001: Do NOT combine auto_class_weights with sample_weight
    eval_set = None
    if X_eval is not None and y_eval is not None:
        eval_set = (X_eval, y_eval + 1)

    model.fit(X_train, y_mapped, eval_set=eval_set, use_best_model=True)

    if eval_set is not None:
        evals = model.get_evals_result()
        if 'validation' in evals and 'MultiClass' in evals['validation']:
            val_loss = evals['validation']['MultiClass']
            logger.info(f"  Eval loss: first={val_loss[0]:.4f}, "
                        f"best={min(val_loss):.4f} @ iter {val_loss.index(min(val_loss))}, "
                        f"last={val_loss[-1]:.4f} (total {len(val_loss)} iters)")

    return model


def evaluate_with_confidence(model, X_test, y_test, labels_test,
                             threshold: float) -> dict:
    """
    Evaluate model at a given confidence threshold.
    threshold=0.0 means no filter (baseline).
    """
    y_pred_proba = model.predict_proba(X_test)
    y_pred_mapped = np.argmax(y_pred_proba, axis=1)
    y_pred = y_pred_mapped - 1  # back to -1/0/1

    max_proba = y_pred_proba.max(axis=1)
    if threshold > 0:
        conf_mask = max_proba >= threshold
    else:
        conf_mask = np.ones(len(y_pred), dtype=bool)

    n_total = len(y_pred)
    n_confident = conf_mask.sum()

    y_pred_conf = y_pred[conf_mask]
    y_test_conf = y_test[conf_mask]
    y_test_mapped_conf = y_test_conf + 1

    if len(y_pred_conf) == 0:
        logger.info(f"  @{threshold}: No predictions above threshold")
        return _empty_metrics(threshold, n_total)

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

    logger.info(f"  @{threshold:.1f}: Acc={acc:.3f} | AUC={roc_auc:.3f} | "
                f"Confident={n_confident}/{n_total} | "
                f"Trades={n_trades} | WR={win_rate:.1f}% | "
                f"Profit={total_profit:.1f}% | Sharpe={sharpe:.2f} | "
                f"MaxDD={max_drawdown:.1f}%")

    return metrics


def _empty_metrics(threshold, n_total):
    return {
        'threshold': threshold,
        'accuracy': 0, 'roc_auc': 0,
        'confusion_matrix': [[0]*3]*3, 'classification_report': {},
        'n_total_test': n_total, 'n_confident': 0,
        'n_trades': 0, 'win_rate': 0, 'total_profit_pct': 0,
        'avg_gain_pct': 0, 'sharpe_ratio': 0, 'max_drawdown_pct': 0,
        'equity_curve': [0], 'predictions': [], 'actuals': [],
    }


def get_feature_importance(model, feature_names, top_n: int = 30) -> list:
    importance = model.get_feature_importance()
    indices = np.argsort(importance)[::-1][:top_n]
    return [(feature_names[i], float(importance[i])) for i in indices]


def run_training():
    """
    Full pipeline: ETL 5-matrix features (13mo trim) -> Fixed SL/TP labels ->
    Train CatBoost -> Evaluate at confidence thresholds 0.0/0.6/0.7.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("=" * 70)
    logger.info("ML TRAINING PIPELINE -- 5-MATRIX AUGMENTED + CATBOOST")
    logger.info(f"SL={SL_PCT}% | TP={TP_PCT}% | Max Hold={MAX_HOLD}")
    logger.info(f"Confidence thresholds: {CONFIDENCE_THRESHOLDS}")
    logger.info("=" * 70)

    # Step 1: Build features
    logger.info("\n[STEP 1] Building 5-matrix augmented features (13-month trim)...")
    feat_data = run_etl_features(trim_months=13)

    features = feat_data['features']
    feat_times = feat_data['feat_times']
    cat_indices = feat_data['cat_indices']

    logger.info(f"  Features: {features.shape}")
    logger.info(f"  Categorical indices: {len(cat_indices)}")

    RESULTS_DIR.mkdir(exist_ok=True)

    # Step 2: Build labels
    logger.info(f"\n[STEP 2] Building labels (SL={SL_PCT}%, TP={TP_PCT}%)...")
    labels = build_labels(sl_pct=SL_PCT, tp_pct=TP_PCT, max_hold=MAX_HOLD)
    label_dist = labels['label'].value_counts().to_dict()
    logger.info(f"  Labels: {labels.shape} | Dist: {label_dist}")

    # Align
    aligned_features, aligned_labels = align_features_labels(
        features, feat_times, labels
    )
    n_rows = len(aligned_labels)
    logger.info(f"  Aligned to {n_rows} rows")

    # Recompute cat_indices after alignment (columns unchanged, just rows filtered)
    # cat_indices remain the same since columns are unchanged

    # Step 3: Split
    logger.info(f"\n[STEP 3] Temporal split (12mo train / 1mo test)...")
    split = split_data_temporal(aligned_features, aligned_labels)
    logger.info(f"  Train: {len(split['X_train'])} | Test: {len(split['X_test'])}")

    # Diagnostic: label distribution per split
    train_labels = split['y_train']
    test_labels = split['y_test']
    for name, arr in [("Train", train_labels), ("Test", test_labels)]:
        classes, counts = np.unique(arr, return_counts=True)
        total = len(arr)
        dist_str = ", ".join([f"{c}: {cnt} ({cnt/total*100:.1f}%)" for c, cnt in zip(classes, counts)])
        logger.info(f"  {name} label dist: {dist_str}")

    # Step 4a: Train on BASE features only (no lags) for diagnostic
    base_cols = [c for c in aligned_features.columns if '_lag' not in c]
    base_cat_indices = [i for i, c in enumerate(base_cols) if c.startswith("AGMM_")]
    logger.info(f"\n[STEP 4a] Diagnostic: Training on BASE features only ({len(base_cols)} cols)...")
    X_train_base = split['X_train'][base_cols]
    X_test_base = split['X_test'][base_cols]
    model_base = train_catboost(
        X_train_base, split['y_train'], base_cat_indices,
        X_eval=X_test_base, y_eval=split['y_test']
    )
    base_best_iter = model_base.get_best_iteration()
    logger.info(f"  Base-only best_iteration: {base_best_iter}")
    base_metrics = evaluate_with_confidence(
        model_base, X_test_base, split['y_test'],
        split['labels_test'], 0.0
    )
    logger.info(f"  Base-only result: AUC={base_metrics['roc_auc']:.3f}, "
                f"WR={base_metrics['win_rate']:.1f}%, Trades={base_metrics['n_trades']}")

    # Step 4b: Train on FULL features (with lags)
    logger.info(f"\n[STEP 4b] Training CatBoost on FULL features ({aligned_features.shape[1]} cols, GPU + early stopping)...")
    model = train_catboost(
        split['X_train'], split['y_train'], cat_indices,
        X_eval=split['X_test'], y_eval=split['y_test']
    )
    full_best_iter = model.get_best_iteration()
    logger.info(f"  Full best_iteration: {full_best_iter}")

    # Feature importance
    fi = get_feature_importance(model, list(aligned_features.columns))

    # Save model
    model_path = RESULTS_DIR / "model_catboost.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Also save in CatBoost native format
    model.save_model(str(RESULTS_DIR / "model_catboost.cbm"))

    # Step 5: Evaluate at confidence thresholds
    logger.info(f"\n[STEP 5] Evaluating at confidence thresholds...")
    all_results = {}
    summary_rows = []

    for thresh in CONFIDENCE_THRESHOLDS:
        metrics = evaluate_with_confidence(
            model, split['X_test'], split['y_test'],
            split['labels_test'], thresh
        )
        metrics['feature_importance'] = fi
        all_results[str(thresh)] = metrics

        summary_rows.append({
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

    # Save results
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_path = RESULTS_DIR / "training_results.json"
    serializable = json.loads(json.dumps(all_results, default=convert))
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)

    # Print summary
    logger.info("\n" + "=" * 90)
    logger.info("RESULTS -- 5-MATRIX AUGMENTED CATBOOST")
    logger.info("=" * 90)
    logger.info(f"{'Thresh':>6} {'Acc':>6} {'AUC':>6} {'Conf':>6} "
                f"{'Trades':>7} {'WR%':>6} {'Profit%':>9} {'Sharpe':>7} {'MaxDD%':>8}")
    logger.info("-" * 90)

    for row in sorted(summary_rows, key=lambda x: x['threshold']):
        logger.info(f"{row['threshold']:>6.1f} {row['accuracy']:>6.3f} "
                    f"{row['roc_auc']:>6.3f} {row['n_confident']:>6} "
                    f"{row['n_trades']:>7} {row['win_rate']:>5.1f}% "
                    f"{row['total_profit_pct']:>8.1f}% {row['sharpe_ratio']:>7.2f} "
                    f"{row['max_drawdown_pct']:>7.1f}%")

    logger.info(f"\nModel: {model_path}")
    logger.info(f"Results: {results_path}")
    logger.info(f"Summary: {summary_path}")

    return all_results


if __name__ == "__main__":
    run_training()
