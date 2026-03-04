"""
CatBoost Training Pipeline — Binary (LONG vs SHORT)
Transposed calendar-grid features with native categorical handling via CatBoost Pool.

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
import time
from pathlib import Path
from typing import Tuple

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score
)

from data.target_labeling import create_sl_tp_labels
from model_training.build_labels import load_labels, LABELS_PATH
from core.config import DATA_DIR

logger = logging.getLogger(__name__)

ACTUAL_DATA_DIR = Path(__file__).parent / "actual_data"
ENCODED_DIR = Path(__file__).parent / "encoded_data"
RESULTS_DIR = Path(__file__).parent / "results"

# SL/TP config
SL_PCT = 2.0
TP_PCT = 4.0
MAX_HOLD = 288

CONFIDENCE_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7]
TEST_DAYS = 30

# Timeframe filter: set to None to use all 55 combos, or list specific TFs
TF_FILTER = ['5M', '15M', '30M', '1H', '2H', '4H']  # Youngs + Adults


# ============================================================================
# Labels
# ============================================================================

def build_labels(sl_pct: float = SL_PCT, tp_pct: float = TP_PCT,
                 max_hold: int = MAX_HOLD) -> pd.DataFrame:
    """Load precomputed labels from CSV. Falls back to computing if not found."""
    if LABELS_PATH.exists():
        logger.info(f"  Loading precomputed labels: {LABELS_PATH}")
        return load_labels()

    logger.info("  Precomputed labels not found, computing from scratch...")
    ml_path = ACTUAL_DATA_DIR / "ml_data_5M.csv"
    csv_path = ml_path if ml_path.exists() else DATA_DIR / "testing_data_5M.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"5M CSV not found: {csv_path}")
    logger.info(f"  Label source: {csv_path}")

    price_data = pd.read_csv(csv_path)
    if 'index' in price_data.columns:
        price_data = price_data.drop(columns=['index'])
    price_data['Open Time'] = pd.to_datetime(price_data['Open Time'])

    labels_df = create_sl_tp_labels(
        price_data, sl_pct=sl_pct, tp_pct=tp_pct,
        max_hold_periods=max_hold, price_col='Close',
        high_col='High', low_col='Low', timestamp_col='Open Time'
    )
    return labels_df


# ============================================================================
# Alignment
# ============================================================================

def align_features_labels(features: pd.DataFrame, feat_times: pd.Series,
                          labels: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align feature matrix with labels by timestamp."""
    label_times = pd.to_datetime(labels['timestamp'])
    common_times = set(feat_times.values) & set(label_times.values)
    logger.info(f"  Overlapping timestamps: {len(common_times)}")

    if len(common_times) == 0:
        raise ValueError(
            "No timestamp overlap. "
            f"Features: {feat_times.min()} to {feat_times.max()}, "
            f"Labels: {label_times.min()} to {label_times.max()}"
        )

    common_sorted = sorted(common_times)
    feat_mask = feat_times.isin(common_sorted)
    label_mask = label_times.isin(common_sorted)
    return (
        features[feat_mask.values].reset_index(drop=True),
        labels[label_mask.values].reset_index(drop=True),
    )


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_catboost(X_train, y_train, cat_features=None,
                   X_eval=None, y_eval=None) -> CatBoostClassifier:
    """Train CatBoost binary: SHORT(0) / LONG(1).
    Uses Pool with cat_features for native categorical handling."""

    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    n_features = X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train.columns)
    n_cat = len(cat_features) if cat_features else 0

    CLASS_NAMES = {0: 'SHORT', 1: 'LONG'}
    logger.info(f"  CatBoost binary: {total} samples, {n_features} features "
                f"({n_cat} categorical, {n_features - n_cat} numeric)")
    for c, cnt in zip(classes, counts):
        logger.info(f"    {CLASS_NAMES.get(int(c), c)}: {cnt} ({cnt/total*100:.1f}%)")

    # Build Pool objects for native categorical support
    train_pool = Pool(X_train, y_train, cat_features=cat_features)

    eval_pool = None
    if X_eval is not None and y_eval is not None:
        eval_pool = Pool(X_eval, y_eval, cat_features=cat_features)

    model = CatBoostClassifier(
        iterations=2000,
        depth=8,
        learning_rate=0.03,
        l2_leaf_reg=10,
        min_data_in_leaf=100,
        loss_function='Logloss',
        eval_metric='Logloss',
        random_seed=42,
        verbose=100,
        task_type='GPU',
        auto_class_weights='Balanced',
        early_stopping_rounds=200,
        subsample=0.8,
        bootstrap_type='Bernoulli',
    )

    model.fit(train_pool, eval_set=eval_pool, use_best_model=True)

    best_iter = model.get_best_iteration()
    logger.info(f"  best_iteration: {best_iter}")

    if eval_pool is not None:
        evals = model.get_evals_result()
        if 'validation' in evals:
            for metric_name, vals in evals['validation'].items():
                if vals:
                    best_val = min(vals)
                    logger.info(f"  Eval {metric_name}: first={vals[0]:.4f}, "
                                f"best={best_val:.4f} @ iter {vals.index(best_val)}, "
                                f"last={vals[-1]:.4f} ({len(vals)} iters)")

    return model


def evaluate_with_confidence(model, X_test, y_test, labels_test,
                             threshold: float, cat_features=None) -> dict:
    """
    Evaluate binary model at a confidence threshold.
    Confidence = max(P(SHORT), P(LONG)).
    """
    test_pool = Pool(X_test, cat_features=cat_features)
    y_pred_proba = model.predict_proba(test_pool)  # [P(SHORT), P(LONG)]
    p_short = y_pred_proba[:, 0]
    p_long = y_pred_proba[:, 1]

    # Predicted class: argmax
    y_pred = np.argmax(y_pred_proba, axis=1)  # 0=SHORT, 1=LONG

    # Trade confidence = max probability
    trade_confidence = np.maximum(p_short, p_long)

    # Only trade when confidence >= threshold
    trade_mask = trade_confidence >= threshold
    n_total = len(y_pred)
    n_confident = int(trade_mask.sum())

    y_pred_trade = y_pred[trade_mask]
    y_test_trade = y_test[trade_mask]

    if len(y_pred_trade) == 0:
        logger.info(f"  @{threshold:.2f}: No predictions above threshold")
        return _empty_metrics(threshold, n_total)

    # Overall accuracy
    acc_all = accuracy_score(y_test, y_pred)

    # Confident-only accuracy
    acc_conf = accuracy_score(y_test_trade, y_pred_trade)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = classification_report(y_test, y_pred, labels=[0, 1],
                                   target_names=['SHORT', 'LONG'],
                                   output_dict=True, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_test, p_long)
    except Exception:
        roc_auc = 0.0

    # Trading simulation
    trades = []
    equity = [0.0]
    for idx in np.where(trade_mask)[0]:
        pred = y_pred[idx]
        actual = y_test[idx]

        if pred == actual:
            gain = TP_PCT
        else:
            gain = -SL_PCT

        trades.append({
            'index': int(idx), 'prediction': int(pred), 'actual': int(actual),
            'gain_pct': gain, 'confidence': float(trade_confidence[idx]),
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
        sharpe = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(n_trades) if len(returns) > 1 else 0
        equity_arr = np.array(equity)
        peak = np.maximum.accumulate(equity_arr)
        max_drawdown = float((equity_arr - peak).min())
    else:
        win_rate = total_profit = avg_gain = sharpe = max_drawdown = 0.0

    metrics = {
        'threshold': threshold, 'accuracy': acc_all, 'accuracy_confident': acc_conf,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(), 'classification_report': report,
        'n_total_test': n_total, 'n_confident': n_confident, 'n_trades': n_trades,
        'win_rate': win_rate, 'total_profit_pct': total_profit,
        'avg_gain_pct': avg_gain, 'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_drawdown, 'equity_curve': equity,
        'predictions': y_pred_trade.tolist(), 'actuals': y_test_trade.tolist(),
    }

    if threshold == CONFIDENCE_THRESHOLDS[0]:
        metrics['y_pred_proba'] = y_pred_proba.tolist()
        metrics['y_test_full'] = y_test.tolist()
    if n_trades > 0:
        metrics['trades'] = trades_df.to_dict('records')

    logger.info(f"  @{threshold:.2f}: Acc={acc_all:.3f} | ConfAcc={acc_conf:.3f} | "
                f"AUC={roc_auc:.3f} | Trades={n_trades}/{n_total} | "
                f"WR={win_rate:.1f}% | Profit={total_profit:.1f}% | "
                f"Sharpe={sharpe:.2f} | MaxDD={max_drawdown:.1f}%")

    return metrics


def _empty_metrics(threshold, n_total):
    return {
        'threshold': threshold, 'accuracy': 0, 'accuracy_confident': 0,
        'roc_auc': 0,
        'confusion_matrix': [[0, 0], [0, 0]],
        'classification_report': {},
        'n_total_test': n_total, 'n_confident': 0, 'n_trades': 0,
        'win_rate': 0, 'total_profit_pct': 0,
        'avg_gain_pct': 0, 'sharpe_ratio': 0, 'max_drawdown_pct': 0,
        'equity_curve': [0], 'predictions': [], 'actuals': [],
    }


def get_feature_importance(model, feature_names, top_n: int = 30) -> list:
    importance = model.get_feature_importance()
    indices = np.argsort(importance)[::-1][:top_n]
    return [(feature_names[i], float(importance[i])) for i in indices]


def save_confusion_matrix_image(cm, labels, save_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix (Binary: SHORT vs LONG)')
        fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
        logger.info(f"  Saved: {save_path}")
    except Exception as e:
        logger.warning(f"  Could not save confusion matrix: {e}")


# ============================================================================
# Main Pipeline
# ============================================================================

def run_training():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    t_start = time.time()

    logger.info("=" * 70)
    logger.info("ML PIPELINE v2 — TRANSPOSED GRID (Binary: SHORT vs LONG)")
    logger.info(f"SL={SL_PCT}% | TP={TP_PCT}% | max_hold={MAX_HOLD}")
    logger.info("=" * 70)

    # Step 1: Load parquet
    logger.info("\n[STEP 1] Loading transposed feature matrix...")
    parquet_path = ENCODED_DIR / "feature_matrix.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing: {parquet_path}\nRun: python model_training/encode.py")

    feat_df = pd.read_parquet(parquet_path)
    logger.info(f"  Parquet: {feat_df.shape[0]} rows, {feat_df.shape[1]} cols")

    # Identify feature types
    feature_names = [c for c in feat_df.columns if c != 'time']
    cat_feature_names = [c for c in feature_names if c.startswith('cell_')]
    num_feature_names = [c for c in feature_names
                         if c.startswith('sum_') or c.startswith('slope_long')]
    logger.info(f"  All features: {len(feature_names)} total "
                f"({len(cat_feature_names)} categorical, {len(num_feature_names)} numeric)")

    # Apply TF filter if set
    if TF_FILTER:
        logger.info(f"  TF_FILTER: {TF_FILTER}")
        def _tf_match(col):
            """Check if column belongs to a filtered TF."""
            if not col.startswith('cell_'):
                return col.startswith('sum_') or col.startswith('slope_long')
            # cell_{TF}_w{ws}[_lag{n}] — extract TF between first _ and _w
            parts = col.split('_w')
            if len(parts) >= 2:
                tf = parts[0].replace('cell_', '')
                return tf in TF_FILTER
            return False

        keep_cols = [c for c in feature_names if _tf_match(c)]
        drop_cols = [c for c in feature_names if c not in keep_cols]
        feat_df.drop(columns=drop_cols, inplace=True)
        feature_names = [c for c in feat_df.columns if c != 'time']
        cat_feature_names = [c for c in feature_names if c.startswith('cell_')]
        num_feature_names = [c for c in feature_names
                             if c.startswith('sum_') or c.startswith('slope_long')]
        logger.info(f"  After filter: {len(feature_names)} features "
                    f"({len(cat_feature_names)} cat, {len(num_feature_names)} num)")

    RESULTS_DIR.mkdir(exist_ok=True)

    # Step 2: Build labels
    logger.info(f"\n[STEP 2] Building labels (SL={SL_PCT}%, TP={TP_PCT}%, max_hold={MAX_HOLD})...")
    labels = build_labels()
    label_dist = labels['label'].value_counts().to_dict()
    logger.info(f"  Labels: {labels.shape} | Dist: {label_dist}")

    # Align features with labels
    feat_times = pd.to_datetime(feat_df['time'])
    feat_df.drop(columns=['time'], inplace=True)
    aligned_features, aligned_labels = align_features_labels(feat_df, feat_times, labels)
    del feat_df
    import gc; gc.collect()
    logger.info(f"  Aligned: {len(aligned_labels)} rows")

    # Binary: drop NO_TRADE (label == 0), keep LONG(1) and SHORT(-1)
    trade_mask = aligned_labels['label'] != 0
    aligned_features = aligned_features[trade_mask.values].reset_index(drop=True)
    aligned_labels = aligned_labels[trade_mask.values].reset_index(drop=True)
    logger.info(f"  After dropping NO_TRADE: {len(aligned_labels)} rows")

    # Map labels: -1 (SHORT) -> 0, +1 (LONG) -> 1
    y_raw = aligned_labels['label'].values
    y = ((y_raw + 1) // 2).astype(np.int8)  # -1->0, +1->1

    label_dist_binary = pd.Series(y).value_counts().to_dict()
    logger.info(f"  Binary dist: {{0: SHORT={label_dist_binary.get(0, 0)}, "
                f"1: LONG={label_dist_binary.get(1, 0)}}}")
    logger.info(f"  Feature:sample ratio = 1:{len(aligned_labels) / len(feature_names):.0f}")

    # Step 3: Label-encode categorical columns as int8
    logger.info("\n[STEP 3] Label-encoding categorical columns...")

    cat_cols_present = [c for c in cat_feature_names if c in aligned_features.columns]
    num_cols_present = [c for c in num_feature_names if c in aligned_features.columns]

    from model_training.encode import _CELL_STRINGS
    cell_to_int = {s: i for i, s in enumerate(_CELL_STRINGS)}
    logger.info(f"  Cell vocabulary: {len(cell_to_int)} unique values")

    for col in cat_cols_present:
        aligned_features[col] = aligned_features[col].map(cell_to_int).fillna(0).astype(np.int8)

    cat_feature_indices = [i for i, c in enumerate(aligned_features.columns)
                           if c.startswith('cell_')]
    logger.info(f"  {len(cat_cols_present)} cat cols encoded to int8, "
                f"{len(num_cols_present)} num cols")
    logger.info(f"  cat_features: {len(cat_feature_indices)} indices for Pool")
    logger.info(f"  Shape: {aligned_features.shape}")

    # Step 4: Temporal split
    label_times = pd.to_datetime(aligned_labels['timestamp'])
    n = len(aligned_features)

    cutoff = label_times.max() - pd.Timedelta(days=TEST_DAYS)
    train_end = int((label_times <= cutoff).sum())

    logger.info(f"\n[STEP 4] Temporal split (test = last {TEST_DAYS} days, cutoff: {cutoff})...")
    logger.info(f"  Split: {train_end} train / {n - train_end} test")
    logger.info(f"  Train: {label_times.iloc[0]} to {label_times.iloc[train_end - 1]}")
    logger.info(f"  Test:  {label_times.iloc[train_end]} to {label_times.iloc[-1]}")

    X_train = aligned_features.iloc[:train_end].reset_index(drop=True)
    X_test = aligned_features.iloc[train_end:].reset_index(drop=True)
    y_train, y_test = y[:train_end], y[train_end:]
    labels_test = aligned_labels.iloc[train_end:].reset_index(drop=True)
    del aligned_features

    for name, arr in [("Train", y_train), ("Test", y_test)]:
        classes, counts = np.unique(arr, return_counts=True)
        total = len(arr)
        CLASS_NAMES = {0: 'SHORT', 1: 'LONG'}
        dist_str = ", ".join([f"{CLASS_NAMES.get(int(c), c)}: {cnt} ({cnt/total*100:.1f}%)"
                              for c, cnt in zip(classes, counts)])
        logger.info(f"  {name}: {dist_str}")

    # Step 5: Train
    logger.info(f"\n[STEP 5] Training CatBoost ({len(feature_names)} features, Binary, GPU)...")
    model = train_catboost(X_train, y_train, cat_features=cat_feature_indices,
                           X_eval=X_test, y_eval=y_test)
    best_iter = model.get_best_iteration()

    fi = get_feature_importance(model, list(X_train.columns))
    logger.info("\n  Top 20 features:")
    for fname, fval in fi[:20]:
        logger.info(f"    {fname}: {fval:.4f}")

    all_importance = model.get_feature_importance()
    n_used = int((all_importance > 0).sum())
    logger.info(f"  Features used: {n_used}/{len(feature_names)}")

    # Save model
    model_path = RESULTS_DIR / "model_catboost.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    model.save_model(str(RESULTS_DIR / "model_catboost.cbm"))

    # Step 6: Evaluate
    logger.info(f"\n[STEP 6] Evaluating at confidence thresholds...")
    all_results = {}
    summary_rows = []

    for thresh in CONFIDENCE_THRESHOLDS:
        metrics = evaluate_with_confidence(model, X_test, y_test, labels_test, thresh,
                                           cat_features=cat_feature_indices)
        metrics['feature_importance'] = fi
        all_results[str(thresh)] = metrics
        summary_rows.append({
            'threshold': thresh, 'accuracy': metrics['accuracy'],
            'accuracy_confident': metrics['accuracy_confident'],
            'roc_auc': metrics['roc_auc'], 'n_confident': metrics['n_confident'],
            'n_trades': metrics['n_trades'],
            'win_rate': metrics['win_rate'],
            'total_profit_pct': metrics['total_profit_pct'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown_pct': metrics['max_drawdown_pct'],
        })

    # Save confusion matrix (binary)
    baseline_key = str(CONFIDENCE_THRESHOLDS[0])
    if baseline_key in all_results:
        cm = np.array(all_results[baseline_key]['confusion_matrix'])
        save_confusion_matrix_image(cm, ['SHORT', 'LONG'],
                                     RESULTS_DIR / "confusion_matrix.png")

    # Save results
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    results_path = RESULTS_DIR / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(json.loads(json.dumps(all_results, default=convert)), f, indent=2)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)

    elapsed = time.time() - t_start

    logger.info("\n" + "=" * 100)
    logger.info("RESULTS — BINARY (SHORT vs LONG)")
    logger.info(f"SL={SL_PCT}% | TP={TP_PCT}% | max_hold={MAX_HOLD} | best_iter={best_iter}")
    logger.info(f"Features: {len(feature_names)} total ({len(cat_feature_names)} cat + "
                f"{len(num_feature_names)} num) | {n_used} used")
    logger.info(f"Time: {elapsed:.0f}s")
    logger.info("=" * 100)
    logger.info(f"{'Thresh':>7} {'Acc':>6} {'ConfAcc':>8} {'AUC':>6} {'Trades':>7} "
                f"{'WR%':>6} {'Profit%':>9} {'Sharpe':>7} {'MaxDD%':>8}")
    logger.info("-" * 100)

    for row in sorted(summary_rows, key=lambda x: x['threshold']):
        logger.info(f"{row['threshold']:>7.2f} {row['accuracy']:>6.3f} "
                    f"{row['accuracy_confident']:>8.3f} "
                    f"{row['roc_auc']:>6.3f} {row['n_trades']:>7} "
                    f"{row['win_rate']:>5.1f}% "
                    f"{row['total_profit_pct']:>8.1f}% {row['sharpe_ratio']:>7.2f} "
                    f"{row['max_drawdown_pct']:>7.1f}%")

    logger.info(f"\nModel: {model_path}")
    logger.info(f"Results: {results_path}")

    return all_results


if __name__ == "__main__":
    run_training()
