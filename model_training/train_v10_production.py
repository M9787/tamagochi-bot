"""
V10 Production Model Training — Train on ALL 6yr data, save models for live deployment.

Only run this AFTER train_v10_2yr_oos.py passes all criteria.

Split:
  Train: 2020-02-17 -> 2026-01-15 (everything minus 1 month)
  Val:   2026-01-15 -> 2026-02-15 (last month, early stopping only)

No test set — this is the production model that uses all available data.

Saves:
  - 3 CatBoost models (seeds 42, 123, 777)
  - production_metadata.json
  - feature_importance.csv

Output: results_v10/production/

Usage:
  python model_training/train_v10_production.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import logging
import json
import time
from pathlib import Path
from datetime import datetime

from catboost import CatBoostClassifier, Pool

from model_training.build_labels import load_labels

logger = logging.getLogger(__name__)

ENCODED_DIR = Path(__file__).parent / "encoded_data"
RESULTS_DIR = Path(__file__).parent / "results_v10" / "production"

# Class names
CLASS_NAMES = {0: 'NO_TRADE', 1: 'LONG', 2: 'SHORT'}

# V7 d8 winner params (identical to 2yr OOS test)
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

# Production split — use all data
SPLIT = {
    'train_start': pd.Timestamp('2020-02-17'),
    'train_end': pd.Timestamp('2026-01-15'),     # Everything minus 1 month
    'val_end': pd.Timestamp('2026-02-15'),        # Last month = early stopping only
}

# Default production threshold
RECOMMENDED_THRESHOLD = 0.75
SL_PCT = 2.0
TP_PCT = 4.0


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


def run_production_training():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    t_start = time.time()

    logger.info("=" * 90)
    logger.info("V10 PRODUCTION MODEL TRAINING — Full 6yr Data")
    logger.info(f"  Config: V7 d8 (depth={MODEL_PARAMS['depth']}, iter={MODEL_PARAMS['iterations']})")
    logger.info(f"  Seeds: {SEEDS}")
    logger.info(f"  Train: {SPLIT['train_start'].date()} -> {SPLIT['train_end'].date()}")
    logger.info(f"  Val:   {SPLIT['train_end'].date()} -> {SPLIT['val_end'].date()} (early stopping only)")
    logger.info(f"  Recommended threshold: {RECOMMENDED_THRESHOLD}")
    logger.info("=" * 90)

    # Check that 2yr OOS passed
    oos_results_path = Path(__file__).parent / "results_v10" / "2yr_oos" / "v10_2yr_oos_results.json"
    if oos_results_path.exists():
        with open(oos_results_path) as f:
            oos_results = json.load(f)
        verdict = oos_results.get('pass_fail', {}).get('overall', 'UNKNOWN')
        logger.info(f"\n  2yr OOS verdict: {verdict}")
        if verdict != 'PASS':
            logger.error("  2yr OOS did NOT pass! Training anyway but results may not be reliable.")
    else:
        logger.warning(f"  2yr OOS results not found at {oos_results_path}")
        logger.warning("  Proceeding without OOS validation.")

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

    # ------------------------------------------------------------------
    # Step 2: Load labels
    # ------------------------------------------------------------------
    logger.info("\n[STEP 2] Loading 3-class labels...")
    labels = load_labels()
    labels = prepare_3class_labels(labels)

    # ------------------------------------------------------------------
    # Step 3: Align
    # ------------------------------------------------------------------
    logger.info("\n[STEP 3] Aligning features and labels...")
    X_aligned, y_df, times = align_features_labels(features, labels)
    del features

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 4: Split data
    # ------------------------------------------------------------------
    logger.info("\n[STEP 4] Splitting data...")

    train_mask = (times >= SPLIT['train_start']) & (times < SPLIT['train_end'])
    val_mask = (times >= SPLIT['train_end']) & (times < SPLIT['val_end'])

    n_train = int(train_mask.sum())
    n_val = int(val_mask.sum())

    logger.info(f"  Train: {n_train:>8,} rows ({SPLIT['train_start'].date()} -> {SPLIT['train_end'].date()})")
    logger.info(f"  Val:   {n_val:>8,} rows ({SPLIT['train_end'].date()} -> {SPLIT['val_end'].date()})")

    if n_train == 0 or n_val == 0:
        raise ValueError(f"Empty split: train={n_train}, val={n_val}")

    X_train = X_aligned.loc[train_mask.values, feature_cols].reset_index(drop=True)
    X_val = X_aligned.loc[val_mask.values, feature_cols].reset_index(drop=True)

    y_train = y_df.loc[train_mask.values, 'label_3class'].values.astype(np.int8)
    y_val = y_df.loc[val_mask.values, 'label_3class'].values.astype(np.int8)

    # Label distributions
    for name, arr in [('Train', y_train), ('Val', y_val)]:
        unique, counts = np.unique(arr, return_counts=True)
        dist_str = ", ".join([f"{CLASS_NAMES[int(u)]}={c}" for u, c in zip(unique, counts)])
        logger.info(f"  {name} labels: {dist_str}")

    # ------------------------------------------------------------------
    # Step 5: Train per seed
    # ------------------------------------------------------------------
    logger.info(f"\n[STEP 5] Training {len(SEEDS)} production models...")

    all_fi = {}
    seed_info = []

    for seed in SEEDS:
        logger.info(f"\n  {'='*60}")
        logger.info(f"  SEED {seed}")
        logger.info(f"  {'='*60}")

        seed_t0 = time.time()

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

        logger.info(f"  Top 10 features:")
        for fname, fval in fi_sorted[:10]:
            logger.info(f"    {fname}: {fval:.2f}")
            if fname not in all_fi:
                all_fi[fname] = []
            all_fi[fname].append(fval)

        for fname, fval in fi_sorted[10:40]:
            if fname not in all_fi:
                all_fi[fname] = []
            all_fi[fname].append(fval)

        # Save model
        model_path = RESULTS_DIR / f"production_model_s{seed}.cbm"
        model.save_model(str(model_path))
        model_size = model_path.stat().st_size / 1024 / 1024
        logger.info(f"  Model saved: {model_path} ({model_size:.1f} MB)")

        seed_time = time.time() - seed_t0
        seed_info.append({
            'seed': seed,
            'best_iteration': best_iter,
            'n_features_used': n_used,
            'model_file': f"production_model_s{seed}.cbm",
            'model_size_mb': round(model_size, 2),
            'training_time_sec': round(seed_time, 1),
        })

        logger.info(f"  Training time: {seed_time:.0f}s")

    # ------------------------------------------------------------------
    # Step 6: Save feature importance
    # ------------------------------------------------------------------
    fi_avg = sorted([(k, float(np.mean(v))) for k, v in all_fi.items()], key=lambda x: -x[1])
    fi_df = pd.DataFrame(fi_avg, columns=['feature', 'avg_importance'])
    fi_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    logger.info(f"\n  Feature importance saved ({len(fi_avg)} features)")

    # ------------------------------------------------------------------
    # Step 7: Save production metadata
    # ------------------------------------------------------------------
    logger.info(f"\n[STEP 7] Saving production metadata...")

    # Load 2yr OOS results if available for reference
    oos_ref = {}
    if oos_results_path.exists():
        with open(oos_results_path) as f:
            oos_data = json.load(f)
        agg = oos_data.get('aggregate', {})
        oos_ref = {
            'auc': agg.get('mean_auc'),
            'profit_070': agg.get('mean_profit_070'),
            'profit_075': agg.get('mean_profit_075'),
            'verdict': oos_data.get('pass_fail', {}).get('overall'),
        }

    metadata = {
        'model_version': 'V10',
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_range': {
            'start': str(SPLIT['train_start'].date()),
            'end': str(SPLIT['val_end'].date()),
        },
        'n_training_rows': n_train,
        'n_val_rows': n_val,
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'model_params': {k: str(v) if not isinstance(v, (int, float, list, bool)) else v
                         for k, v in MODEL_PARAMS.items()},
        'seeds': SEEDS,
        'seed_details': seed_info,
        'recommended_threshold': RECOMMENDED_THRESHOLD,
        'sl_pct': SL_PCT,
        'tp_pct': TP_PCT,
        'oos_results': oos_ref,
        'top_20_features': fi_avg[:20],
    }

    metadata_path = RESULTS_DIR / "production_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(json.loads(json.dumps(metadata, default=convert_for_json)),
                  f, indent=2)

    elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    logger.info(f"\n[VERIFICATION]")
    for seed in SEEDS:
        model_path = RESULTS_DIR / f"production_model_s{seed}.cbm"
        if model_path.exists():
            size = model_path.stat().st_size
            if size > 100 * 1024:  # > 100KB
                logger.info(f"  OK: {model_path.name} ({size / 1024 / 1024:.1f} MB)")
            else:
                logger.warning(f"  WARN: {model_path.name} is only {size / 1024:.1f} KB (expected > 100KB)")
        else:
            logger.error(f"  FAIL: {model_path.name} missing!")

    # Test that models load correctly
    try:
        test_model = CatBoostClassifier()
        test_model.load_model(str(RESULTS_DIR / f"production_model_s{SEEDS[0]}.cbm"))
        logger.info(f"  OK: Model loads successfully (s{SEEDS[0]})")
    except Exception as e:
        logger.error(f"  FAIL: Model load failed: {e}")

    if metadata_path.exists():
        with open(metadata_path) as f:
            loaded = json.load(f)
        if len(loaded.get('feature_names', [])) == len(feature_cols):
            logger.info(f"  OK: Metadata has {len(feature_cols)} feature names")
        else:
            logger.error(f"  FAIL: Metadata feature count mismatch")
    else:
        logger.error(f"  FAIL: Metadata file missing!")

    # ------------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*90}")
    logger.info(f"V10 PRODUCTION MODEL TRAINING COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"  Models: {len(SEEDS)} x CatBoost d8")
    logger.info(f"  Training data: {n_train:,} rows ({SPLIT['train_start'].date()} -> {SPLIT['train_end'].date()})")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Recommended threshold: {RECOMMENDED_THRESHOLD}")
    logger.info(f"  Output: {RESULTS_DIR}")
    logger.info(f"\n  >>> Next step: python model_training/live_predict.py <<<")
    logger.info(f"{'='*90}")

    return metadata


if __name__ == "__main__":
    run_production_training()
