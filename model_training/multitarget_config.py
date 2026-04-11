"""
Multi-Target Configuration — Shared constants for 8 SL/TP target system.

8 base models (scalp → position) + 8 stacking models.
"""
import pandas as pd
from pathlib import Path

# ============================================================================
# 8 Target Configurations
# ============================================================================
TARGET_CONFIGS = {
    'T1': {'sl': 0.5, 'tp': 1.0, 'max_hold': 72,  'cooldown': 15, 'category': 'Scalp'},
    'T2': {'sl': 0.5, 'tp': 1.5, 'max_hold': 108, 'cooldown': 15, 'category': 'Scalp'},
    'T3': {'sl': 0.5, 'tp': 2.0, 'max_hold': 144, 'cooldown': 15, 'category': 'Scalp'},
    'T4': {'sl': 1.0, 'tp': 2.0, 'max_hold': 144, 'cooldown': 30, 'category': 'Intraday'},
    'T5': {'sl': 1.0, 'tp': 3.0, 'max_hold': 216, 'cooldown': 30, 'category': 'Intraday'},
    'T6': {'sl': 1.0, 'tp': 4.0, 'max_hold': 288, 'cooldown': 30, 'category': 'Swing'},
    'T7': {'sl': 2.0, 'tp': 4.0, 'max_hold': 288, 'cooldown': 60, 'category': 'Swing'},
    'T8': {'sl': 2.0, 'tp': 6.0, 'max_hold': 432, 'cooldown': 60, 'category': 'Position'},
}

TARGET_NAMES = list(TARGET_CONFIGS.keys())

# Scale groups for interaction features
SCALE_GROUPS = {
    'scalp': ['T1', 'T2', 'T3'],
    'intra': ['T4', 'T5', 'T6'],
    'position': ['T7', 'T8'],
}

# ============================================================================
# CatBoost Hyperparameters
# ============================================================================
BASE_MODEL_PARAMS = dict(
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

META_MODEL_PARAMS = dict(
    depth=5,
    iterations=3000,
    learning_rate=0.0427,
    l2_leaf_reg=0.635,
    min_data_in_leaf=1700,
    loss_function='MultiClass',
    eval_metric='TotalF1:average=Macro;use_weights=false',
    verbose=500,
    task_type='GPU',
    class_weights=[0.89, 1.32, 1.32],
    early_stopping_rounds=600,
    has_time=True,
    random_strength=1.43,
    border_count=50,
    subsample=0.52,
    bootstrap_type='Bernoulli',
    random_seed=42,
)

MIN_ITERATIONS = 200
SEEDS = [42, 123, 777]
THRESHOLDS = [0.42, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
EMBARGO_DAYS = 7

# ============================================================================
# Timeline
# ============================================================================

# Base model training: everything before this date
BASE_TRAIN_CUTOFF = pd.Timestamp('2024-01-01')

# Walk-forward folds (all within pre-2024 data)
WALKFORWARD_FOLDS = [
    {
        'name': 'Fold 0',
        'train_end': pd.Timestamp('2022-01-01'),
        'val_end': pd.Timestamp('2022-02-01'),
        'embargo_end': pd.Timestamp('2022-02-08'),
        'test_end': pd.Timestamp('2022-07-01'),
    },
    {
        'name': 'Fold 1',
        'train_end': pd.Timestamp('2022-07-01'),
        'val_end': pd.Timestamp('2022-08-01'),
        'embargo_end': pd.Timestamp('2022-08-08'),
        'test_end': pd.Timestamp('2023-01-01'),
    },
    {
        'name': 'Fold 2',
        'train_end': pd.Timestamp('2023-01-01'),
        'val_end': pd.Timestamp('2023-02-01'),
        'embargo_end': pd.Timestamp('2023-02-08'),
        'test_end': pd.Timestamp('2023-07-01'),
    },
    {
        'name': 'Fold 3',
        'train_end': pd.Timestamp('2023-07-01'),
        'val_end': pd.Timestamp('2023-08-01'),
        'embargo_end': pd.Timestamp('2023-08-08'),
        'test_end': pd.Timestamp('2024-01-01'),
    },
]

# OOS prediction period (base models predict here, stacking trains here)
OOS_START = pd.Timestamp('2024-01-02')
OOS_END = pd.Timestamp('2026-03-03')

# Stacking temporal split
STACK_TRAIN_END = pd.Timestamp('2025-06-01')
STACK_VAL_END = pd.Timestamp('2025-07-01')
STACK_TEST_END = pd.Timestamp('2026-03-03')

# Final base model internal validation (for early stopping only)
BASE_FINAL_VAL_START = pd.Timestamp('2023-12-15')

# ============================================================================
# Paths
# ============================================================================
ENCODED_DIR = Path(__file__).parent / "encoded_data"
ACTUAL_DATA_DIR = Path(__file__).parent / "actual_data"
RESULTS_DIR = Path(__file__).parent / "results_v10" / "multitarget"
BASE_MODELS_DIR = RESULTS_DIR / "base_models"
LABEL_CACHE_DIR = RESULTS_DIR / "label_cache"
OOS_DIR = RESULTS_DIR / "oos_probabilities"
STACKING_DIR = RESULTS_DIR / "stacking"

CLASS_NAMES = {0: 'NO_TRADE', 1: 'LONG', 2: 'SHORT'}
TRADE_CLASSES = [1, 2]

# Number of top raw features to include in stacking input
TOP_RAW_FEATURES = 50
