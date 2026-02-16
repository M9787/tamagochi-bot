# Work Log

| # | Timestamp | Description | Status |
|---|-----------|-------------|--------|
| 1 | 2026-02-16 | Fix F001: Remove sample_weight from train_catboost (double class-weighting) | done |
| 2 | 2026-02-16 | Add task_type='GPU' to CatBoostClassifier | done |
| 3 | 2026-02-16 | Add early_stopping_rounds=50 with eval set | done |
| 4 | 2026-02-16 | Fix F002: Replace positional alignment fallback with ValueError | done |
| 5 | 2026-02-16 | End-to-end training run: ETL 285 base/3135 total features, CatBoost GPU+early_stop, results saved | done |
| 6 | 2026-02-16 | ADJUST: Added train/test label distribution logging | done |
| 7 | 2026-02-16 | ADJUST: Added base-only vs full feature diagnostic comparison | done |
| 8 | 2026-02-16 | ADJUST: Added eval loss logging (first/best/last) | done |
| 9 | 2026-02-16 | ADJUST: Ran diagnostic -- confirmed distribution shift + features lack signal | done |
