# Critique Log

Reviews by actor-critique agent.

## Review #1 - 2026-02-16
Work Units: 1-10
Points: +3
Delta: 17% (2/12 PASS)
Quality: GOOD
Decision: ADJUST

### Reasoning
ETL rewrite (M1-M5 matrices) and CatBoost integration are structurally sound. Code quality is high with clear naming, proper docstrings, and correct matrix dimensions (55+77+55+55+55=297 base, 3267 with lags). GMM zone computation is properly isolated per-TF.

### Issues
1. **Double class-weighting bug** (train.py:95-107): `auto_class_weights='Balanced'` AND manual `sample_weight=compute_sample_weights()` are both applied. CatBoost will apply balanced weights internally, then the external sample_weight doubles the effect. Fix: remove either `auto_class_weights` or `sample_weight`, not both.
2. **Risky fallback alignment** (etl.py:506-509): When no timestamp overlap exists, position-based alignment (`tail`/`head`) silently pairs mismatched rows. Should raise ValueError instead of silently proceeding.
3. **10/12 convergence criteria FAIL**: Most failures are because end-to-end run hasn't been attempted yet (Task #10 pending). Code logic appears correct but is unvalidated.

### Adjustment Required
- Remove `sample_weight` parameter from `model.fit()` in train.py:107 (keep `auto_class_weights='Balanced'` which is the cleaner approach)
- Replace position-based fallback in etl.py:506-509 with `raise ValueError("No timestamp overlap")`
- Then proceed to Task #10 (end-to-end validation)
