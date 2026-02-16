# Fails to Avoid

## Index
| # | Pattern | Category | Date |
|---|---------|----------|------|
| F001 | Double class-weighting in CatBoost | code-bug | 2026-02-16 |
| F002 | Position-based alignment fallback | code-bug | 2026-02-16 |
| F003 | CatBoost CPU training timeout | config | 2026-02-16 |
| F004 | Governance loop breaks after ADJUST | architecture | 2026-02-16 |
| F005 | Learner never triggered (wrong trigger condition) | architecture | 2026-02-16 |
| F006 | Work unit counter phantom hook | config | 2026-02-16 |
| F007 | Memory crash on large timeframe processing | memory | 2026-02-16 |
| F008 | XGBoost predicts only majority class | code-bug | 2026-02-16 |

## F001: Double class-weighting in CatBoost
**Category**: code-bug
**Trigger**: Using `auto_class_weights='Balanced'` AND `sample_weight=compute_sample_weights()` together
**Symptom**: Model loss function is distorted, predictions biased toward minority class overcorrection
**Fix**: Remove `sample_weight` parameter. Keep `auto_class_weights='Balanced'` only.
**Prevention**: Never combine CatBoost auto-balancing with manual sample weights

## F002: Position-based alignment fallback
**Category**: code-bug
**Trigger**: No timestamp overlap between features and labels DataFrames
**Symptom**: Silently pairs mismatched rows, producing garbage feature-label combinations
**Fix**: Replace `tail`/`head` fallback with `raise ValueError("No timestamp overlap")`
**Prevention**: Always validate timestamp overlap before alignment. Never use positional fallback.

## F003: CatBoost CPU training timeout
**Category**: config
**Trigger**: Training 3267 features x 100K+ rows on CPU with 500 iterations
**Symptom**: Training takes ~47 minutes, agent times out before completion
**Fix**: Use `task_type='GPU'` in CatBoostClassifier. Add `early_stopping_rounds=50`.
**Prevention**: Always use GPU for CatBoost with >1000 features. Always add early stopping.

## F004: Governance loop breaks after ADJUST
**Category**: architecture
**Trigger**: Actor-critique returns ADJUST but orchestrator doesn't create follow-up work
**Symptom**: Bugs identified in critique are never fixed. Team shuts down with known issues.
**Fix**: Orchestrator must forward ADJUST issues directly to worker as SELF-CORRECT message.
**Prevention**: ADJUST handler in orchestrator explicitly routes fixes to worker and resumes loop.

## F005: Learner never triggered (wrong trigger condition)
**Category**: architecture
**Trigger**: Learner only activated on ROLLBACK or VOTE_NEEDED, which rarely happen
**Symptom**: Empty learnings/index.md, empty decay.json, zero value from learner agent
**Fix**: Trigger learner on ANY error from any agent, not just catastrophic events.
**Prevention**: Design trigger conditions for common events, not rare edge cases.

## F006: Work unit counter phantom hook
**Category**: config
**Trigger**: orchestrator.md references PostToolUse hook that doesn't exist
**Symptom**: Work unit counter resets to 0 or is never incremented
**Fix**: Remove hook reference. Orchestrator manually increments counter.
**Prevention**: Never reference infrastructure that hasn't been implemented.

## F007: Memory crash on large timeframe processing
**Category**: memory
**Trigger**: Parallel processing of short-interval timeframes (1H/30M/15M/5M) with multiprocessing pool
**Symptom**: Process pool crashes, memory errors during ETL feature extraction
**Fix**: Process timeframes sequentially or use chunked processing for high-row-count TFs
**Prevention**: Estimate memory per TF before choosing parallel vs sequential. Add memory guards.

## F008: XGBoost predicts only majority class
**Category**: code-bug
**Trigger**: Severe class imbalance (~83% NO_TRADE) with default XGBoost settings
**Symptom**: 83.3% accuracy but 0 trades, 0% win rate — predicts NO_TRADE for everything
**Fix**: Use CatBoost with auto_class_weights='Balanced' or custom loss function
**Prevention**: Always check confusion matrix for majority-class-only predictions after training
