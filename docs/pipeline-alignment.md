# Data Pipeline Alignment (Audit: 2026-03-05)

## Source of Truth

**Backtest Dashboard** (via `backfill_predictions.py`) is the verified reference. All other paths must produce equivalent outputs.

## Three Encoding Paths

| Path | Entry Point | Data Source | ETL | Encoder | Context Window |
|------|-------------|-------------|-----|---------|----------------|
| **A: Backfill** | `backfill_predictions.py` | Fresh Binance download | `run_live_etl()` (batch) | `encode_live_features()` (batch) | Full download (~1400 bars/TF) |
| **B: Data Service Batch** | `layers.py:_update_predictions_batch()` | Persistent CSVs | `run_incremental_etl()` | `encode_live_features()` (batch) | `tail(BOOTSTRAP_BARS=1400)` |
| **C: Data Service Incremental** | `layers.py:_update_predictions_incremental()` | Persistent CSVs + state | `run_incremental_etl()` | `IncrementalEncoder.compute_row()` | Stateful (no window, persisted EMA/buffers) |

Path A and B call the **same** `encode_live_features()` function from `model_training/live_predict.py`. Path C is a stateful re-implementation that produces identical feature names (518) but computes them incrementally from persisted state.

## Alignment Status (Verified)

| Parameter | Backfill | Data Service | Trading Bot | Telegram | Status |
|-----------|----------|-------------|-------------|----------|--------|
| SL/TP | 2.0% / 4.0% | N/A (no sim) | 2.0% / 4.0% | Display only | ALIGNED |
| Features | 518 | 518 | N/A (reads CSV) | N/A | ALIGNED |
| Models | 3 seeds (42/123/777) | 3 seeds | N/A | N/A | ALIGNED |
| Max Hold | 288 candles | N/A | 86400s (=288 candles) | N/A | ALIGNED |
| BOOTSTRAP_BARS | 1400 | 1400 | N/A | N/A | ALIGNED |

## Threshold Architecture (Intentionally Dynamic)

```
Data Service -> writes predictions.csv with prob_no_trade/prob_long/prob_short + signal at 0.75
Trading Bot  -> re-derives signal from raw probabilities at $TRADING_THRESHOLD (default 0.70)
Telegram     -> per-subscriber threshold (default $TRADING_THRESHOLD or 0.70), re-derives from probabilities
Dashboard    -> slider-adjustable threshold, re-derives from probabilities
```

The data service's `signal` column is informational. All consumers re-derive signals from probability columns using the **canonical algorithm** (see CLAUDE.md).

## Known Discrepancies (Accepted or Fixed)

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | **ATR init**: batch uses `mean(tr[:14])`, incremental uses `h-lo` | LOW | Accept |
| 2 | **EMA warm-up**: batch `ewm(min_periods=1)` vs incremental manual update | LOW | Accept |
| 3 | ~~Incomplete candle in backfill~~ | ~~LOW~~ | **FIXED** |
| 4 | ~~Leverage/amount only via CLI args~~ | ~~N/A~~ | **FIXED** |
| 5 | ~~Telegram push: wrong signal derivation~~ | ~~MEDIUM~~ | **FIXED** |
| 6 | ~~`/status` shows data service threshold~~ | ~~LOW~~ | **FIXED** |
| 7 | ~~Trade log append not atomic~~ | ~~LOW~~ | **FIXED** |
| 8 | ~~Telegram DEFAULT_THRESHOLD hardcoded~~ | ~~LOW~~ | **FIXED** |

## Pipeline Invariants (Must Hold)

1. **All paths produce 518 features** with identical names (verified)
2. **SL=2%, TP=4%, max_hold=288** in all training, labeling, and trading code
3. **BOOTSTRAP_BARS=1400** across `live_predict.py:BARS_PER_TF` and `gap_detector.py`
4. **Batch encoder = single function** (`encode_live_features` in `live_predict.py`) shared by Path A and B
5. **Incremental state initialized from full-history feature matrix** -- never from scratch
6. **Threshold is per-consumer** -- data service writes probabilities, consumers re-derive

## Incremental Pipeline (L3 Continuous)

The data service runs L3 in two modes:

**Incremental mode** (default, steady-state):
- Loads `feature_state.json` on startup (with 3-backup corruption recovery)
- Each 5M cycle: `IncrementalEncoder.compute_row()` -> append to `features.csv` -> save state
- Gap recovery: walks through missed 5M candles sequentially, feeding TF data only when native-TF index changes (searchsorted + index tracking)
- Max backfill: 2016 candles (7 days). Larger gaps fall back to batch mode.

**Batch mode** (fallback):
- Triggered when: no state file, gap > 2016 candles, or state corruption
- Calls `encode_live_features()` on tail(1400) of persistent CSVs
- After batch, re-initializes incremental state from the batch output
