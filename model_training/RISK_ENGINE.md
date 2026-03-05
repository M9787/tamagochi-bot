# Risk Engine & Network Resilience — Implementation Reference

> Technical specification of all risk management and network resilience features in the V10 trading bot. Implemented 2026-03-04.

## Overview

The risk engine operates at three layers:

1. **Network Layer** — API timeout, retry with backoff, client reconnection, circuit breaker
2. **Position Layer** — SL/TP verification, emergency close, partial fill handling, max hold timer
3. **Safety Layer** — Rolling win rate, consecutive loss limits, profit lock trailing stop

All features are production-ready and active on Binance Futures testnet.

---

## 1. Network Layer

### 1.1 API Timeout

**File**: `trading/executor.py:54-56`

```python
self.client = Client(..., requests_params={"timeout": 10})
```

- 10-second connect+read timeout on every Binance API call
- Prevents indefinite blocking on degraded API or network issues
- Worst case per operation: 10s timeout × 3 retries + backoff = ~37s max

### 1.2 API Retry (`_api_retry`)

**File**: `trading/executor.py:92-113`

Generic retry wrapper applied to all 7 API-calling methods:

| Method | Operation Name | Notes |
|--------|---------------|-------|
| `get_position()` | GET_POSITION | Position info query |
| `get_mark_price()` | GET_MARK_PRICE | Price query |
| `open_position()` | OPEN BUY/SELL | Market order + partial fill handling |
| `add_to_position()` | ADD BUY/SELL | Same as open |
| `close_position()` | CLOSE_POSITION | Market close |
| `cancel_all_orders()` | CANCEL_ALL | Cancel SL/TP before update |
| `get_open_orders()` | GET_OPEN_ORDERS | For SL/TP verification |

**Backoff schedule**: Attempt 1 (immediate) → Attempt 2 (wait 2s) → Attempt 3 (wait 5s)

**Contract**: Raises last exception after all retries exhausted. Callers' existing `try/except` blocks catch this and return `None` — preserving the original API contract.

### 1.3 Client Reconnection

**File**: `trading/executor.py:80-90`

```python
def reconnect(self):
```

- Recreates the Binance Client object (new TCP connections, new auth)
- Preserves old client on failure (defensive — restores if `_init_client()` throws)
- Triggered automatically by circuit breaker at exactly 3 consecutive errors
- Can be called manually via `executor.reconnect()`

### 1.4 Circuit Breaker

**File**: `trading_bot.py:511-522, 646-657`

**Trigger**: 3+ consecutive cycle errors (any unhandled exception in main loop)

**Behavior**:
- Normal interval: `base_interval` (default 300s)
- After 3 errors: `min(base_interval * 2^(errors-2), 600)` seconds
- At exactly 3 errors: triggers `executor.reconnect()`
- Max backoff: 600s (10 minutes)
- Resets to 0 on first successful cycle

**Backoff progression** (base=300s):

| Consecutive Errors | Sleep Interval | Notes |
|-------------------|----------------|-------|
| 0-2 | 300s | Normal |
| 3 | 300s | + reconnect triggered |
| 4 | 600s | 300 × 2^2 = 1200, capped at 600 |
| 5+ | 600s | Stays at cap |

---

## 2. Position Layer

### 2.1 SL/TP Verification

**File**: `trading/executor.py:470-505` (verify), `trading_bot.py:459-487` (startup), `trading_bot.py:532-545` (resync)

**Two verification points**:

1. **On startup**: After loading state + exchange sync, verifies that STOP_MARKET and TAKE_PROFIT_MARKET orders exist on exchange. Re-places if missing.

2. **After RESYNC**: When exchange has a position that local state didn't know about, verifies orders after syncing.

**`verify_sl_tp_orders()` returns**:
```python
{'has_sl': bool, 'has_tp': bool,
 'sl_price': float, 'tp_price': float, 'order_count': int}
```

**Failure mode**: On API failure, returns `has_sl=True, has_tp=True` with `order_count=-1`. This prevents false re-placement that could cancel existing orders. The sentinel `-1` lets callers detect the failure.

**Re-placement flow**:
1. Calls `executor.update_sl_tp()` with position_mgr's SL/TP prices
2. `update_sl_tp()` cancels all orders first, then places new SL + TP
3. If re-placement fails → `position_mgr.reset()` (state cleared, emergency close handles exchange)

### 2.2 Emergency Close

**File**: `trading/executor.py:421-468`

Aggressive blocking retry loop for naked positions (position open but no SL/TP protection).

**Parameters**:
- `max_attempts=30`
- `interval=10` seconds between attempts

**Each attempt**:
1. Check `_shutdown_requested` flag (allows SIGTERM to interrupt)
2. Call `close_position()` (which internally retries 3x)
3. If close returns None, check if exchange is already flat (SL/TP may have triggered)
4. Sleep 10s and retry

**Worst case**: 30 attempts × (~37s retry + 10s sleep) ≈ 23 minutes

**Trigger points**:
- `_place_sl_tp()` fails → `close_position()` fails → `emergency_close()`
- Startup SL/TP re-placement fails (via trading_bot.py, position_mgr.reset handles it)

**SIGTERM interaction**: Checks `trading_bot._shutdown_requested` each iteration. Docker's `stop_grace_period: 60s` gives ~6 emergency close attempts before SIGKILL.

### 2.3 Partial Fill Handling

**File**: `trading/executor.py:196-217` (open), `trading/executor.py:243-265` (add)

Both `open_position()` and `add_to_position()` now:

1. Read `executedQty` from order response (not just requested quantity)
2. **Zero fill guard**: If `filled_qty < 1e-8`, returns `None` (prevents division-by-zero in `add_entry()`)
3. **Partial fill warning**: If `filled_qty < requested * 0.99`, logs `PARTIAL FILL` warning
4. Returns actual `filled_qty` in result dict (not requested quantity)

This ensures `position_manager.add_entry()` receives the real filled quantity for accurate average entry calculation.

### 2.4 Max Hold Aggressive Cycle

**File**: `trading/position_manager.py:110-115` (helper), `trading_bot.py:563-567` (logic)

**`remaining_hold_seconds()`**: Returns seconds until max_hold expiry (24h default), or `None` if no position.

**Aggressive cycle**: When `remaining < 1800` (30 minutes):
- `sleep_interval = min(sleep_interval, 60)` — cycle every 60s instead of 300s
- Ensures the bot can react quickly to max_hold expiry
- Logs: `Max hold approaching (Ns left) — 60s cycle`

---

## 3. Safety Layer

### 3.1 Safety Monitor

**File**: `trading/safety.py`

| Check | Threshold | Behavior |
|-------|-----------|----------|
| Rolling win rate | < 33.3% (over last 20 trades) | Pauses trading |
| Consecutive losses | >= 5 | Pauses trading |
| Pause cooldown | 3600s (1 hour) | Auto-resumes |

**Cooldown grace** (`cooldown_grace` flag): After a pause expires, the same losing trades would immediately re-trigger the pause. The grace flag skips re-evaluation until a new trade arrives via `record_trade()`.

**Persistence**: Trade history, `paused_since`, and `cooldown_grace` are serialized in `trading_state/state.json` via `to_list()`/`load_from_list()`.

### 3.2 Profit Lock

**File**: `trading/position_manager.py:110-138`

Trailing stop on profitable positions:

1. Track `high_water_mark_pct` (best unrealized PnL since entry)
2. When HWM >= 3.5% → activate profit lock
3. When current PnL drops to <= 3.0% → force close

Prevents giving back large gains that haven't hit TP.

---

## 4. Interaction Diagram

```
Normal cycle:
  sleep(interval) → sync_exchange → max_hold_check → profit_lock → predict → safety → trade
                                                                                         │
  On error: ─────────────────────────────────── consecutive_errors++ ─── circuit breaker ─┘
  On success: ─────────────────────────────────  consecutive_errors=0 ────────────────────┘

Position opening:
  open_position (3x retry) → fill → add_entry(filled_qty) → update_sl_tp
                                                                    │
                                          SL/TP fail → close_position (3x retry)
                                                              │
                                                    close fail → emergency_close (30x)

Startup:
  load_state → sync_exchange → verify_sl_tp → [re-place if missing] → main loop
                    │
              RESYNCED → verify_sl_tp → [re-place if missing]
```

---

## 5. Configuration Quick Reference

| Parameter | Value | Can Change? | Location |
|-----------|-------|-------------|----------|
| API timeout | 10s | Yes | `executor.py:56` |
| Retry attempts | 3 | Yes | `_api_retry` default |
| Retry backoff | [0, 2, 5]s | Yes | `executor.py:98` |
| Emergency close attempts | 30 | Yes | `executor.py:421` |
| Emergency close interval | 10s | Yes | `executor.py:421` |
| Circuit breaker trigger | 3 errors | Yes | `trading_bot.py:521` |
| Circuit breaker max | 600s | Yes | `trading_bot.py:522` |
| Aggressive cycle threshold | 1800s (30min) | Yes | `trading_bot.py:565` |
| Aggressive cycle interval | 60s | Yes | `trading_bot.py:566` |
| Stop grace period | 60s | Yes | `docker-compose.yml:22` |

All values are constants/defaults — changing requires code edit + rebuild. No runtime configuration (intentional for safety).
