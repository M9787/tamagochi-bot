"""Binance USDT-M Futures executor — order placement, position queries, SL/TP management."""

import logging
import math
import os
import time

logger = logging.getLogger(__name__)

SYMBOL = "BTCUSDT"
QUANTITY_PRECISION = 3  # BTC futures: 0.001 minimum
PRICE_PRECISION = 1     # BTCUSDT price tick: 0.1


class BinanceFuturesExecutor:
    """Wraps python-binance Client for USDT-M Futures operations.

    Supports testnet and production via env vars:
        Testnet:    BINANCE_TESTNET_KEY, BINANCE_TESTNET_SECRET
        Production: BINANCE_KEY, BINANCE_SECRET
    """

    def __init__(self, testnet: bool = True, leverage: int = 10,
                 usdt_amount: float = 100.0, dry_run: bool = False):
        self.testnet = testnet
        self.leverage = leverage
        self.usdt_amount = usdt_amount
        self.dry_run = dry_run
        self.client = None

        if dry_run:
            logger.info("DRY RUN mode — no orders will be placed")
            return

        self._init_client()

    def _init_client(self):
        """Initialize Binance client with appropriate keys."""
        from binance import Client

        if self.testnet:
            api_key = os.environ.get("BINANCE_TESTNET_KEY", "")
            api_secret = os.environ.get("BINANCE_TESTNET_SECRET", "")
            if not api_key or not api_secret:
                raise RuntimeError(
                    "Set BINANCE_TESTNET_KEY and BINANCE_TESTNET_SECRET env vars")
        else:
            api_key = os.environ.get("BINANCE_KEY", "")
            api_secret = os.environ.get("BINANCE_SECRET", "")
            if not api_key or not api_secret:
                raise RuntimeError(
                    "Set BINANCE_KEY and BINANCE_SECRET env vars")

        self.client = Client(api_key=api_key, api_secret=api_secret,
                             testnet=self.testnet,
                             requests_params={"timeout": 10})

        # Configure leverage and margin type
        try:
            self.client.futures_change_leverage(
                symbol=SYMBOL, leverage=self.leverage)
            logger.info(f"Leverage set to {self.leverage}x on {SYMBOL}")
        except Exception as e:
            logger.warning(f"Set leverage failed (may already be set): {e}")

        try:
            self.client.futures_change_margin_type(
                symbol=SYMBOL, marginType="ISOLATED")
            logger.info(f"Margin type set to ISOLATED on {SYMBOL}")
        except Exception as e:
            # -4046 = "No need to change margin type" (already ISOLATED)
            if "-4046" in str(e):
                logger.info("Margin type already ISOLATED")
            else:
                logger.warning(f"Set margin type failed: {e}")

        mode = "TESTNET" if self.testnet else "PRODUCTION"
        logger.info(f"Binance Futures executor initialized ({mode})")

    def reconnect(self):
        """Recreate Binance client (e.g., after persistent API failures)."""
        logger.warning("Reconnecting Binance client...")
        old_client = self.client
        try:
            self._init_client()
            logger.info("Binance client reconnected")
        except Exception as e:
            logger.error(f"Reconnect failed: {e}")
            if old_client is not None:
                self.client = old_client

    def _api_retry(self, fn, *args, max_attempts=3, operation="API call", **kwargs):
        """Retry API call with exponential backoff.

        Backoff: 0s, 2s, 5s between attempts.
        Raises last exception if all attempts fail.
        """
        backoff = [0, 2, 5]
        last_err = None
        for attempt in range(max_attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                if attempt < max_attempts - 1:
                    wait = backoff[attempt + 1]
                    logger.warning(
                        f"{operation} failed (attempt {attempt+1}/{max_attempts}): {e}. "
                        f"Retry in {wait}s")
                    time.sleep(wait)
                else:
                    logger.error(f"{operation} failed after {max_attempts} attempts: {e}")
        raise last_err

    def get_position(self) -> dict:
        """Get current BTCUSDT position from exchange.

        Returns dict with: positionAmt, entryPrice, unRealizedProfit, leverage, etc.
        Returns empty-position dict in dry-run mode.
        """
        if self.dry_run:
            return {"positionAmt": "0", "entryPrice": "0", "unRealizedProfit": "0"}

        positions = self._api_retry(
            self.client.futures_position_information,
            symbol=SYMBOL, operation="GET_POSITION")
        if positions:
            return positions[0]
        return {"positionAmt": "0", "entryPrice": "0", "unRealizedProfit": "0"}

    def get_mark_price(self) -> float:
        """Get current mark price for BTCUSDT."""
        if self.dry_run:
            # Use ticker price from public API as fallback
            from binance import Client
            client = Client()
            ticker = client.futures_symbol_ticker(symbol=SYMBOL)
            return float(ticker["price"])

        ticker = self._api_retry(
            self.client.futures_mark_price,
            symbol=SYMBOL, operation="GET_MARK_PRICE")
        return float(ticker["markPrice"])

    def get_last_trade_pnl(self) -> float | None:
        """Get realized PnL from the most recent trade fill.

        Returns the PnL value, or None if lookup fails.
        """
        if self.dry_run:
            return None
        try:
            trades = self.client.futures_account_trades(symbol=SYMBOL, limit=5)
            if trades:
                # Find the last fill with non-zero realizedPnl (the close fill)
                for trade in reversed(trades):
                    pnl = float(trade.get('realizedPnl', 0))
                    if abs(pnl) > 0.001:
                        return pnl
            return 0.0
        except Exception as e:
            logger.warning(f"Failed to get last trade PnL: {e}")
            return None

    def get_account_balance(self) -> dict | None:
        """Get account balance info from Binance Futures.

        Returns: {total_balance, available_balance, unrealized_pnl} or None.
        Only called on trade close events (not every cycle).
        """
        if self.dry_run:
            return None
        try:
            account = self._api_retry(
                self.client.futures_account,
                operation="GET_ACCOUNT_BALANCE")
            return {
                "total_balance": float(account.get("totalWalletBalance", 0)),
                "available_balance": float(account.get("availableBalance", 0)),
                "unrealized_pnl": float(account.get("totalUnrealizedProfit", 0)),
            }
        except Exception as e:
            logger.warning(f"Failed to get account balance: {e}")
            return None

    def _calc_quantity(self, mark_price: float) -> float:
        """Calculate order quantity: usdt_amount * leverage / mark_price, rounded down."""
        raw = (self.usdt_amount * self.leverage) / mark_price
        # Round down to precision
        factor = 10 ** QUANTITY_PRECISION
        return math.floor(raw * factor) / factor

    def open_position(self, side: str,
                      sl_price: float = 0.0, tp_price: float = 0.0) -> dict | None:
        """Open a new position with market order and immediately place SL/TP.

        SL/TP placement happens within this call to minimize the naked position
        window (previously 2-5s gap). If SL/TP placement fails, the position is
        emergency-closed before returning.

        Args:
            side: 'BUY' (for LONG) or 'SELL' (for SHORT)
            sl_price: stop-loss price (required for non-dry-run)
            tp_price: take-profit price (required for non-dry-run)

        Returns: dict with price, quantity, order_id, sl_tp_ok — or None on failure
        """
        mark_price = self.get_mark_price()
        quantity = self._calc_quantity(mark_price)

        if quantity <= 0:
            logger.error(f"Calculated quantity is 0 (mark={mark_price}, "
                         f"amount={self.usdt_amount}, lev={self.leverage})")
            return None

        logger.info(f"Opening {side} {quantity} {SYMBOL} @ ~{mark_price:.2f} "
                     f"(${self.usdt_amount} x {self.leverage}x)")

        if self.dry_run:
            return {"price": mark_price, "quantity": quantity, "order_id": "DRY_RUN",
                    "sl_tp_ok": True}

        try:
            order = self._api_retry(
                self.client.futures_create_order,
                symbol=SYMBOL, side=side, type="MARKET", quantity=quantity,
                operation=f"OPEN {side}")

            order_id = order["orderId"]
            fill_price = float(order.get("avgPrice", mark_price))
            filled_qty = float(order.get("executedQty", quantity))

            if fill_price == 0:
                fill_price = mark_price

            if filled_qty < 1e-8:
                logger.error(f"Order {order_id} filled 0 quantity — treating as failure")
                return None

            if filled_qty < quantity * 0.99:
                logger.warning(f"PARTIAL FILL: requested={quantity}, filled={filled_qty}")

            logger.info(f"Market order filled: {order_id} @ {fill_price:.2f} qty={filled_qty}")

            # Immediately place SL/TP to minimize naked position window
            sl_tp_ok = True
            if sl_price > 0 and tp_price > 0:
                sl_tp_ok = self._place_sl_tp(side, sl_price, tp_price)
                # _place_sl_tp already handles emergency close on failure

            return {"price": fill_price, "quantity": filled_qty, "order_id": order_id,
                    "sl_tp_ok": sl_tp_ok}

        except Exception as e:
            logger.error(f"Open position failed: {e}")
            return None

    def add_to_position(self, side: str) -> dict | None:
        """Add to existing position with market order.

        Market order placed FIRST — existing SL/TP remain intact if this fails.
        Caller must call update_sl_tp() after fill with new avg entry.

        Returns: dict with price, quantity, order_id — or None on failure
        """
        mark_price = self.get_mark_price()
        quantity = self._calc_quantity(mark_price)

        if quantity <= 0:
            logger.error("Calculated quantity is 0")
            return None

        logger.info(f"Adding {side} {quantity} {SYMBOL} @ ~{mark_price:.2f}")

        if self.dry_run:
            return {"price": mark_price, "quantity": quantity, "order_id": "DRY_RUN"}

        try:
            # Market order FIRST — existing SL/TP stay intact if this fails
            order = self._api_retry(
                self.client.futures_create_order,
                symbol=SYMBOL, side=side, type="MARKET", quantity=quantity,
                operation=f"ADD {side}")

            order_id = order["orderId"]
            fill_price = float(order.get("avgPrice", mark_price))
            filled_qty = float(order.get("executedQty", quantity))

            if fill_price == 0:
                fill_price = mark_price

            if filled_qty < 1e-8:
                logger.error(f"Order {order_id} filled 0 quantity — treating as failure")
                return None

            if filled_qty < quantity * 0.99:
                logger.warning(f"PARTIAL FILL: requested={quantity}, filled={filled_qty}")

            logger.info(f"Add-to-position filled: {order_id} @ {fill_price:.2f} qty={filled_qty}")
            return {"price": fill_price, "quantity": filled_qty, "order_id": order_id}

        except Exception as e:
            logger.error(f"Add to position failed: {e}")
            return None

    def update_sl_tp(self, side: str, sl_price: float, tp_price: float) -> bool:
        """Cancel existing SL/TP and place new ones at updated levels.

        Args:
            side: 'LONG' or 'SHORT' (position direction)
            sl_price: new stop-loss price
            tp_price: new take-profit price

        Returns: True if both SL and TP placed successfully
        """
        if self.dry_run:
            logger.info(f"[DRY] Update SL/TP: SL={sl_price:.2f} TP={tp_price:.2f}")
            return True

        order_side = "BUY" if side == "LONG" else "SELL"
        self.cancel_all_orders()
        return self._place_sl_tp(order_side, sl_price, tp_price)

    def _place_sl_tp(self, entry_side: str, sl_price: float, tp_price: float) -> bool:
        """Place SL and TP orders with retry. Closes position if placement fails.

        Args:
            entry_side: 'BUY' or 'SELL' (the entry side — SL/TP use opposite)
            sl_price: stop-loss trigger price
            tp_price: take-profit trigger price

        Returns: True if both placed successfully
        """
        close_side = "SELL" if entry_side == "BUY" else "BUY"
        sl_rounded = round(sl_price, PRICE_PRECISION)
        tp_rounded = round(tp_price, PRICE_PRECISION)

        # SL with retry (3 attempts)
        sl_ok = False
        for attempt in range(3):
            try:
                self.client.futures_create_order(
                    symbol=SYMBOL,
                    side=close_side,
                    type="STOP_MARKET",
                    stopPrice=str(sl_rounded),
                    closePosition="true",
                )
                logger.info(f"SL placed: {close_side} STOP_MARKET @ {sl_rounded}")
                sl_ok = True
                break
            except Exception as e:
                logger.warning(f"SL attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(1)

        # TP with retry (3 attempts)
        tp_ok = False
        for attempt in range(3):
            try:
                self.client.futures_create_order(
                    symbol=SYMBOL,
                    side=close_side,
                    type="TAKE_PROFIT_MARKET",
                    stopPrice=str(tp_rounded),
                    closePosition="true",
                )
                logger.info(f"TP placed: {close_side} TAKE_PROFIT_MARKET @ {tp_rounded}")
                tp_ok = True
                break
            except Exception as e:
                logger.warning(f"TP attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(1)

        if not sl_ok or not tp_ok:
            logger.critical(
                f"SL/TP FAILED after retries (SL={'OK' if sl_ok else 'FAIL'}, "
                f"TP={'OK' if tp_ok else 'FAIL'}). CLOSING POSITION for safety.")
            result = self.close_position()
            if result is None:
                self.emergency_close()
            return False
        return True

    def close_position(self, local_qty: float = 0.0,
                       local_side: str | None = None) -> dict | None:
        """Close the entire current position.

        Cancels all open orders, then market-closes the position.

        Args:
            local_qty: Local position quantity (used in dry-run where exchange returns 0).
            local_side: Local position side 'LONG'/'SHORT' (used in dry-run).

        Returns: dict with order details or None.
        """
        # In dry-run, exchange always reports 0 — use local state instead
        if self.dry_run and local_qty > 0:
            close_side = "SELL" if local_side == "LONG" else "BUY"
            logger.info(f"[DRY] Closing position: {close_side} {local_qty} {SYMBOL}")
            mark = self.get_mark_price()
            return {"price": mark, "quantity": local_qty, "order_id": "DRY_RUN_CLOSE"}

        pos = self.get_position()
        amt = float(pos.get("positionAmt", 0))

        if abs(amt) < 1e-8:
            logger.info("No position to close")
            return None

        close_side = "SELL" if amt > 0 else "BUY"
        close_qty = abs(amt)

        logger.info(f"Closing position: {close_side} {close_qty} {SYMBOL}")

        try:
            self.cancel_all_orders()

            order = self._api_retry(
                self.client.futures_create_order,
                symbol=SYMBOL, side=close_side, type="MARKET", quantity=close_qty,
                operation="CLOSE_POSITION")

            order_id = order["orderId"]
            fill_price = float(order.get("avgPrice", 0))
            logger.info(f"Position closed: {order_id} @ {fill_price:.2f}")
            return {"price": fill_price, "quantity": close_qty, "order_id": order_id}

        except Exception as e:
            logger.error(f"Close position failed: {e}")
            return None

    def cancel_all_orders(self):
        """Cancel all open orders for BTCUSDT."""
        if self.dry_run:
            logger.info("[DRY] Cancel all open orders")
            return

        try:
            self._api_retry(
                self.client.futures_cancel_all_open_orders,
                symbol=SYMBOL, operation="CANCEL_ALL")
            logger.info("All open orders cancelled")
        except Exception as e:
            logger.warning(f"Cancel orders failed: {e}")

    def get_open_orders(self) -> list:
        """Get all open orders for BTCUSDT."""
        if self.dry_run:
            return []
        return self._api_retry(
            self.client.futures_get_open_orders,
            symbol=SYMBOL, operation="GET_OPEN_ORDERS")

    def emergency_close(self, max_attempts=30, interval=10) -> bool:
        """EMERGENCY: Aggressive retry loop for naked position.

        Blocks ALL other operations until position is closed or max_attempts
        exhausted. Each close_position() call internally retries 3x,
        so worst case = max_attempts * (~37s retry + 10s sleep).

        Checks global _shutdown_requested to allow graceful Docker stop.

        Returns True if position confirmed closed.
        """
        import trading_bot

        for attempt in range(max_attempts):
            if hasattr(trading_bot, '_shutdown_requested') and trading_bot._shutdown_requested:
                logger.critical("EMERGENCY CLOSE interrupted by SIGTERM")
                return False

            logger.critical(
                f"EMERGENCY CLOSE attempt {attempt+1}/{max_attempts} — "
                f"naked position on exchange!")

            try:
                result = self.close_position()
                if result is not None:
                    logger.critical("EMERGENCY CLOSE succeeded")
                    return True
            except Exception as e:
                logger.critical(f"Emergency close_position error: {e}")

            # Check if exchange is already flat (SL/TP may have triggered)
            try:
                pos = self.get_position()
                if abs(float(pos.get("positionAmt", 0))) < 1e-8:
                    logger.critical(
                        "EMERGENCY: Exchange reports FLAT — "
                        "position may have closed via SL/TP")
                    return True
            except Exception:
                pass

            if attempt < max_attempts - 1:
                time.sleep(interval)

        logger.critical(
            f"EMERGENCY CLOSE FAILED after {max_attempts} attempts — "
            f"MANUAL INTERVENTION REQUIRED")
        return False

    def verify_sl_tp_orders(self) -> dict:
        """Verify SL and TP orders exist on exchange.

        Returns: {'has_sl': bool, 'has_tp': bool,
                  'sl_price': float, 'tp_price': float, 'order_count': int}
        Returns all-True in dry_run mode or on API failure (defensive).
        """
        if self.dry_run:
            return {'has_sl': True, 'has_tp': True,
                    'sl_price': 0, 'tp_price': 0, 'order_count': 0}

        try:
            orders = self.get_open_orders()
        except Exception as e:
            logger.warning(f"verify_sl_tp_orders failed to query: {e}")
            return {'has_sl': True, 'has_tp': True,
                    'sl_price': 0, 'tp_price': 0, 'order_count': -1}

        has_sl = False
        has_tp = False
        sl_price = 0.0
        tp_price = 0.0

        for o in orders:
            if o.get('type') == 'STOP_MARKET':
                has_sl = True
                sl_price = float(o.get('stopPrice', 0))
            elif o.get('type') == 'TAKE_PROFIT_MARKET':
                has_tp = True
                tp_price = float(o.get('stopPrice', 0))

        return {
            'has_sl': has_sl, 'has_tp': has_tp,
            'sl_price': sl_price, 'tp_price': tp_price,
            'order_count': len(orders),
        }
