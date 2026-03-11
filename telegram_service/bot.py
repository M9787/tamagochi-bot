"""Telegram monitoring bot — commands, push notifications, polling jobs."""

import logging
from datetime import datetime, timezone

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from core.config import STALENESS_THRESHOLD_SEC
from telegram_service import formatters, readers
from telegram_service.subscribers import SubscriberStore

logger = logging.getLogger(__name__)


class TelegramMonitorBot:
    """Push + interactive Telegram bot for monitoring the trading system."""

    def __init__(self, token: str, poll_interval: int = 60,
                 subscriber_path: str = "/app/telegram_data/subscribers.json"):
        self.token = token
        self.poll_interval = poll_interval
        self.subscribers = SubscriberStore(path=subscriber_path)

        # Watermarks for change detection (set by _initialize_tracking_state)
        self._last_prediction_time: str | None = None
        self._last_trade_count: int = 0

    def _initialize_tracking_state(self):
        """Set watermarks to current data so old signals are not re-sent on restart."""
        pred = readers.read_latest_prediction()
        if pred:
            self._last_prediction_time = pred.get("time")
            logger.info(f"Tracking init: last prediction at {self._last_prediction_time}")

        trades = readers.read_recent_trades(n_days=1)
        if trades is not None:
            self._last_trade_count = len(trades)
            logger.info(f"Tracking init: {self._last_trade_count} trades today")

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Subscribe to notifications."""
        chat_id = update.effective_chat.id
        is_new = self.subscribers.add(chat_id)
        if is_new:
            threshold = self.subscribers.get_threshold(chat_id)
            await update.message.reply_text(
                "\u2705 Subscribed to Tamagochi trading alerts!\n\n"
                "You'll receive:\n"
                "- Signal alerts (LONG/SHORT predictions)\n"
                "- Trade events (opened, closed, SL/TP)\n"
                "- Hourly dashboard reports\n\n"
                f"Alert threshold: {threshold}\n"
                "Use /threshold <0.4-0.9> to customize.\n\n"
                "Commands: /status /stats /position /balance /pnl "
                "/equity /trades /health /threshold /stop /help"
            )
        else:
            await update.message.reply_text("Already subscribed. Use /help for commands.")

    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Unsubscribe from notifications."""
        chat_id = update.effective_chat.id
        removed = self.subscribers.remove(chat_id)
        if removed:
            await update.message.reply_text("\U0001f6d1 Unsubscribed. Send /start to re-subscribe.")
        else:
            await update.message.reply_text("You weren't subscribed.")

    async def cmd_threshold(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set personal alert threshold for signal notifications."""
        chat_id = update.effective_chat.id

        # Parse argument
        if not context.args:
            current = self.subscribers.get_threshold(chat_id)
            await update.message.reply_text(
                f"\U0001f3af Your alert threshold: <b>{current}</b>\n\n"
                f"Usage: /threshold 0.50\n"
                f"Range: 0.40 - 0.90\n\n"
                f"Lower = more alerts (less precise)\n"
                f"Higher = fewer alerts (more precise)",
                parse_mode=ParseMode.HTML,
            )
            return

        try:
            value = float(context.args[0])
        except (ValueError, IndexError):
            await update.message.reply_text(
                "\u274c Invalid value. Usage: /threshold 0.50")
            return

        if value < 0.40 or value > 0.90:
            await update.message.reply_text(
                "\u274c Threshold must be between 0.40 and 0.90")
            return

        value = round(value, 2)
        ok = self.subscribers.set_threshold(chat_id, value)
        if ok:
            await update.message.reply_text(
                f"\u2705 Alert threshold set to <b>{value}</b>\n"
                f"You'll get alerts when confidence \u2265 {value}",
                parse_mode=ParseMode.HTML,
            )
        else:
            await update.message.reply_text(
                "You're not subscribed. Use /start first.")

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Status with all probabilities, signal at subscriber's threshold."""
        pred = readers.read_latest_prediction()
        btc = readers.read_latest_btc_price()
        state = readers.read_trading_state()
        position = state.get("position") if state else None

        # Re-derive signal at subscriber's personal threshold
        if pred:
            chat_id = update.effective_chat.id
            user_threshold = self.subscribers.get_threshold(chat_id)
            prob_nt = pred.get("prob_no_trade", 0)
            prob_l = pred.get("prob_long", 0)
            prob_s = pred.get("prob_short", 0)
            probs = [prob_nt, prob_l, prob_s]
            pc = int(max(range(3), key=lambda i: probs[i]))
            if pc in (1, 2) and probs[pc] >= user_threshold:
                pred["signal"] = "LONG" if pc == 1 else "SHORT"
            else:
                pred["signal"] = "NO_TRADE"
            pred["confidence"] = probs[pc]

        text = formatters.fmt_status_oneliner(pred, btc, position, state=state)
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Full hourly report + probability chart."""
        # Send text report
        text = self._build_hourly_report()
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)

        # Send probability chart (all available data)
        predictions_all = readers.read_predictions(last_n_hours=0)
        chart_buf = formatters.generate_probability_chart(predictions_all)
        if chart_buf:
            span_label = ""
            if predictions_all is not None and not predictions_all.empty:
                span_h = (predictions_all["time"].max() - predictions_all["time"].min()).total_seconds() / 3600
                span_label = f" ({span_h:.0f}h)" if span_h <= 48 else f" ({span_h / 24:.0f}d)"
            await update.message.reply_photo(
                photo=chart_buf,
                caption=f"Probability Timeline{span_label}")
        else:
            await update.message.reply_text(
                "\U0001f4c9 Chart: Not enough prediction data yet")

    async def cmd_position(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Detailed position with PnL estimate."""
        state = readers.read_trading_state()
        position = state.get("position") if state else None
        btc = readers.read_latest_btc_price()

        text = formatters.fmt_position_detail(position, btc, state=state)
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async def cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Last 10 trades."""
        trades = readers.read_last_n_trades(10)
        text = formatters.fmt_trades_list(trades)
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Account balance and cumulative PnL."""
        account = readers.read_account_summary()
        state = readers.read_trading_state()
        position = state.get("position") if state else None
        btc = readers.read_latest_btc_price()

        text = formatters.fmt_balance(account, position, btc, state=state)
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async def cmd_pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """PnL summary across time periods."""
        summary = readers.compute_pnl_summary(n_days=30)
        text = formatters.fmt_pnl_summary(summary)
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async def cmd_equity(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """ASCII equity curve of recent trades."""
        trades = readers.read_trades_with_pnl(n_days=30)
        text = formatters.fmt_equity_curve(trades)
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async def cmd_health(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """System health check."""
        data_status = readers.read_data_service_status()
        state = readers.read_trading_state()

        text = formatters.fmt_health(data_status, state)
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Signal history with paginated inline keyboard."""
        df = readers.read_signal_history(n=100)
        if df is None or df.empty:
            await update.message.reply_text(
                "\U0001f4dc No signal history available.\n"
                "Signals will appear once the data service produces LONG/SHORT predictions.",
                parse_mode=ParseMode.HTML)
            return

        summary = formatters.compute_history_summary(df)
        text = formatters.fmt_history_page(df, 0, 10, summary)
        total_pages = max(1, (len(df) + 9) // 10)
        keyboard = self._history_keyboard(0, total_pages)
        await update.message.reply_text(
            text, reply_markup=keyboard, parse_mode=ParseMode.HTML)

    async def history_callback(self, update: Update,
                               context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard page navigation for /history."""
        query = update.callback_query
        await query.answer()

        data = query.data
        if data == "hist_noop":
            return

        try:
            page = int(data.split("_")[1])
        except (IndexError, ValueError):
            return

        df = readers.read_signal_history(n=100)
        if df is None or df.empty:
            await query.edit_message_text("No data available.")
            return

        summary = formatters.compute_history_summary(df)
        total_pages = max(1, (len(df) + 9) // 10)
        page = max(0, min(page, total_pages - 1))
        text = formatters.fmt_history_page(df, page, 10, summary)
        keyboard = self._history_keyboard(page, total_pages)
        await query.edit_message_text(
            text, reply_markup=keyboard, parse_mode=ParseMode.HTML)

    @staticmethod
    def _history_keyboard(page: int, total_pages: int) -> InlineKeyboardMarkup:
        """Build Prev/Page/Next inline keyboard for history pagination."""
        buttons = []
        if page > 0:
            buttons.append(InlineKeyboardButton(
                "\u25c0 Prev", callback_data=f"hist_{page - 1}"))
        buttons.append(InlineKeyboardButton(
            f"{page + 1}/{total_pages}", callback_data="hist_noop"))
        if page < total_pages - 1:
            buttons.append(InlineKeyboardButton(
                "Next \u25b6", callback_data=f"hist_{page + 1}"))
        return InlineKeyboardMarkup([buttons])

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List all commands."""
        text = (
            "\U0001f916 <b>Tamagochi Trading Bot</b>\n\n"
            "/status    \u2014 Status with probabilities\n"
            "/stats     \u2014 Dashboard + 24h probability chart\n"
            "/position  \u2014 Detailed position info with PnL\n"
            "/balance   \u2014 Account balance and cumulative PnL\n"
            "/pnl       \u2014 PnL summary (today/7d/30d/all-time)\n"
            "/equity    \u2014 Equity curve (last 20 trades)\n"
            "/trades    \u2014 Last 10 trades\n"
            "/history   \u2014 Signal history with stats (last 100)\n"
            "/health    \u2014 System health check\n"
            "/threshold \u2014 Set alert threshold (0.4-0.9)\n"
            "/start     \u2014 Subscribe to push notifications\n"
            "/stop      \u2014 Unsubscribe\n"
            "/help      \u2014 This message"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    # ------------------------------------------------------------------
    # Push notification broadcasting
    # ------------------------------------------------------------------

    async def _broadcast(self, app: Application, text: str,
                         parse_mode: str = ParseMode.HTML):
        """Send a message to all subscribers."""
        chat_ids = self.subscribers.get_all()
        if not chat_ids:
            return
        for chat_id in chat_ids:
            try:
                await app.bot.send_message(
                    chat_id=chat_id, text=text, parse_mode=parse_mode)
            except Exception as e:
                logger.warning(f"Failed to send to {chat_id}: {e}")

    # ------------------------------------------------------------------
    # Polling jobs
    # ------------------------------------------------------------------

    async def poll_changes_job(self, context: ContextTypes.DEFAULT_TYPE):
        """Poll for new predictions and trades (runs every poll_interval seconds)."""
        # Check for new prediction signal
        pred = readers.read_latest_prediction()
        if pred and pred.get("time") != self._last_prediction_time:
            self._last_prediction_time = pred.get("time")

            # Staleness check: skip signal alerts for stale predictions
            # but do NOT skip trade events (checked below, independent of predictions)
            age_sec = pred.get("age_seconds", 0)
            is_stale = age_sec > STALENESS_THRESHOLD_SEC
            if is_stale:
                logger.warning(
                    f"Stale prediction ({age_sec:.0f}s old), skipping signal alert")

            if not is_stale:
                # Per-subscriber threshold filtering
                prob_nt = pred.get("prob_no_trade", 0)
                prob_long = pred.get("prob_long", 0)
                prob_short = pred.get("prob_short", 0)

                # Derive signal using same algorithm as trading bot:
                # argmax first, then threshold check
                probs = [prob_nt, prob_long, prob_short]
                pred_class = int(max(range(3), key=lambda i: probs[i]))

                all_subs = self.subscribers.get_all_with_settings()
                for chat_id, settings in all_subs.items():
                    user_threshold = settings.get("threshold", 0.70)

                    # Only alert if argmax is a trade class AND confidence >= threshold
                    if pred_class in (1, 2) and probs[pred_class] >= user_threshold:
                        user_signal = "LONG" if pred_class == 1 else "SHORT"
                    else:
                        continue

                    # Build alert with user's derived signal
                    user_pred = dict(pred)
                    user_pred["signal"] = user_signal
                    user_pred["confidence"] = probs[pred_class]
                    text = formatters.fmt_signal_alert(user_pred)
                    try:
                        await context.application.bot.send_message(
                            chat_id=chat_id, text=text,
                            parse_mode=ParseMode.HTML)
                        logger.info(
                            f"Signal alert to {chat_id}: {user_signal} "
                            f"(threshold={user_threshold})")
                    except Exception as e:
                        logger.warning(f"Failed to send to {chat_id}: {e}")

        # Check for new trade events (broadcast to all — these are actual trades)
        trades = readers.read_recent_trades(n_days=1)
        if trades is not None:
            current_count = len(trades)
            if current_count > self._last_trade_count:
                new_count = current_count - self._last_trade_count
                # Broadcast each new trade (most recent first in df, so reverse)
                new_trades = trades.head(new_count).iloc[::-1]
                for _, row in new_trades.iterrows():
                    text = formatters.fmt_trade_event(row.to_dict())
                    await self._broadcast(context.application, text)
                    logger.info(f"Broadcast trade event: {row.get('action', '?')}")
                self._last_trade_count = current_count

    async def hourly_report_job(self, context: ContextTypes.DEFAULT_TYPE):
        """Send full dashboard report to all subscribers (runs every hour)."""
        text = self._build_hourly_report()
        await self._broadcast(context.application, text)
        logger.info("Broadcast hourly report")

    def _build_hourly_report(self) -> str:
        """Build the hourly report text from all data sources."""
        predictions = readers.read_predictions(last_n_hours=1)
        btc = readers.read_latest_btc_price()
        state = readers.read_trading_state()
        position = state.get("position") if state else None
        data_status = readers.read_data_service_status()
        balance_info = readers.read_account_summary()

        # Extract safety stats from trade_history
        safety_stats = None
        if state:
            th = state.get("trade_history", {})
            if isinstance(th, dict):
                trades_list = th.get("trades", [])
                total = len(trades_list)
                total_wins = sum(1 for t in trades_list if t.get("win"))

                # Filter to last 7 days for 7d stats
                from datetime import timedelta
                now = datetime.now(timezone.utc)
                cutoff_7d = now - timedelta(days=7)
                trades_7d = []
                for t in trades_list:
                    ts = t.get("timestamp")
                    if ts:
                        try:
                            trade_time = datetime.fromisoformat(ts)
                            if trade_time.tzinfo is None:
                                trade_time = trade_time.replace(tzinfo=timezone.utc)
                            if trade_time >= cutoff_7d:
                                trades_7d.append(t)
                        except (ValueError, TypeError):
                            pass

                n_7d = len(trades_7d)
                wins_7d = sum(1 for t in trades_7d if t.get("win"))

                safety_stats = {
                    "total_trades": total,
                    "total_wins": total_wins,
                    "total_wr": (total_wins / total * 100) if total > 0 else 0,
                    "7d_trades": n_7d,
                    "7d_wr": (wins_7d / n_7d * 100) if n_7d > 0 else None,
                    "paused": th.get("paused", False),
                    "pause_reason": th.get("pause_reason", ""),
                }

        return formatters.fmt_hourly_report(
            predictions, btc, position, safety_stats, data_status,
            balance_info=balance_info)

    # ------------------------------------------------------------------
    # Application lifecycle
    # ------------------------------------------------------------------

    async def _post_init(self, app: Application):
        """Register bot commands menu with Telegram after startup."""
        commands = [
            BotCommand("status", "Status with probabilities"),
            BotCommand("stats", "Dashboard + probability chart"),
            BotCommand("position", "Position details with PnL"),
            BotCommand("balance", "Account balance and PnL"),
            BotCommand("pnl", "PnL summary by period"),
            BotCommand("equity", "Equity curve chart"),
            BotCommand("trades", "Last 10 trades"),
            BotCommand("health", "System health check"),
            BotCommand("history", "Signal history (last 100)"),
            BotCommand("threshold", "Set alert threshold"),
            BotCommand("start", "Subscribe to alerts"),
            BotCommand("stop", "Unsubscribe"),
            BotCommand("help", "List commands"),
        ]
        await app.bot.set_my_commands(commands)
        logger.info("Bot commands registered with Telegram")

    def build_and_run(self):
        """Build the Application, register handlers and jobs, start polling."""
        self._initialize_tracking_state()

        app = (
            Application.builder()
            .token(self.token)
            .post_init(self._post_init)
            .build()
        )

        # Register command handlers
        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("stop", self.cmd_stop))
        app.add_handler(CommandHandler("threshold", self.cmd_threshold))
        app.add_handler(CommandHandler("status", self.cmd_status))
        app.add_handler(CommandHandler("stats", self.cmd_stats))
        app.add_handler(CommandHandler("position", self.cmd_position))
        app.add_handler(CommandHandler("balance", self.cmd_balance))
        app.add_handler(CommandHandler("pnl", self.cmd_pnl))
        app.add_handler(CommandHandler("equity", self.cmd_equity))
        app.add_handler(CommandHandler("trades", self.cmd_trades))
        app.add_handler(CommandHandler("health", self.cmd_health))
        app.add_handler(CommandHandler("history", self.cmd_history))
        app.add_handler(CommandHandler("help", self.cmd_help))
        app.add_handler(CallbackQueryHandler(
            self.history_callback, pattern=r"^hist_"))

        # Schedule polling jobs
        job_queue = app.job_queue
        job_queue.run_repeating(
            self.poll_changes_job,
            interval=self.poll_interval,
            first=self.poll_interval,  # don't fire immediately
            name="poll_changes",
        )
        job_queue.run_repeating(
            self.hourly_report_job,
            interval=3600,
            first=3600,  # first report after 1 hour
            name="hourly_report",
        )

        logger.info(
            f"Starting bot (poll_interval={self.poll_interval}s, "
            f"subscribers={self.subscribers.count()})")

        # Blocks until SIGTERM/SIGINT — native shutdown handling
        app.run_polling(drop_pending_updates=True)
