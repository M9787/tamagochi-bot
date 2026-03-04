"""Telegram monitoring bot — commands, push notifications, polling jobs."""

import logging
from datetime import datetime, timezone

from telegram import BotCommand, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

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
            await update.message.reply_text(
                "\u2705 Subscribed to Tamagochi trading alerts!\n\n"
                "You'll receive:\n"
                "- Signal alerts (LONG/SHORT predictions)\n"
                "- Trade events (opened, closed, SL/TP)\n"
                "- Hourly dashboard reports\n\n"
                "Commands: /status /stats /position /trades /health /stop /help"
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

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Quick one-liner: position, price, last prediction."""
        pred = readers.read_latest_prediction()
        btc = readers.read_latest_btc_price()
        state = readers.read_trading_state()
        position = state.get("position") if state else None

        text = formatters.fmt_status_oneliner(pred, btc, position)
        await update.message.reply_text(text)

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Full hourly report (same as push)."""
        text = self._build_hourly_report()
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async def cmd_position(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Detailed position with PnL estimate."""
        state = readers.read_trading_state()
        position = state.get("position") if state else None
        btc = readers.read_latest_btc_price()

        text = formatters.fmt_position_detail(position, btc)
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async def cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Last 10 trades."""
        trades = readers.read_last_n_trades(10)
        text = formatters.fmt_trades_list(trades)
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async def cmd_health(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """System health check."""
        data_status = readers.read_data_service_status()
        state = readers.read_trading_state()

        text = formatters.fmt_health(data_status, state)
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List all commands."""
        text = (
            "\U0001f916 <b>Tamagochi Trading Bot</b>\n\n"
            "/status  — Quick status (position, BTC, last signal)\n"
            "/stats   — Full hourly dashboard report\n"
            "/position — Detailed position info with PnL\n"
            "/trades  — Last 10 trades\n"
            "/health  — System health check\n"
            "/start   — Subscribe to push notifications\n"
            "/stop    — Unsubscribe\n"
            "/help    — This message"
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
            signal = pred.get("signal", "NO_TRADE")
            if signal in ("LONG", "SHORT"):
                text = formatters.fmt_signal_alert(pred)
                await self._broadcast(context.application, text)
                logger.info(f"Broadcast signal alert: {signal}")

        # Check for new trade events
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

        # Extract safety stats from trade_history
        safety_stats = None
        if state:
            th = state.get("trade_history", {})
            if isinstance(th, dict):
                trades_list = th.get("trades", [])
                total = len(trades_list)
                total_wins = sum(1 for t in trades_list if t.get("win"))
                safety_stats = {
                    "total_trades": total,
                    "total_wins": total_wins,
                    "total_wr": (total_wins / total * 100) if total > 0 else 0,
                    "7d_trades": total,  # simplified; full logic in SafetyMonitor
                    "7d_wr": (total_wins / total * 100) if total > 0 else None,
                    "paused": th.get("paused", False),
                    "pause_reason": th.get("pause_reason", ""),
                }

        return formatters.fmt_hourly_report(
            predictions, btc, position, safety_stats, data_status)

    # ------------------------------------------------------------------
    # Application lifecycle
    # ------------------------------------------------------------------

    async def _post_init(self, app: Application):
        """Register bot commands menu with Telegram after startup."""
        commands = [
            BotCommand("status", "Quick status overview"),
            BotCommand("stats", "Full hourly dashboard"),
            BotCommand("position", "Position details with PnL"),
            BotCommand("trades", "Last 10 trades"),
            BotCommand("health", "System health check"),
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
        app.add_handler(CommandHandler("status", self.cmd_status))
        app.add_handler(CommandHandler("stats", self.cmd_stats))
        app.add_handler(CommandHandler("position", self.cmd_position))
        app.add_handler(CommandHandler("trades", self.cmd_trades))
        app.add_handler(CommandHandler("health", self.cmd_health))
        app.add_handler(CommandHandler("help", self.cmd_help))

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
