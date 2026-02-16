#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Telegram Bot for Cryptocurrency Trading Signal System

Commands:
- /start - Subscribe to alerts
- /status - Current signals across timeframes
- /settings - Configure which timeframes to monitor
- /help - Show help message

Auto-sends alerts when signals trigger.
"""

import asyncio
import logging
from typing import Dict, List, Set, Optional
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    JobQueue
)

from notifications.alerts import Alert, AlertEngine, SignalType
from core.processor import TimeframeProcessor
from core.config import (
    TELEGRAM_BOT_TOKEN,
    TIMEFRAME_ORDER,
    WINDOW_SIZES,
    DEFAULT_REFRESH_INTERVAL_MINUTES
)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Storage for user subscriptions
user_subscriptions: Dict[int, Set[str]] = {}  # user_id -> set of timeframes
user_windows: Dict[int, Set[int]] = {}  # user_id -> set of window sizes


class TelegramAlertBot:
    """Telegram bot for trading signal alerts."""

    def __init__(self, token: str):
        """
        Initialize the bot.

        Args:
            token: Telegram bot token from BotFather
        """
        self.token = token
        self.application: Optional[Application] = None
        self.alert_engine = AlertEngine()
        self.processor: Optional[TimeframeProcessor] = None

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        user_id = update.effective_user.id
        username = update.effective_user.username or "User"

        # Initialize user with default settings
        if user_id not in user_subscriptions:
            user_subscriptions[user_id] = set(TIMEFRAME_ORDER[:5])  # Default: top 5 timeframes
            user_windows[user_id] = {30, 60, 100}  # Default windows

        await update.message.reply_text(
            f"Welcome {username}! 🚀\n\n"
            "I'll send you cryptocurrency trading signals based on technical analysis.\n\n"
            "Commands:\n"
            "/status - View current signals\n"
            "/settings - Configure alerts\n"
            "/help - Show this message\n\n"
            f"Currently monitoring: {', '.join(sorted(user_subscriptions[user_id]))}\n"
            f"Window sizes: {', '.join(map(str, sorted(user_windows[user_id])))}"
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        help_text = """
🤖 *Trading Signal Bot Help*

*Commands:*
/start - Start the bot and subscribe to alerts
/status - View current signals for all monitored timeframes
/settings - Configure which timeframes and windows to monitor
/help - Show this help message

*Signal Types:*
🟢 BUY - Strong buy signal (slope up, resistance)
🔴 SELL - Strong sell signal (slope down, resistance)
🔵 HOLD (Up) - Hold in uptrend
🟣 HOLD (Down) - Hold in downtrend

*Timeframes:*
3D, 1D, 12H, 8H, 6H, 4H, 2H, 1H, 30M, 15M, 5M

*Window Sizes:*
30, 60, 100, 120, 160 (periods for regression analysis)

*How it works:*
The bot analyzes BTC/USDT price data using rolling linear regression
to detect trend changes and generate trading signals.
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command - show current signals."""
        user_id = update.effective_user.id

        # Get user's subscribed timeframes
        timeframes = user_subscriptions.get(user_id, set(TIMEFRAME_ORDER[:3]))
        windows = user_windows.get(user_id, {30, 60})

        await update.message.reply_text("⏳ Analyzing signals...")

        try:
            # Process data
            if self.processor is None:
                self.processor = TimeframeProcessor(list(timeframes), list(windows))
                self.processor.load_all_data()
                self.processor.process_all()

            signals_df = self.processor.get_signals_table()

            if len(signals_df) == 0:
                await update.message.reply_text("❌ No data available. Please try again later.")
                return

            # Filter by user preferences
            filtered = signals_df[
                (signals_df['timeframe'].isin(timeframes)) &
                (signals_df['window'].isin(windows))
            ]

            # Format message
            message_lines = ["📊 *Current Signals*\n"]

            for _, row in filtered.iterrows():
                signal = row['signal']
                emoji = {
                    'BUY': '🟢',
                    'SELL': '🔴',
                    'HOLD': '🟡',
                    'HOLD_STRONG': '🔵'
                }.get(signal, '⚪')

                message_lines.append(
                    f"{emoji} *{row['timeframe']}* (w={row['window']}): {signal}"
                )

                if row['slope_f'] is not None:
                    message_lines.append(f"   Slope: {row['slope_f']:.6f}")

            # Summary
            buy_count = len(filtered[filtered['signal'] == 'BUY'])
            sell_count = len(filtered[filtered['signal'] == 'SELL'])

            message_lines.append(f"\n📈 BUY signals: {buy_count}")
            message_lines.append(f"📉 SELL signals: {sell_count}")
            message_lines.append(f"\n_Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_")

            await update.message.reply_text(
                '\n'.join(message_lines),
                parse_mode='Markdown'
            )

        except Exception as e:
            logger.error(f"Error in status: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /settings command - show settings menu."""
        keyboard = [
            [InlineKeyboardButton("📊 Timeframes", callback_data="settings_timeframes")],
            [InlineKeyboardButton("📏 Window Sizes", callback_data="settings_windows")],
            [InlineKeyboardButton("🔔 Alert Preferences", callback_data="settings_alerts")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "⚙️ *Settings*\n\nChoose what to configure:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def settings_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle settings button callbacks."""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        data = query.data

        if data == "settings_timeframes":
            # Show timeframe selection
            current = user_subscriptions.get(user_id, set())
            keyboard = []

            for tf in TIMEFRAME_ORDER:
                status = "✅" if tf in current else "⬜"
                keyboard.append([
                    InlineKeyboardButton(f"{status} {tf}", callback_data=f"toggle_tf_{tf}")
                ])

            keyboard.append([InlineKeyboardButton("← Back", callback_data="settings_back")])
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                "📊 *Select Timeframes to Monitor*\n\nTap to toggle:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )

        elif data == "settings_windows":
            # Show window size selection
            current = user_windows.get(user_id, set())
            keyboard = []

            for ws in WINDOW_SIZES:
                status = "✅" if ws in current else "⬜"
                keyboard.append([
                    InlineKeyboardButton(f"{status} Window {ws}", callback_data=f"toggle_ws_{ws}")
                ])

            keyboard.append([InlineKeyboardButton("← Back", callback_data="settings_back")])
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                "📏 *Select Window Sizes*\n\nTap to toggle:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )

        elif data.startswith("toggle_tf_"):
            # Toggle timeframe
            tf = data.replace("toggle_tf_", "")
            if user_id not in user_subscriptions:
                user_subscriptions[user_id] = set()

            if tf in user_subscriptions[user_id]:
                user_subscriptions[user_id].discard(tf)
            else:
                user_subscriptions[user_id].add(tf)

            # Refresh the timeframes view
            await self.settings_callback(
                Update(update.update_id, callback_query=query._replace(data="settings_timeframes")),
                context
            )

        elif data.startswith("toggle_ws_"):
            # Toggle window size
            ws = int(data.replace("toggle_ws_", ""))
            if user_id not in user_windows:
                user_windows[user_id] = set()

            if ws in user_windows[user_id]:
                user_windows[user_id].discard(ws)
            else:
                user_windows[user_id].add(ws)

            # Refresh the windows view
            await self.settings_callback(
                Update(update.update_id, callback_query=query._replace(data="settings_windows")),
                context
            )

        elif data == "settings_back":
            # Go back to main settings
            keyboard = [
                [InlineKeyboardButton("📊 Timeframes", callback_data="settings_timeframes")],
                [InlineKeyboardButton("📏 Window Sizes", callback_data="settings_windows")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                "⚙️ *Settings*\n\nChoose what to configure:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )

    async def send_alert(self, chat_id: int, alert: Alert) -> None:
        """Send an alert to a specific chat."""
        if self.application is None:
            return

        emoji = {
            SignalType.BUY: '🟢',
            SignalType.SELL: '🔴',
            SignalType.HOLD_UP: '🔵',
            SignalType.HOLD_DOWN: '🟣'
        }.get(alert.signal_type, '⚪')

        message = f"""
{emoji} *{alert.signal_type.value} SIGNAL*

📊 Timeframe: {alert.timeframe}
📏 Window: {alert.window_size}
⏰ Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M UTC') if alert.timestamp else 'N/A'}

📈 Forward Slope: {alert.slope_f:.6f if alert.slope_f else 'N/A'}
📊 Spearman: {alert.spearman:.4f if alert.spearman else 'N/A'}
📐 Angle: {f'{alert.angle:.2f}°' if alert.angle else 'N/A'}
        """

        try:
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Failed to send alert to {chat_id}: {e}")

    async def broadcast_alert(self, alert: Alert) -> int:
        """Broadcast an alert to all subscribed users."""
        sent_count = 0

        for user_id, timeframes in user_subscriptions.items():
            # Check if user is subscribed to this timeframe
            if alert.timeframe not in timeframes:
                continue

            # Check if user is subscribed to this window size
            windows = user_windows.get(user_id, set())
            if alert.window_size not in windows:
                continue

            await self.send_alert(user_id, alert)
            sent_count += 1

        return sent_count

    async def check_signals_job(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Periodic job to check for new signals."""
        logger.info("Running signal check job...")

        try:
            # Reload and process data
            self.processor = TimeframeProcessor()
            self.processor.load_all_data()
            self.processor.process_all()

            # Check for alerts
            alerts = self.alert_engine.check_and_alert(self.processor.results)

            for alert in alerts:
                count = await self.broadcast_alert(alert)
                logger.info(f"Sent alert {alert.signal_type.value} to {count} users")

        except Exception as e:
            logger.error(f"Error in signal check job: {e}")

    def setup_handlers(self) -> None:
        """Set up command handlers."""
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("status", self.status))
        self.application.add_handler(CommandHandler("settings", self.settings))
        self.application.add_handler(CallbackQueryHandler(self.settings_callback))

    def run(self, check_interval_minutes: int = DEFAULT_REFRESH_INTERVAL_MINUTES) -> None:
        """
        Run the bot.

        Args:
            check_interval_minutes: How often to check for new signals
        """
        if not self.token:
            logger.error("No Telegram bot token configured!")
            print("\n" + "=" * 50)
            print("TELEGRAM BOT SETUP")
            print("=" * 50)
            print("\n1. Message @BotFather on Telegram")
            print("2. Send /newbot and follow the prompts")
            print("3. Copy the token and set it in config.py or as environment variable:")
            print("   export TELEGRAM_BOT_TOKEN='your-token-here'")
            print("\n" + "=" * 50)
            return

        self.application = Application.builder().token(self.token).build()
        self.setup_handlers()

        # Set up periodic signal checking
        job_queue = self.application.job_queue
        job_queue.run_repeating(
            self.check_signals_job,
            interval=check_interval_minutes * 60,
            first=10  # Start first check after 10 seconds
        )

        logger.info(f"Starting Telegram bot with {check_interval_minutes} minute check interval")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    """Main entry point."""
    bot = TelegramAlertBot(TELEGRAM_BOT_TOKEN)
    bot.run()


if __name__ == "__main__":
    main()
