#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Discord Bot for Cryptocurrency Trading Signal System

Slash Commands:
- /status - Current signals across timeframes
- /settings - Configure which timeframes to monitor
- /subscribe - Subscribe channel to alerts
- /unsubscribe - Unsubscribe channel from alerts

Auto-sends alerts when signals trigger.
"""

import asyncio
import logging
from typing import Dict, List, Set, Optional
from datetime import datetime

import discord
from discord import app_commands
from discord.ext import commands, tasks

from notifications.alerts import Alert, AlertEngine, SignalType
from core.processor import TimeframeProcessor
from core.config import (
    DISCORD_BOT_TOKEN,
    DISCORD_ALERT_CHANNEL_ID,
    TIMEFRAME_ORDER,
    WINDOW_SIZES,
    DEFAULT_REFRESH_INTERVAL_MINUTES
)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Storage for channel subscriptions
subscribed_channels: Dict[int, Dict] = {}  # channel_id -> {timeframes: set, windows: set}


class TradingSignalBot(commands.Bot):
    """Discord bot for trading signal alerts."""

    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(command_prefix="!", intents=intents)

        self.alert_engine = AlertEngine()
        self.processor: Optional[TimeframeProcessor] = None

    async def setup_hook(self):
        """Set up the bot after login."""
        # Add cog with commands
        await self.add_cog(SignalCommands(self))

        # Sync slash commands
        await self.tree.sync()
        logger.info("Slash commands synced")

        # Start the signal checking task
        self.check_signals_task.start()

    async def on_ready(self):
        """Called when the bot is ready."""
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} guilds")

        # Set activity
        activity = discord.Activity(
            type=discord.ActivityType.watching,
            name="BTC/USDT signals"
        )
        await self.change_presence(activity=activity)

    @tasks.loop(minutes=DEFAULT_REFRESH_INTERVAL_MINUTES)
    async def check_signals_task(self):
        """Periodic task to check for new signals."""
        logger.info("Running Discord signal check...")

        try:
            # Reload and process data
            self.processor = TimeframeProcessor()
            self.processor.load_all_data()
            self.processor.process_all()

            # Check for alerts
            alerts = self.alert_engine.check_and_alert(self.processor.results)

            for alert in alerts:
                await self.broadcast_alert(alert)

        except Exception as e:
            logger.error(f"Error in signal check task: {e}")

    @check_signals_task.before_loop
    async def before_check_signals(self):
        """Wait until the bot is ready before starting the task."""
        await self.wait_until_ready()

    async def broadcast_alert(self, alert: Alert) -> int:
        """Broadcast an alert to all subscribed channels."""
        sent_count = 0

        for channel_id, settings in subscribed_channels.items():
            # Check if channel is subscribed to this timeframe
            if alert.timeframe not in settings.get('timeframes', set()):
                continue

            # Check if channel is subscribed to this window size
            if alert.window_size not in settings.get('windows', set()):
                continue

            channel = self.get_channel(channel_id)
            if channel is None:
                continue

            embed = self.create_alert_embed(alert)

            try:
                await channel.send(embed=embed)
                sent_count += 1
            except Exception as e:
                logger.error(f"Failed to send alert to channel {channel_id}: {e}")

        return sent_count

    def create_alert_embed(self, alert: Alert) -> discord.Embed:
        """Create a Discord embed for an alert."""
        color_map = {
            SignalType.BUY: discord.Color.green(),
            SignalType.SELL: discord.Color.red(),
            SignalType.HOLD_UP: discord.Color.blue(),
            SignalType.HOLD_DOWN: discord.Color.purple()
        }

        emoji_map = {
            SignalType.BUY: '🟢',
            SignalType.SELL: '🔴',
            SignalType.HOLD_UP: '🔵',
            SignalType.HOLD_DOWN: '🟣'
        }

        embed = discord.Embed(
            title=f"{emoji_map.get(alert.signal_type, '⚪')} {alert.signal_type.value} Signal",
            color=color_map.get(alert.signal_type, discord.Color.default()),
            timestamp=alert.timestamp or datetime.utcnow()
        )

        embed.add_field(name="Timeframe", value=alert.timeframe, inline=True)
        embed.add_field(name="Window", value=str(alert.window_size), inline=True)

        if alert.slope_f is not None:
            embed.add_field(name="Forward Slope", value=f"{alert.slope_f:.6f}", inline=True)

        if alert.spearman is not None:
            embed.add_field(name="Spearman", value=f"{alert.spearman:.4f}", inline=True)

        if alert.angle is not None:
            embed.add_field(name="Angle", value=f"{alert.angle:.2f}°", inline=True)

        embed.set_footer(text="Trading Signal Bot")

        return embed


class SignalCommands(commands.Cog):
    """Cog containing slash commands for the trading signal bot."""

    def __init__(self, bot: TradingSignalBot):
        self.bot = bot

    @app_commands.command(name="status", description="View current trading signals")
    @app_commands.describe(
        timeframe="Specific timeframe to check (optional)"
    )
    async def status(
        self,
        interaction: discord.Interaction,
        timeframe: Optional[str] = None
    ):
        """Show current trading signals."""
        await interaction.response.defer()

        try:
            # Process data if not already done
            if self.bot.processor is None:
                self.bot.processor = TimeframeProcessor()
                self.bot.processor.load_all_data()
                self.bot.processor.process_all()

            signals_df = self.bot.processor.get_signals_table()

            if len(signals_df) == 0:
                await interaction.followup.send("❌ No data available.")
                return

            # Filter by timeframe if specified
            if timeframe:
                signals_df = signals_df[signals_df['timeframe'] == timeframe.upper()]

            # Create embed
            embed = discord.Embed(
                title="📊 Current Trading Signals",
                color=discord.Color.blue(),
                timestamp=datetime.utcnow()
            )

            # Group by signal type
            buy_signals = signals_df[signals_df['signal'] == 'BUY']
            sell_signals = signals_df[signals_df['signal'] == 'SELL']

            if len(buy_signals) > 0:
                buy_text = "\n".join([
                    f"• {row['timeframe']} (w={row['window']})"
                    for _, row in buy_signals.iterrows()
                ])
                embed.add_field(name="🟢 BUY Signals", value=buy_text[:1024], inline=False)

            if len(sell_signals) > 0:
                sell_text = "\n".join([
                    f"• {row['timeframe']} (w={row['window']})"
                    for _, row in sell_signals.iterrows()
                ])
                embed.add_field(name="🔴 SELL Signals", value=sell_text[:1024], inline=False)

            # Summary
            embed.add_field(
                name="Summary",
                value=f"BUY: {len(buy_signals)} | SELL: {len(sell_signals)} | Total: {len(signals_df)}",
                inline=False
            )

            embed.set_footer(text="Use /subscribe to get automatic alerts")

            await interaction.followup.send(embed=embed)

        except Exception as e:
            logger.error(f"Error in status command: {e}")
            await interaction.followup.send(f"❌ Error: {str(e)}")

    @app_commands.command(name="subscribe", description="Subscribe this channel to trading alerts")
    @app_commands.describe(
        timeframes="Comma-separated timeframes (e.g., 1H,4H,1D) or 'all'"
    )
    @app_commands.checks.has_permissions(manage_channels=True)
    async def subscribe(
        self,
        interaction: discord.Interaction,
        timeframes: str = "1H,4H,1D"
    ):
        """Subscribe a channel to alerts."""
        channel_id = interaction.channel_id

        # Parse timeframes
        if timeframes.lower() == 'all':
            tf_set = set(TIMEFRAME_ORDER)
        else:
            tf_set = {tf.strip().upper() for tf in timeframes.split(',')}
            tf_set = tf_set.intersection(set(TIMEFRAME_ORDER))

        if not tf_set:
            await interaction.response.send_message(
                f"❌ Invalid timeframes. Valid options: {', '.join(TIMEFRAME_ORDER)}"
            )
            return

        # Default window sizes
        ws_set = {30, 60, 100}

        subscribed_channels[channel_id] = {
            'timeframes': tf_set,
            'windows': ws_set
        }

        embed = discord.Embed(
            title="✅ Subscribed to Trading Alerts",
            color=discord.Color.green()
        )
        embed.add_field(name="Timeframes", value=", ".join(sorted(tf_set)), inline=False)
        embed.add_field(name="Window Sizes", value=", ".join(map(str, sorted(ws_set))), inline=False)

        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="unsubscribe", description="Unsubscribe this channel from trading alerts")
    @app_commands.checks.has_permissions(manage_channels=True)
    async def unsubscribe(self, interaction: discord.Interaction):
        """Unsubscribe a channel from alerts."""
        channel_id = interaction.channel_id

        if channel_id in subscribed_channels:
            del subscribed_channels[channel_id]
            await interaction.response.send_message("✅ Unsubscribed from trading alerts.")
        else:
            await interaction.response.send_message("ℹ️ This channel was not subscribed.")

    @app_commands.command(name="settings", description="View current subscription settings")
    async def settings(self, interaction: discord.Interaction):
        """Show current subscription settings for this channel."""
        channel_id = interaction.channel_id

        if channel_id not in subscribed_channels:
            await interaction.response.send_message(
                "ℹ️ This channel is not subscribed to alerts.\n"
                "Use `/subscribe` to start receiving alerts."
            )
            return

        settings = subscribed_channels[channel_id]

        embed = discord.Embed(
            title="⚙️ Subscription Settings",
            color=discord.Color.blue()
        )
        embed.add_field(
            name="Timeframes",
            value=", ".join(sorted(settings['timeframes'])),
            inline=False
        )
        embed.add_field(
            name="Window Sizes",
            value=", ".join(map(str, sorted(settings['windows']))),
            inline=False
        )

        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="timeframes", description="List available timeframes")
    async def timeframes(self, interaction: discord.Interaction):
        """List all available timeframes."""
        tf_list = "\n".join([f"• {tf}" for tf in TIMEFRAME_ORDER])

        embed = discord.Embed(
            title="📊 Available Timeframes",
            description=tf_list,
            color=discord.Color.blue()
        )

        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="help", description="Show bot help information")
    async def help_command(self, interaction: discord.Interaction):
        """Show help information."""
        embed = discord.Embed(
            title="🤖 Trading Signal Bot Help",
            color=discord.Color.blue()
        )

        embed.add_field(
            name="Commands",
            value=(
                "`/status` - View current signals\n"
                "`/subscribe` - Subscribe channel to alerts\n"
                "`/unsubscribe` - Unsubscribe from alerts\n"
                "`/settings` - View subscription settings\n"
                "`/timeframes` - List available timeframes\n"
                "`/help` - Show this message"
            ),
            inline=False
        )

        embed.add_field(
            name="Signal Types",
            value=(
                "🟢 **BUY** - Strong buy signal\n"
                "🔴 **SELL** - Strong sell signal\n"
                "🔵 **HOLD (Up)** - Hold in uptrend\n"
                "🟣 **HOLD (Down)** - Hold in downtrend"
            ),
            inline=False
        )

        embed.add_field(
            name="How It Works",
            value=(
                "The bot analyzes BTC/USDT price data using rolling "
                "linear regression to detect trend changes. Signals are "
                "generated when the forward slope and Spearman correlation "
                "cross statistical thresholds."
            ),
            inline=False
        )

        await interaction.response.send_message(embed=embed)


def main():
    """Main entry point."""
    if not DISCORD_BOT_TOKEN:
        logger.error("No Discord bot token configured!")
        print("\n" + "=" * 50)
        print("DISCORD BOT SETUP")
        print("=" * 50)
        print("\n1. Go to Discord Developer Portal:")
        print("   https://discord.com/developers/applications")
        print("\n2. Create a new application")
        print("\n3. Go to Bot section and click 'Add Bot'")
        print("\n4. Copy the token and set it in config.py or as environment variable:")
        print("   export DISCORD_BOT_TOKEN='your-token-here'")
        print("\n5. Go to OAuth2 > URL Generator:")
        print("   - Select 'bot' and 'applications.commands' scopes")
        print("   - Select permissions: Send Messages, Embed Links")
        print("   - Use the generated URL to add bot to your server")
        print("\n" + "=" * 50)
        return

    bot = TradingSignalBot()
    bot.run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()
