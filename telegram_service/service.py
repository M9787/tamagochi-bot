"""Telegram monitoring service entry point.

Usage:
    python -u -m telegram_service.service
    python -u -m telegram_service.service --data-dir ./persistent_data --poll-interval 30
"""

import argparse
import logging
import os
import sys

from telegram_service.bot import TelegramMonitorBot
from telegram_service.readers import configure_paths

logger = logging.getLogger("telegram_service")


def main():
    parser = argparse.ArgumentParser(
        description="Telegram monitoring bot for Tamagochi trading system")
    parser.add_argument("--data-dir", type=str, default="/data",
                        help="Data service volume mount (default: /data)")
    parser.add_argument("--state-dir", type=str, default="/app/trading_state",
                        help="Trading state directory (default: /app/trading_state)")
    parser.add_argument("--logs-dir", type=str, default="/app/trading_logs",
                        help="Trading logs directory (default: /app/trading_logs)")
    parser.add_argument("--subscriber-dir", type=str, default="/app/telegram_data",
                        help="Subscriber persistence directory (default: /app/telegram_data)")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between change-detection polls (default: 60)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Token
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.critical("TELEGRAM_BOT_TOKEN environment variable not set")
        sys.exit(1)

    # Configure readers with actual paths
    configure_paths(
        data_dir=args.data_dir,
        state_dir=args.state_dir,
        logs_dir=args.logs_dir,
    )

    subscriber_path = os.path.join(args.subscriber_dir, "subscribers.json")

    logger.info("Telegram monitoring service starting")
    logger.info(f"  data_dir: {args.data_dir}")
    logger.info(f"  state_dir: {args.state_dir}")
    logger.info(f"  logs_dir: {args.logs_dir}")
    logger.info(f"  poll_interval: {args.poll_interval}s")

    bot = TelegramMonitorBot(
        token=token,
        poll_interval=args.poll_interval,
        subscriber_path=subscriber_path,
    )
    bot.build_and_run()

    logger.info("Telegram monitoring service stopped")


if __name__ == "__main__":
    main()
