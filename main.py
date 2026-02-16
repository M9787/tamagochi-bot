#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Entry Point for Cryptocurrency Trading Signal System

Usage:
    python main.py --mode manual          # Single analysis run
    python main.py --mode continuous      # Continuous monitoring
    python main.py --mode dashboard       # Launch Streamlit dashboard
    python main.py --mode telegram        # Run Telegram bot
    python main.py --mode discord         # Run Discord bot
    python main.py --download             # Download fresh data from Binance
"""

import argparse
import sys
import subprocess
from pathlib import Path
from typing import List, Optional

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.config import (
    TIMEFRAME_ORDER,
    WINDOW_SIZES,
    DATA_DIR,
    validate_config,
    DEFAULT_REFRESH_INTERVAL_MINUTES
)


def print_banner():
    """Print the application banner."""
    banner = """
======================================================================
     CRYPTOCURRENCY TRADING SIGNAL ANALYSIS SYSTEM
     BTC/USDT Multi-Timeframe Analysis
======================================================================
"""
    print(banner)


def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    required = [
        'pandas',
        'numpy',
        'scipy',
    ]

    optional = {
        'streamlit': 'dashboard',
        'plotly': 'dashboard',
        'telegram': 'telegram bot',
        'discord': 'discord bot',
        'colorama': 'colored console output',
        'binance': 'data download'
    }

    missing_required = []
    missing_optional = {}

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing_required.append(pkg)

    for pkg, feature in optional.items():
        try:
            __import__(pkg)
        except ImportError:
            missing_optional[pkg] = feature

    if missing_required:
        print("Missing required packages:")
        for pkg in missing_required:
            print(f"  - {pkg}")
        print("\nInstall with: pip install " + " ".join(missing_required))
        return False

    if missing_optional:
        print("Optional packages not installed:")
        for pkg, feature in missing_optional.items():
            print(f"  - {pkg} (needed for {feature})")
        print("")

    return True


def run_download():
    """Run the data download script."""
    print("Downloading data from Binance...")
    script_path = Path(__file__).parent / "data" / "downloader.py"

    if not script_path.exists():
        print(f"Error: Data extraction script not found at {script_path}")
        return False

    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running download script: {e}")
        return False


def run_manual(
    timeframes: Optional[List[str]] = None,
    window_sizes: Optional[List[int]] = None,
    export: bool = True
):
    """Run a single analysis pass."""
    from core.processor import run_analysis
    from notifications.console_alerts import print_signal_table, print_summary, print_header

    print_header()

    processor = run_analysis(timeframes, window_sizes, export)
    signals_df = processor.get_signals_table()

    if len(signals_df) > 0:
        print_signal_table(signals_df)
        print_summary(signals_df)
    else:
        print("No data available. Run with --download to fetch data first.")


def run_continuous(
    interval: int = DEFAULT_REFRESH_INTERVAL_MINUTES,
    timeframes: Optional[List[str]] = None,
    window_sizes: Optional[List[int]] = None
):
    """Run continuous monitoring mode."""
    from notifications.console_alerts import run_continuous as console_continuous

    console_continuous(
        interval_minutes=interval,
        timeframes=timeframes,
        window_sizes=window_sizes,
        export_csv=True
    )


def run_dashboard():
    """Launch the Streamlit dashboard."""
    try:
        import streamlit
    except ImportError:
        print("Error: Streamlit not installed.")
        print("Install with: pip install streamlit plotly")
        return

    dashboard_path = Path(__file__).parent / "dashboard.py"

    if not dashboard_path.exists():
        print(f"Error: Dashboard script not found at {dashboard_path}")
        return

    print("Launching Streamlit dashboard...")
    print("Open http://localhost:8501 in your browser")
    print("Press Ctrl+C to stop\n")

    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])


def run_telegram():
    """Run the Telegram bot."""
    try:
        from notifications.telegram_bot import main as telegram_main
        telegram_main()
    except ImportError as e:
        print(f"Error: {e}")
        print("Install with: pip install python-telegram-bot")


def run_discord():
    """Run the Discord bot."""
    try:
        from notifications.discord_bot import main as discord_main
        discord_main()
    except ImportError as e:
        print(f"Error: {e}")
        print("Install with: pip install discord.py")


def run_validate():
    """Validate configuration and data availability."""
    print("Validating configuration...\n")

    checks = validate_config()

    all_ok = True
    for key, value in checks.items():
        status = "OK" if value else "MISSING"
        symbol = "[+]" if value else "[-]"
        print(f"  {symbol} {key}: {status}")

        if not value and not key.startswith('csv_'):
            all_ok = False

    # Count available CSVs
    csv_checks = {k: v for k, v in checks.items() if k.startswith('csv_')}
    csv_available = sum(csv_checks.values())

    print(f"\nData files: {csv_available}/{len(csv_checks)} available")

    if csv_available == 0:
        print("\nNo data files found. Run with --download to fetch data.")
        all_ok = False

    if all_ok:
        print("\nConfiguration OK!")
    else:
        print("\nSome issues found. Check the output above.")

    return all_ok


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cryptocurrency Trading Signal Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode manual           Run single analysis
  python main.py --mode continuous -i 5  Monitor every 5 minutes
  python main.py --mode dashboard        Launch web dashboard
  python main.py --download              Download fresh data
  python main.py --validate              Check configuration
        """
    )

    parser.add_argument(
        '--mode', '-m',
        choices=['manual', 'continuous', 'dashboard', 'telegram', 'discord'],
        default='manual',
        help='Operation mode'
    )

    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=DEFAULT_REFRESH_INTERVAL_MINUTES,
        help='Refresh interval in minutes (for continuous mode)'
    )

    parser.add_argument(
        '--timeframes', '-t',
        type=str,
        default=None,
        help='Comma-separated timeframes (e.g., 1H,4H,1D)'
    )

    parser.add_argument(
        '--windows', '-w',
        type=str,
        default=None,
        help='Comma-separated window sizes (e.g., 30,60,100)'
    )

    parser.add_argument(
        '--download', '-d',
        action='store_true',
        help='Download fresh data from Binance before analysis'
    )

    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Validate configuration and exit'
    )

    parser.add_argument(
        '--no-export',
        action='store_true',
        help='Disable CSV export'
    )

    args = parser.parse_args()

    print_banner()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Validate mode
    if args.validate:
        success = run_validate()
        sys.exit(0 if success else 1)

    # Download if requested
    if args.download:
        if not run_download():
            print("Download failed. Continuing with existing data...")

    # Parse timeframes
    timeframes = None
    if args.timeframes:
        timeframes = [tf.strip().upper() for tf in args.timeframes.split(',')]

    # Parse window sizes
    window_sizes = None
    if args.windows:
        window_sizes = [int(w.strip()) for w in args.windows.split(',')]

    # Run selected mode
    if args.mode == 'manual':
        run_manual(timeframes, window_sizes, not args.no_export)

    elif args.mode == 'continuous':
        run_continuous(args.interval, timeframes, window_sizes)

    elif args.mode == 'dashboard':
        run_dashboard()

    elif args.mode == 'telegram':
        run_telegram()

    elif args.mode == 'discord':
        run_discord()


if __name__ == "__main__":
    main()
