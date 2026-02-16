#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Console Alert Module for Cryptocurrency Trading Signal System

Features:
- Colored output (green=BUY, red=SELL)
- Real-time monitoring mode
- CSV logging
- Table display of signals
"""

import sys
import time
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Fallback: no colors
    class Fore:
        GREEN = RED = YELLOW = CYAN = MAGENTA = WHITE = RESET = ""
    class Back:
        BLACK = ""
    class Style:
        BRIGHT = DIM = RESET_ALL = ""

from notifications.alerts import Alert, AlertEngine, SignalType, create_log_handler
from core.processor import TimeframeProcessor
from core.config import (
    TIMEFRAME_ORDER,
    WINDOW_SIZES,
    SIGNAL_COLORS,
    get_output_path
)


def get_signal_color(signal_type: SignalType) -> str:
    """Get the color code for a signal type."""
    color_map = {
        SignalType.BUY: Fore.GREEN + Style.BRIGHT,
        SignalType.SELL: Fore.RED + Style.BRIGHT,
        SignalType.HOLD_UP: Fore.CYAN,
        SignalType.HOLD_DOWN: Fore.MAGENTA,
        SignalType.NONE: Fore.WHITE
    }
    return color_map.get(signal_type, Fore.WHITE)


def format_signal_row(
    timeframe: str,
    window: int,
    signal: str,
    slope_f: Optional[float],
    spearman: Optional[float],
    angle: Optional[float]
) -> str:
    """Format a signal row for console output."""
    # Determine color based on signal
    if signal == 'BUY':
        color = Fore.GREEN + Style.BRIGHT
    elif signal == 'SELL':
        color = Fore.RED + Style.BRIGHT
    elif 'HOLD' in signal:
        color = Fore.YELLOW
    else:
        color = Fore.WHITE

    slope_str = f"{slope_f:.6f}" if slope_f is not None else "N/A"
    spearman_str = f"{spearman:.4f}" if spearman is not None else "N/A"
    angle_str = f"{angle:.2f}" if angle is not None else "N/A"

    return (
        f"{color}{timeframe:>4} | {window:>3} | {signal:<12} | "
        f"slope={slope_str:>10} | spearman={spearman_str:>8} | angle={angle_str:>6}°{Style.RESET_ALL}"
    )


def print_header():
    """Print the console header."""
    print("\n" + "=" * 80)
    print(f"{Style.BRIGHT}  CRYPTOCURRENCY TRADING SIGNAL MONITOR{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}BTC/USDT Analysis{Style.RESET_ALL}")
    print("=" * 80)


def print_signal_table(signals_df) -> None:
    """Print the signals as a formatted table."""
    print("\n" + "-" * 80)
    print(f"{'TF':>4} | {'WIN':>3} | {'SIGNAL':<12} | {'SLOPE_F':>16} | {'SPEARMAN':>10} | {'ANGLE':>8}")
    print("-" * 80)

    for _, row in signals_df.iterrows():
        print(format_signal_row(
            row['timeframe'],
            row['window'],
            row['signal'],
            row.get('slope_f'),
            row.get('spearman'),
            row.get('angle')
        ))

    print("-" * 80)


def print_summary(signals_df) -> None:
    """Print signal summary statistics."""
    buy_count = len(signals_df[signals_df['signal'] == 'BUY'])
    sell_count = len(signals_df[signals_df['signal'] == 'SELL'])
    hold_count = len(signals_df[signals_df['signal'].str.contains('HOLD', na=False)])

    print(f"\n{Style.BRIGHT}Summary:{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}BUY:  {buy_count}{Style.RESET_ALL}")
    print(f"  {Fore.RED}SELL: {sell_count}{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}HOLD: {hold_count}{Style.RESET_ALL}")
    print(f"  Total: {len(signals_df)}")


def create_console_alert_handler():
    """Create an alert handler that prints to console."""
    def handler(alert: Alert) -> None:
        color = get_signal_color(alert.signal_type)
        timestamp = alert.timestamp.strftime('%Y-%m-%d %H:%M:%S') if alert.timestamp else 'N/A'

        print(f"\n{color}{'='*60}")
        print(f"  NEW SIGNAL: {alert.signal_type.value}")
        print(f"{'='*60}{Style.RESET_ALL}")
        print(f"  Timeframe: {alert.timeframe}")
        print(f"  Window: {alert.window_size}")
        print(f"  Time: {timestamp}")

        if alert.slope_f is not None:
            print(f"  Forward Slope: {alert.slope_f:.6f}")
        if alert.spearman is not None:
            print(f"  Spearman: {alert.spearman:.4f}")
        if alert.angle is not None:
            print(f"  Angle: {alert.angle:.2f}°")

        print(f"{color}{'='*60}{Style.RESET_ALL}\n")

    return handler


class CSVLogger:
    """Log alerts and signals to CSV files."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the CSV logger."""
        self.output_dir = output_dir or get_output_path("").parent / "logs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.alerts_file = self.output_dir / "alerts_log.csv"
        self.signals_file = self.output_dir / "signals_log.csv"

        self._init_files()

    def _init_files(self):
        """Initialize CSV files with headers if they don't exist."""
        if not self.alerts_file.exists():
            with open(self.alerts_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'log_time', 'signal_time', 'signal_type', 'timeframe',
                    'window', 'slope_f', 'spearman', 'angle'
                ])

        if not self.signals_file.exists():
            with open(self.signals_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'log_time', 'timeframe', 'window', 'signal',
                    'slope_f', 'spearman', 'angle', 'recent_buys', 'recent_sells'
                ])

    def log_alert(self, alert: Alert) -> None:
        """Log an alert to CSV."""
        with open(self.alerts_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(),
                alert.timestamp.isoformat() if alert.timestamp else '',
                alert.signal_type.value,
                alert.timeframe,
                alert.window_size,
                alert.slope_f,
                alert.spearman,
                alert.angle
            ])

    def log_signals(self, signals_df) -> None:
        """Log all current signals to CSV."""
        log_time = datetime.utcnow().isoformat()

        with open(self.signals_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for _, row in signals_df.iterrows():
                writer.writerow([
                    log_time,
                    row['timeframe'],
                    row['window'],
                    row['signal'],
                    row.get('slope_f'),
                    row.get('spearman'),
                    row.get('angle'),
                    row.get('recent_buys'),
                    row.get('recent_sells')
                ])

    def create_handler(self):
        """Create an alert handler that logs to CSV."""
        def handler(alert: Alert) -> None:
            self.log_alert(alert)

        return handler


def run_once(
    timeframes: Optional[List[str]] = None,
    window_sizes: Optional[List[int]] = None,
    export_csv: bool = False
) -> None:
    """
    Run analysis once and display results.

    Args:
        timeframes: List of timeframes to process
        window_sizes: List of window sizes
        export_csv: Whether to log results to CSV
    """
    print_header()
    print(f"\n{Fore.CYAN}Loading and processing data...{Style.RESET_ALL}")

    processor = TimeframeProcessor(timeframes, window_sizes)
    processor.load_all_data()
    processor.process_all()

    signals_df = processor.get_signals_table()

    if len(signals_df) == 0:
        print(f"\n{Fore.YELLOW}No data available. Check if CSV files exist.{Style.RESET_ALL}")
        return

    print_signal_table(signals_df)
    print_summary(signals_df)

    if export_csv:
        logger = CSVLogger()
        logger.log_signals(signals_df)
        print(f"\n{Fore.CYAN}Signals logged to: {logger.signals_file}{Style.RESET_ALL}")

    print(f"\n{Style.DIM}Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}{Style.RESET_ALL}")


def run_continuous(
    interval_minutes: int = 5,
    timeframes: Optional[List[str]] = None,
    window_sizes: Optional[List[int]] = None,
    export_csv: bool = True
) -> None:
    """
    Run continuous monitoring mode.

    Args:
        interval_minutes: Minutes between updates
        timeframes: List of timeframes to process
        window_sizes: List of window sizes
        export_csv: Whether to log results to CSV
    """
    print_header()
    print(f"\n{Fore.CYAN}Starting continuous monitoring mode...{Style.RESET_ALL}")
    print(f"  Refresh interval: {interval_minutes} minutes")
    print(f"  Press Ctrl+C to stop\n")

    # Set up alert engine
    alert_engine = AlertEngine()
    alert_engine.add_handler(create_console_alert_handler())
    alert_engine.add_handler(create_log_handler())

    if export_csv:
        csv_logger = CSVLogger()
        alert_engine.add_handler(csv_logger.create_handler())

    processor = TimeframeProcessor(timeframes, window_sizes)

    iteration = 0

    try:
        while True:
            iteration += 1
            print(f"\n{Style.DIM}[{datetime.utcnow().strftime('%H:%M:%S')}] Iteration {iteration}...{Style.RESET_ALL}")

            try:
                # Load and process data
                processor.load_all_data()
                processor.process_all()

                # Get signals
                signals_df = processor.get_signals_table()

                if len(signals_df) > 0:
                    # Print compact summary
                    buy_count = len(signals_df[signals_df['signal'] == 'BUY'])
                    sell_count = len(signals_df[signals_df['signal'] == 'SELL'])

                    status_color = Fore.GREEN if buy_count > sell_count else (Fore.RED if sell_count > buy_count else Fore.YELLOW)
                    print(f"  {status_color}BUY: {buy_count} | SELL: {sell_count}{Style.RESET_ALL}")

                    # Check for new alerts
                    alerts = alert_engine.check_and_alert(processor.results)

                    if export_csv:
                        csv_logger.log_signals(signals_df)

                else:
                    print(f"  {Fore.YELLOW}No data available{Style.RESET_ALL}")

            except Exception as e:
                print(f"  {Fore.RED}Error: {e}{Style.RESET_ALL}")

            # Wait for next iteration
            print(f"{Style.DIM}  Next update in {interval_minutes} minutes...{Style.RESET_ALL}")
            time.sleep(interval_minutes * 60)

    except KeyboardInterrupt:
        print(f"\n\n{Fore.CYAN}Monitoring stopped by user.{Style.RESET_ALL}")

        # Save alert history
        if len(alert_engine.alert_history) > 0:
            alert_engine.save_history()
            print(f"Alert history saved: {len(alert_engine.alert_history)} alerts")


def main():
    """Main entry point with command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Console-based cryptocurrency trading signal monitor"
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['once', 'continuous'],
        default='once',
        help='Run mode: once (single analysis) or continuous (monitoring)'
    )
    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=5,
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
        '--no-csv',
        action='store_true',
        help='Disable CSV logging'
    )

    args = parser.parse_args()

    # Parse timeframes
    timeframes = None
    if args.timeframes:
        timeframes = [tf.strip().upper() for tf in args.timeframes.split(',')]

    # Parse window sizes
    window_sizes = None
    if args.windows:
        window_sizes = [int(w.strip()) for w in args.windows.split(',')]

    export_csv = not args.no_csv

    if args.mode == 'once':
        run_once(timeframes, window_sizes, export_csv)
    else:
        run_continuous(args.interval, timeframes, window_sizes, export_csv)


if __name__ == "__main__":
    main()
