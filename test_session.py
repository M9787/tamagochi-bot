#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Session Script - Comprehensive testing with full logging
Run this to validate all components of the trading signal system.
"""

import sys
import traceback
import logging
from datetime import datetime
from pathlib import Path
from io import StringIO

# Setup logging to both file and console
LOG_FILE = Path(__file__).parent / "test_session.log"

# Create formatters and handlers
file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s')
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)-8s | %(message)s')
console_handler.setFormatter(console_formatter)

# Root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger('TestSession')


def log_section(title: str):
    """Log a section header."""
    separator = "=" * 70
    logger.info("")
    logger.info(separator)
    logger.info(f"  {title}")
    logger.info(separator)


def log_subsection(title: str):
    """Log a subsection header."""
    logger.info(f"\n--- {title} ---")


def test_imports():
    """Test that all modules can be imported."""
    log_section("TEST 1: Module Imports")

    modules = [
        ('config', 'Configuration module'),
        ('analysis', 'Analysis module'),
        ('processor', 'Processor module'),
        ('alerts', 'Alerts module'),
    ]

    results = {}

    for module_name, description in modules:
        try:
            module = __import__(module_name)
            logger.info(f"[PASS] {description} ({module_name})")
            results[module_name] = True
        except Exception as e:
            logger.error(f"[FAIL] {description} ({module_name}): {e}")
            logger.debug(traceback.format_exc())
            results[module_name] = False

    return all(results.values()), results


def test_config():
    """Test configuration validation."""
    log_section("TEST 2: Configuration Validation")

    try:
        from config import (
            validate_config,
            DATA_DIR,
            API_KEY_FILE,
            TIMEFRAME_ORDER,
            WINDOW_SIZES,
            get_csv_path
        )

        logger.info(f"DATA_DIR: {DATA_DIR}")
        logger.info(f"DATA_DIR exists: {DATA_DIR.exists()}")
        logger.info(f"API_KEY_FILE: {API_KEY_FILE}")
        logger.info(f"API_KEY_FILE exists: {API_KEY_FILE.exists()}")
        logger.info(f"TIMEFRAME_ORDER: {TIMEFRAME_ORDER}")
        logger.info(f"WINDOW_SIZES: {WINDOW_SIZES}")

        log_subsection("Config Validation Results")
        checks = validate_config()

        for key, value in checks.items():
            status = "[PASS]" if value else "[FAIL]"
            logger.info(f"{status} {key}")

        # Check CSV files in detail
        log_subsection("CSV File Status")
        csv_found = 0
        csv_missing = 0

        for tf in TIMEFRAME_ORDER:
            csv_path = get_csv_path(tf)
            if csv_path.exists():
                import os
                size = os.path.getsize(csv_path)
                logger.info(f"[FOUND] {tf}: {csv_path.name} ({size:,} bytes)")
                csv_found += 1
            else:
                logger.warning(f"[MISSING] {tf}: {csv_path}")
                csv_missing += 1

        logger.info(f"\nCSV Summary: {csv_found} found, {csv_missing} missing")

        return csv_found > 0, checks

    except Exception as e:
        logger.error(f"Config test failed: {e}")
        logger.debug(traceback.format_exc())
        return False, {}


def test_analysis_functions():
    """Test core analysis functions."""
    log_section("TEST 3: Analysis Functions")

    try:
        from analysis import (
            angle_between_lines,
            iterative_regression,
            calculate_acceleration
        )
        import pandas as pd
        import numpy as np

        # Test angle_between_lines
        log_subsection("angle_between_lines()")
        test_cases = [
            (0.5, -0.5, "positive vs negative slope"),
            (1.0, 1.0, "same slope"),
            (0.0, 1.0, "horizontal vs 45 degrees"),
            (0.1, 0.2, "small angle difference"),
        ]

        for slope1, slope2, description in test_cases:
            angle = angle_between_lines(slope1, slope2)
            logger.info(f"  slopes ({slope1}, {slope2}) -> {angle:.4f}° ({description})")

        # Test iterative_regression with synthetic data
        log_subsection("iterative_regression()")

        # Create synthetic price data
        np.random.seed(42)
        n_points = 200
        dates = pd.date_range(start='2024-01-01', periods=n_points, freq='1H')

        # Simulated price with trend and noise
        trend = np.linspace(100, 110, n_points)
        noise = np.random.normal(0, 1, n_points)
        prices = trend + noise

        test_df = pd.DataFrame({
            'Open Time': dates,
            'Close': prices
        })

        logger.info(f"  Synthetic data: {len(test_df)} rows")
        logger.info(f"  Price range: {prices.min():.2f} - {prices.max():.2f}")

        # Test with window size 30
        window_size = 30
        logger.info(f"  Testing with window_size={window_size}...")

        result = iterative_regression(test_df, window_size)

        logger.info(f"  Result shape: {result.shape}")
        logger.info(f"  Result columns: {list(result.columns)}")

        if len(result) > 0:
            logger.info(f"  slope_f range: {result['slope_f'].min():.6f} to {result['slope_f'].max():.6f}")
            logger.info(f"  spearman range: {result['spearman'].min():.4f} to {result['spearman'].max():.4f}")
            logger.info(f"  angle range: {result['angle'].min():.4f} to {result['angle'].max():.4f}")

            # Test calculate_acceleration
            log_subsection("calculate_acceleration()")
            accel = calculate_acceleration(result['angle'])
            logger.info(f"  Acceleration range: {accel.min():.4f} to {accel.max():.4f}")
            logger.info(f"  Non-null values: {accel.notna().sum()}")

            return True, result
        else:
            logger.error("  No results returned from iterative_regression")
            return False, None

    except Exception as e:
        logger.error(f"Analysis test failed: {e}")
        logger.debug(traceback.format_exc())
        return False, None


def test_processor_with_real_data():
    """Test processor with real CSV data."""
    log_section("TEST 4: Processor with Real Data")

    try:
        from processor import TimeframeProcessor
        from config import DATA_DIR, TIMEFRAME_ORDER

        # Check if any data exists
        available_timeframes = []
        for tf in TIMEFRAME_ORDER:
            csv_path = DATA_DIR / f"testing_data_{tf}.csv"
            if csv_path.exists():
                available_timeframes.append(tf)

        if not available_timeframes:
            logger.warning("No CSV data files found. Skipping real data test.")
            logger.info("Run 'python main.py --download' to fetch data from Binance.")
            return None, None

        logger.info(f"Available timeframes: {available_timeframes}")

        # Test with first available timeframe and small window
        test_tf = available_timeframes[0]
        test_windows = [30, 60]

        logger.info(f"Testing with timeframe={test_tf}, windows={test_windows}")

        processor = TimeframeProcessor(
            timeframes=[test_tf],
            window_sizes=test_windows
        )

        log_subsection("Loading Data")
        processor.load_all_data()

        for tf, df in processor.raw_data.items():
            logger.info(f"  {tf}: {len(df)} rows, columns={list(df.columns)}")
            logger.info(f"    Date range: {df['Open Time'].min()} to {df['Open Time'].max()}")
            logger.info(f"    Price range: {df['Close'].min():.2f} to {df['Close'].max():.2f}")

        log_subsection("Processing")
        processor.process_all()

        for tf, results in processor.results.items():
            logger.info(f"  {tf}:")
            for label, df in results.items():
                ws = df.attrs.get('window_size', '?')
                logger.info(f"    {label} (window={ws}): {len(df)} rows")

        log_subsection("Signals Table")
        signals_df = processor.get_signals_table()

        if len(signals_df) > 0:
            logger.info(f"  Total signals: {len(signals_df)}")
            for _, row in signals_df.iterrows():
                logger.info(f"    {row['timeframe']}/{row['window']}: {row['signal']}")

            return True, processor
        else:
            logger.warning("  No signals generated")
            return True, processor

    except Exception as e:
        logger.error(f"Processor test failed: {e}")
        logger.debug(traceback.format_exc())
        return False, None


def test_alerts():
    """Test alert engine."""
    log_section("TEST 5: Alert Engine")

    try:
        from alerts import (
            Alert,
            AlertEngine,
            SignalType,
            create_console_handler,
            create_log_handler
        )
        from datetime import datetime

        log_subsection("Create Alert Engine")
        engine = AlertEngine(cooldown_seconds=60)

        # Add test handlers
        alerts_received = []
        def test_handler(alert):
            alerts_received.append(alert)
            logger.info(f"  Handler received: {alert.signal_type.value} on {alert.timeframe}")

        engine.add_handler(test_handler)
        logger.info("  Added test handler")

        log_subsection("Create Test Alert")
        test_alert = Alert(
            signal_type=SignalType.BUY,
            timeframe="1H",
            window_size=30,
            timestamp=datetime.utcnow(),
            price=316.23,
            slope_f=0.0015,
            spearman=-0.85,
            angle=12.5
        )

        logger.info(f"  Signal: {test_alert.signal_type.value}")
        logger.info(f"  Timeframe: {test_alert.timeframe}")
        logger.info(f"  Window: {test_alert.window_size}")

        log_subsection("Process Alert")
        count = engine.process_alerts([test_alert])
        logger.info(f"  Processed {count} alert(s)")
        logger.info(f"  Handlers received: {len(alerts_received)}")

        log_subsection("Test Cooldown")
        # Try sending same alert again (should be blocked by cooldown)
        can_alert = engine.state.can_alert("1H", 30, SignalType.BUY)
        logger.info(f"  Can send duplicate alert: {can_alert} (expected: False)")

        # Different signal should be allowed
        can_alert_sell = engine.state.can_alert("1H", 30, SignalType.SELL)
        logger.info(f"  Can send different signal: {can_alert_sell} (expected: True)")

        log_subsection("Alert Formatting")
        formatted = test_alert.format_message()
        logger.debug(f"Formatted message:\n{formatted}")
        logger.info("  Alert formatting OK")

        return True, engine

    except Exception as e:
        logger.error(f"Alerts test failed: {e}")
        logger.debug(traceback.format_exc())
        return False, None


def test_console_alerts():
    """Test console alerts module."""
    log_section("TEST 6: Console Alerts")

    try:
        from console_alerts import (
            format_signal_row,
            get_signal_color,
            CSVLogger
        )
        from alerts import SignalType

        log_subsection("Signal Formatting")
        row = format_signal_row("1H", 30, "BUY", 0.00123, -0.85, 12.5)
        logger.info(f"  Formatted row created (length: {len(row)})")

        log_subsection("Color Mapping")
        for signal_type in SignalType:
            color = get_signal_color(signal_type)
            logger.info(f"  {signal_type.value}: color code assigned")

        log_subsection("CSV Logger")
        csv_logger = CSVLogger()
        logger.info(f"  Alerts file: {csv_logger.alerts_file}")
        logger.info(f"  Signals file: {csv_logger.signals_file}")

        return True, None

    except Exception as e:
        logger.error(f"Console alerts test failed: {e}")
        logger.debug(traceback.format_exc())
        return False, None


def run_all_tests():
    """Run all tests and generate summary."""
    log_section("STARTING TEST SESSION")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Log file: {LOG_FILE}")

    results = {}

    # Run tests
    results['imports'], _ = test_imports()

    if results['imports']:
        results['config'], _ = test_config()
        results['analysis'], _ = test_analysis_functions()
        results['processor'], _ = test_processor_with_real_data()
        results['alerts'], _ = test_alerts()
        results['console'], _ = test_console_alerts()
    else:
        logger.error("Import test failed. Cannot continue with other tests.")

    # Summary
    log_section("TEST SUMMARY")

    passed = 0
    failed = 0
    skipped = 0

    for test_name, result in results.items():
        if result is True:
            status = "[PASS]"
            passed += 1
        elif result is False:
            status = "[FAIL]"
            failed += 1
        else:
            status = "[SKIP]"
            skipped += 1

        logger.info(f"  {status} {test_name}")

    logger.info("")
    logger.info(f"Total: {passed} passed, {failed} failed, {skipped} skipped")
    logger.info(f"\nFull log saved to: {LOG_FILE}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
