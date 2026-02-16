#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Alert Engine for Cryptocurrency Trading Signal System

Features:
- Signal change detection (new BUY/SELL signals)
- Configurable thresholds
- Alert deduplication (cooldown period)
- Multiple output channels support
"""

from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from pathlib import Path

import pandas as pd

from core.config import (
    ALERT_COOLDOWN_SECONDS,
    ALERT_SIGNALS,
    SIGNAL_NAMES,
    TIMEFRAME_ORDER,
    WINDOW_SIZES,
    get_output_path
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD_UP = "HOLD_UP"
    HOLD_DOWN = "HOLD_DOWN"
    NONE = "NONE"


@dataclass
class Alert:
    """Represents a trading alert."""
    signal_type: SignalType
    timeframe: str
    window_size: int
    timestamp: datetime
    price: Optional[float] = None
    slope_f: Optional[float] = None
    spearman: Optional[float] = None
    angle: Optional[float] = None
    message: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'signal_type': self.signal_type.value,
            'timeframe': self.timeframe,
            'window_size': self.window_size,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'price': self.price,
            'slope_f': self.slope_f,
            'spearman': self.spearman,
            'angle': self.angle,
            'message': self.message
        }

    def format_message(self) -> str:
        """Format the alert as a readable message."""
        # Use ASCII symbols for Windows console compatibility
        emoji = {
            SignalType.BUY: "[BUY]",
            SignalType.SELL: "[SELL]",
            SignalType.HOLD_UP: "[HOLD+]",
            SignalType.HOLD_DOWN: "[HOLD-]",
            SignalType.NONE: "[---]"
        }.get(self.signal_type, "[---]")

        lines = [
            f"{emoji} {self.signal_type.value} Signal",
            f"Timeframe: {self.timeframe}",
            f"Window: {self.window_size}",
        ]

        if self.timestamp:
            lines.append(f"Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        if self.price:
            lines.append(f"Price (sqrt): {self.price:.4f}")

        if self.slope_f is not None:
            lines.append(f"Forward Slope: {self.slope_f:.6f}")

        if self.spearman is not None:
            lines.append(f"Spearman: {self.spearman:.4f}")

        if self.angle is not None:
            lines.append(f"Angle: {self.angle:.2f}°")

        return "\n".join(lines)


@dataclass
class AlertState:
    """Tracks the state of alerts to prevent duplicates."""
    last_alert_time: Dict[str, datetime] = field(default_factory=dict)
    last_signal: Dict[str, SignalType] = field(default_factory=dict)

    def get_key(self, timeframe: str, window_size: int) -> str:
        """Generate a unique key for timeframe+window combination."""
        return f"{timeframe}_{window_size}"

    def can_alert(
        self,
        timeframe: str,
        window_size: int,
        signal_type: SignalType,
        cooldown_seconds: int = ALERT_COOLDOWN_SECONDS
    ) -> bool:
        """
        Check if an alert can be sent (not in cooldown and signal changed).

        Args:
            timeframe: Timeframe label
            window_size: Window size
            signal_type: New signal type
            cooldown_seconds: Minimum time between alerts

        Returns:
            True if alert should be sent
        """
        key = self.get_key(timeframe, window_size)
        now = datetime.utcnow()

        # Check if signal type changed - if changed, always allow alert
        if key in self.last_signal:
            if self.last_signal[key] == signal_type:
                # Same signal - apply cooldown
                if key in self.last_alert_time:
                    elapsed = (now - self.last_alert_time[key]).total_seconds()
                    if elapsed < cooldown_seconds:
                        return False
            # Signal changed - allow alert immediately (no cooldown check)

        return True

    def record_alert(
        self,
        timeframe: str,
        window_size: int,
        signal_type: SignalType
    ) -> None:
        """Record that an alert was sent."""
        key = self.get_key(timeframe, window_size)
        self.last_alert_time[key] = datetime.utcnow()
        self.last_signal[key] = signal_type


class AlertEngine:
    """
    Main alert engine for detecting and dispatching trading signals.
    """

    def __init__(
        self,
        cooldown_seconds: int = ALERT_COOLDOWN_SECONDS,
        alert_on_signals: Optional[List[str]] = None
    ):
        """
        Initialize the alert engine.

        Args:
            cooldown_seconds: Minimum time between duplicate alerts
            alert_on_signals: List of signal column names to alert on
        """
        self.cooldown_seconds = cooldown_seconds
        self.alert_on_signals = alert_on_signals or ALERT_SIGNALS
        self.state = AlertState()
        self.handlers: List[Callable[[Alert], None]] = []
        self.alert_history: List[Alert] = []

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """
        Add an alert handler function.

        Args:
            handler: Function that takes an Alert and processes it
        """
        self.handlers.append(handler)

    def remove_handler(self, handler: Callable[[Alert], None]) -> None:
        """Remove an alert handler."""
        if handler in self.handlers:
            self.handlers.remove(handler)

    def _detect_signal(self, row: pd.Series) -> SignalType:
        """
        Detect the signal type from a data row using slope_f direction.

        Args:
            row: Series with regression output columns

        Returns:
            SignalType enum value
        """
        slope_f = row.get('slope_f', 0)
        if slope_f is None or slope_f == 0:
            return SignalType.NONE
        if slope_f > 0:
            return SignalType.BUY
        elif slope_f < 0:
            return SignalType.SELL
        return SignalType.NONE

    def check_signals(
        self,
        df: pd.DataFrame,
        timeframe: str,
        window_size: int,
        only_last: bool = True
    ) -> List[Alert]:
        """
        Check a DataFrame for alert-worthy signals.

        Args:
            df: DataFrame with signal columns from auto_labeling
            timeframe: Timeframe label
            window_size: Window size used
            only_last: If True, only check the latest row

        Returns:
            List of Alert objects for new signals
        """
        alerts = []

        if len(df) == 0:
            return alerts

        rows_to_check = [df.iloc[-1]] if only_last else df.iterrows()

        for row in rows_to_check:
            if isinstance(row, tuple):
                _, row = row

            signal_type = self._detect_signal(row)

            # Skip NONE signals
            if signal_type == SignalType.NONE:
                continue

            # Check if we can alert (not in cooldown, signal changed)
            if not self.state.can_alert(timeframe, window_size, signal_type, self.cooldown_seconds):
                continue

            # Create alert
            alert = Alert(
                signal_type=signal_type,
                timeframe=timeframe,
                window_size=window_size,
                timestamp=pd.to_datetime(row.get('time')).to_pydatetime() if 'time' in row else datetime.utcnow(),
                price=row.get('actual'),
                slope_f=row.get('slope_f'),
                spearman=row.get('spearman'),
                angle=row.get('angle')
            )
            alert.message = alert.format_message()

            alerts.append(alert)

        return alerts

    def process_alerts(self, alerts: List[Alert]) -> int:
        """
        Process alerts through all handlers.

        Args:
            alerts: List of Alert objects

        Returns:
            Number of alerts processed
        """
        count = 0

        for alert in alerts:
            # Record the alert
            self.state.record_alert(alert.timeframe, alert.window_size, alert.signal_type)
            self.alert_history.append(alert)

            # Dispatch to handlers
            for handler in self.handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler error: {e}")

            count += 1

        return count

    def check_and_alert(
        self,
        results: Dict[str, Dict[str, pd.DataFrame]]
    ) -> List[Alert]:
        """
        Check all results for signals and send alerts.

        Args:
            results: Nested dict from TimeframeProcessor.results

        Returns:
            List of all alerts generated
        """
        all_alerts = []

        for timeframe, tf_results in results.items():
            for label, df in tf_results.items():
                window_size = df.attrs.get('window_size', 0)

                alerts = self.check_signals(df, timeframe, window_size)

                if alerts:
                    self.process_alerts(alerts)
                    all_alerts.extend(alerts)

        return all_alerts

    def get_alert_history(self, limit: int = 100) -> List[dict]:
        """
        Get recent alert history.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of alert dictionaries
        """
        return [a.to_dict() for a in self.alert_history[-limit:]]

    def save_history(self, filepath: Optional[Path] = None) -> Path:
        """
        Save alert history to JSON file.

        Args:
            filepath: Output file path

        Returns:
            Path to saved file
        """
        filepath = filepath or get_output_path("alert_history.json")

        with open(filepath, 'w') as f:
            json.dump(self.get_alert_history(), f, indent=2, default=str)

        logger.info(f"Saved alert history to {filepath}")
        return filepath

    def load_history(self, filepath: Optional[Path] = None) -> int:
        """
        Load alert history from JSON file.

        Args:
            filepath: Input file path

        Returns:
            Number of alerts loaded
        """
        filepath = filepath or get_output_path("alert_history.json")

        if not filepath.exists():
            return 0

        with open(filepath, 'r') as f:
            data = json.load(f)

        for item in data:
            alert = Alert(
                signal_type=SignalType(item['signal_type']),
                timeframe=item['timeframe'],
                window_size=item['window_size'],
                timestamp=datetime.fromisoformat(item['timestamp']) if item['timestamp'] else None,
                price=item.get('price'),
                slope_f=item.get('slope_f'),
                spearman=item.get('spearman'),
                angle=item.get('angle'),
                message=item.get('message', '')
            )
            self.alert_history.append(alert)

        return len(data)


def create_console_handler() -> Callable[[Alert], None]:
    """Create a simple console alert handler."""
    def handler(alert: Alert) -> None:
        print("\n" + "=" * 50)
        print(alert.format_message())
        print("=" * 50 + "\n")

    return handler


def create_log_handler() -> Callable[[Alert], None]:
    """Create a logging alert handler."""
    def handler(alert: Alert) -> None:
        logger.info(f"ALERT: {alert.signal_type.value} on {alert.timeframe}/{alert.window_size}")

    return handler


if __name__ == "__main__":
    # Demo the alert engine
    print("Alert Engine Demo")
    print("=" * 50)

    engine = AlertEngine()
    engine.add_handler(create_console_handler())
    engine.add_handler(create_log_handler())

    # Create a test alert
    test_alert = Alert(
        signal_type=SignalType.BUY,
        timeframe="1H",
        window_size=30,
        timestamp=datetime.utcnow(),
        price=316.23,  # sqrt of ~100000
        slope_f=0.0015,
        spearman=-0.85,
        angle=12.5
    )
    test_alert.message = test_alert.format_message()

    print("\nTest Alert:")
    engine.process_alerts([test_alert])

    print(f"\nAlert history: {len(engine.alert_history)} alerts")
