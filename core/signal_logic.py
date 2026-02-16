#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Signal Logic Module for Cryptocurrency Trading Signal System

Implements the user story signal generation logic:
1. Reversal Detection - 5-point pattern: [a>b>c>d<e] bottom, [a<b<c<d>e] peak
2. Direction - Forward slope (slope_f) determines LONG/SHORT
3. Acceleration Segmentation - GMM clustering (5 clusters), distant/very_distant = quality
4. Angle Crossings - Like MA crossovers between window sizes
5. Convergence Scoring - Count-based signal quality across 55 signals (11 TF × 5 windows)
6. Prediction - linear extrapolation for 2-step forward prediction (cc, ccc)
7. Calendar Table - Signal convergence visualization with predictions

Signal Formula (from user story):
    signal = reversal_event + GMM_quality + direction(slope_f)

Quality Filter:
- High quality signals: distant/very_distant zone + reversal event + clear direction
- Low quality / False positives: other GMM zones

Crossing signals:
- df up, df1 down = SHORT (smaller window above larger)
- df down, df1 up = LONG (smaller window below larger)

Higher timeframe = higher impact weight
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from sklearn.mixture import GaussianMixture
import logging
import warnings

warnings.filterwarnings('ignore')

from core.processor import TimeframeProcessor
from core.config import TIMEFRAME_ORDER, WINDOW_SIZES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """Signal direction types."""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class AccelerationZone(Enum):
    """GMM-based acceleration zones."""
    VERY_DISTANT = "VERY_DISTANT"  # Strong signal zone
    DISTANT = "DISTANT"            # Good signal zone
    BASELINE = "BASELINE"          # Neutral zone
    CLOSE = "CLOSE"                # Weak zone
    VERY_CLOSE = "VERY_CLOSE"      # Very weak zone


class ReversalType(Enum):
    """Types of reversal signals based on 5-point pattern detection."""
    PEAK = "PEAK"           # Pattern [a<b<c<d>e] - increasing then reversal down
    BOTTOM = "BOTTOM"       # Pattern [a>b>c>d<e] - decreasing then reversal up
    NONE = "NONE"


class PricePredictor:
    """
    Prediction module using simple linear extrapolation.

    Uses last two values and their difference for 2-step forward prediction.
    Formula: next = b + (b - a), where a = previous, b = current.
    Two-step: cc = b + (b-a), ccc = cc + (cc-b).
    """

    @staticmethod
    def predict_next(a: float, b: float) -> float:
        """
        Simple linear extrapolation from last two values.

        next = b + (b - a)

        Args:
            a: Previous value
            b: Current value

        Returns:
            Predicted next value
        """
        return b + (b - a)

    @classmethod
    def predict_forward(cls, series: pd.Series, steps: int = 2) -> List[float]:
        """
        Two-step prediction: cc, ccc.

        Args:
            series: Input series with at least 2 values
            steps: Number of prediction steps (default 2)

        Returns:
            List of predicted values [cc, ccc]
        """
        values = series.dropna().values if hasattr(series, 'dropna') else np.array(series)
        if len(values) < 2:
            return []

        a, b = values[-2], values[-1]

        predictions = []
        for _ in range(steps):
            next_val = cls.predict_next(a, b)
            predictions.append(next_val)
            a, b = b, next_val

        return predictions

    @classmethod
    def predict_all_metrics(cls, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Predict forward values for angle, slope_f, and acceleration.

        Args:
            df: DataFrame with angle, slope_f, acceleration columns

        Returns:
            Dict with predictions for each metric: {'angle': [cc, ccc], 'slope_f': [...], 'acceleration': [...]}
        """
        predictions = {}

        for col in ['angle', 'slope_f', 'acceleration']:
            if col in df.columns:
                predictions[col] = cls.predict_forward(df[col])
            else:
                predictions[col] = []

        return predictions


@dataclass
class Signal:
    """Represents a trading signal."""
    direction: SignalDirection
    strength: float  # 0.0 to 1.0
    timeframe: str
    window_size: int
    timestamp: datetime
    reversal_type: ReversalType
    acceleration_zone: AccelerationZone
    is_crossing: bool = False
    crossing_windows: Tuple[int, int] = None
    details: Dict = field(default_factory=dict)

    def __str__(self):
        cross_info = f" [Cross {self.crossing_windows}]" if self.is_crossing else ""
        return (f"{self.direction.value} | {self.timeframe}/{self.window_size} | "
                f"Strength: {self.strength:.2f} | {self.reversal_type.value} | "
                f"{self.acceleration_zone.value}{cross_info}")


@dataclass
class AggregatedSignal:
    """Aggregated signal across all timeframes."""
    final_direction: SignalDirection
    convergence_score: int  # Number of agreeing signals
    total_signals: int
    long_count: int
    short_count: int
    neutral_count: int
    quality: str  # LOW, MEDIUM, HIGH, VERY_HIGH
    signals: List[Signal]
    timestamp: datetime

    def __str__(self):
        return (f"{self.final_direction.value} | Score: {self.convergence_score}/{self.total_signals} | "
                f"Quality: {self.quality} | L:{self.long_count} S:{self.short_count} N:{self.neutral_count}")


class SignalLogic:
    """
    Advanced signal logic implementing custom trading rules.
    """

    # Timeframe weights (higher = more impact)
    TIMEFRAME_WEIGHTS = {
        "3D": 5, "1D": 4, "12H": 3.5, "8H": 3, "6H": 2.5,
        "4H": 2, "2H": 1.5, "1H": 1.2, "30M": 1.1, "15M": 1.0, "5M": 0.8
    }

    # Window crossing pairs (adjacent + neighbor-of-neighbor)
    CROSSING_PAIRS = [
        (30, 60),   # df - df1 (adjacent)
        (30, 100),  # df - df2 (skip one)
        (60, 100),  # df1 - df2 (adjacent)
        (60, 120),  # df1 - df3 (skip one)
        (100, 120), # df2 - df3 (adjacent)
        (100, 160), # df2 - df4 (skip one)
        (120, 160), # df3 - df4 (adjacent)
    ]

    def __init__(
        self,
        n_gmm_clusters: int = 5,
        zero_threshold: float = 2.0,  # Degrees - threshold for "near zero"
        peak_order: int = 3,  # Points on each side for peak detection
        processor: Optional[TimeframeProcessor] = None  # Pre-loaded processor to avoid duplicate loading
    ):
        """
        Initialize signal logic.

        Args:
            n_gmm_clusters: Number of GMM clusters for acceleration
            zero_threshold: Threshold in degrees for zero crossing detection
            peak_order: Order for local extrema detection
            processor: Optional pre-loaded TimeframeProcessor to reuse (avoids duplicate data loading)
        """
        self.n_gmm_clusters = n_gmm_clusters
        self.zero_threshold = zero_threshold
        self.peak_order = peak_order

        self.processor: Optional[TimeframeProcessor] = processor
        self.gmm_models: Dict[str, GaussianMixture] = {}
        self.signals: List[Signal] = []

    def detect_cycle_events(self, angles: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Detect reversal events using 5-point pattern detection.

        Based on user story specification (lines 87-106):
        - Bottom: [a>b>c>d<e] - decreasing then reversal up
        - Peak: [a<b<c<d>e] - increasing then reversal down

        Args:
            angles: Array of angle values
            window: Window size for pattern detection (default 5)

        Returns:
            Array where:
            - 1 = Event detected (PEAK or BOTTOM - matching user story)
            - 0 = No event
        """
        events = np.zeros(len(angles))

        for i in range(len(angles) - window + 1):
            window_vals = angles[i:i + window]

            # Check for strictly monotonic (no reversal event)
            if all(window_vals[j] > window_vals[j + 1] for j in range(window - 1)) or \
               all(window_vals[j] < window_vals[j + 1] for j in range(window - 1)):
                events[i + window - 1] = 0
            else:
                # Bottom: [a>b>c>d<e] - decreasing then reversal up (user story line 101)
                if (window_vals[0] > window_vals[1] > window_vals[2] > window_vals[3]
                    and window_vals[3] < window_vals[4]):
                    events[i + window - 1] = 1  # Bottom detected
                # Peak: [a<b<c<d>e] - increasing then reversal down (user story line 103)
                elif (window_vals[0] < window_vals[1] < window_vals[2] < window_vals[3]
                      and window_vals[3] > window_vals[4]):
                    events[i + window - 1] = 1  # Peak detected (matching user story: returns 1 for both)

        return events

    def get_reversal_type_at_index(self, events: np.ndarray, idx: int, angles: np.ndarray = None) -> ReversalType:
        """
        Get reversal type at a specific index from events array.

        Args:
            events: Array from detect_cycle_events (1 = event detected)
            idx: Index to check
            angles: Optional angle array to determine PEAK vs BOTTOM from pattern

        Returns:
            ReversalType enum (PEAK or BOTTOM based on last 5 angle values pattern)
        """
        if idx < 0 or idx >= len(events):
            return ReversalType.NONE

        event_val = events[idx]
        if event_val == 1:
            # Determine if PEAK or BOTTOM by examining the pattern
            if angles is not None and idx >= 4:
                window_vals = angles[idx-4:idx+1]
                # Bottom: [a>b>c>d<e] - decreasing then reversal up
                if (window_vals[0] > window_vals[1] > window_vals[2] > window_vals[3]
                    and window_vals[3] < window_vals[4]):
                    return ReversalType.BOTTOM
                # Peak: [a<b<c<d>e] - increasing then reversal down
                elif (window_vals[0] < window_vals[1] < window_vals[2] < window_vals[3]
                      and window_vals[3] > window_vals[4]):
                    return ReversalType.PEAK
            # Default to BOTTOM if can't determine (event was detected)
            return ReversalType.BOTTOM
        else:
            return ReversalType.NONE

    def fit_gmm_acceleration(self, acceleration: np.ndarray, key: str) -> GaussianMixture:
        """
        Fit GMM model to acceleration data for segmentation.

        Args:
            acceleration: Array of acceleration values
            key: Key for caching the model

        Returns:
            Fitted GaussianMixture model
        """
        # Remove NaN values
        valid_accel = acceleration[~np.isnan(acceleration)].reshape(-1, 1)

        if len(valid_accel) < self.n_gmm_clusters:
            logger.warning(f"Not enough data for GMM: {len(valid_accel)} points")
            return None

        gmm = GaussianMixture(
            n_components=self.n_gmm_clusters,
            covariance_type='full',
            random_state=42,
            n_init=3
        )
        gmm.fit(valid_accel)

        self.gmm_models[key] = gmm
        return gmm

    def classify_acceleration(self, value: float, gmm: GaussianMixture) -> AccelerationZone:
        """
        Classify acceleration value into zone using GMM.

        Args:
            value: Acceleration value
            gmm: Fitted GMM model

        Returns:
            AccelerationZone enum
        """
        if gmm is None or np.isnan(value):
            return AccelerationZone.BASELINE

        # Get cluster assignment
        cluster = gmm.predict([[value]])[0]

        # Sort cluster means by absolute value (distance from zero)
        means = gmm.means_.flatten()
        abs_means = np.abs(means)
        sorted_indices = np.argsort(abs_means)

        # Map cluster to zone based on distance from zero
        cluster_rank = np.where(sorted_indices == cluster)[0][0]

        zone_map = {
            0: AccelerationZone.VERY_CLOSE,
            1: AccelerationZone.CLOSE,
            2: AccelerationZone.BASELINE,
            3: AccelerationZone.DISTANT,
            4: AccelerationZone.VERY_DISTANT
        }

        return zone_map.get(cluster_rank, AccelerationZone.BASELINE)

    def detect_angle_crossings(
        self,
        angles_dict: Dict[int, pd.Series],
        times: pd.Series
    ) -> List[Dict]:
        """
        Detect crossings between angle series of different window sizes.

        Args:
            angles_dict: Dict mapping window_size to angle series
            times: Time series for timestamps

        Returns:
            List of crossing events
        """
        crossings = []

        for ws1, ws2 in self.CROSSING_PAIRS:
            if ws1 not in angles_dict or ws2 not in angles_dict:
                continue

            angles1 = angles_dict[ws1].values
            angles2 = angles_dict[ws2].values
            times_arr = times.values

            # Align series (use shorter length of all three)
            min_len = min(len(angles1), len(angles2), len(times_arr))
            angles1 = angles1[-min_len:]
            angles2 = angles2[-min_len:]
            aligned_times = times_arr[-min_len:]

            # Detect crossings
            for i in range(1, min_len):
                prev_diff = angles1[i-1] - angles2[i-1]
                curr_diff = angles1[i] - angles2[i]

                # Crossing detected when sign changes
                if prev_diff * curr_diff < 0:
                    # Determine direction based on user spec:
                    # "df up, df1 down" = SHORT (smaller window above larger)
                    # "df down, df1 up" = LONG (smaller window below larger)
                    if prev_diff < 0 and curr_diff > 0:
                        # ws1 was below ws2, now above = SHORT signal (df crossing above df1)
                        direction = SignalDirection.SHORT
                    else:
                        # ws1 was above ws2, now below = LONG signal (df crossing below df1)
                        direction = SignalDirection.LONG

                    crossings.append({
                        'index': i,
                        'time': aligned_times[i],
                        'windows': (ws1, ws2),
                        'direction': direction,
                        'angle_diff': curr_diff
                    })

        return crossings

    def get_direction_from_slope(self, slope_f: float) -> SignalDirection:
        """
        Determine direction from forward slope.

        Args:
            slope_f: Forward slope value

        Returns:
            SignalDirection
        """
        if slope_f > 0.01:  # Positive slope threshold
            return SignalDirection.LONG
        elif slope_f < -0.01:  # Negative slope threshold
            return SignalDirection.SHORT
        else:
            return SignalDirection.NEUTRAL

    def analyze_timeframe(
        self,
        timeframe: str,
        results: Dict[str, pd.DataFrame]
    ) -> List[Signal]:
        """
        Analyze a single timeframe across all window sizes.

        Args:
            timeframe: Timeframe label
            results: Dict of DataFrames by window label

        Returns:
            List of signals generated
        """
        signals = []

        # Collect angles from all windows
        angles_dict = {}
        slopes_dict = {}
        accels_dict = {}
        times_dict = {}

        for label, df in results.items():
            ws = df.attrs.get('window_size', 30)
            angles_dict[ws] = df['angle']
            slopes_dict[ws] = df['slope_f']
            accels_dict[ws] = df['acceleration']
            times_dict[ws] = df['time']

        # Fit GMM on combined acceleration data
        all_accel = np.concatenate([a.dropna().values for a in accels_dict.values()])
        gmm_key = f"{timeframe}_accel"
        gmm = self.fit_gmm_acceleration(all_accel, gmm_key)

        # Analyze each window size
        for ws, angles in angles_dict.items():
            # Need at least 5 points for reversal detection
            if len(angles) < 5:
                continue

            angle_arr = angles.values
            slope_arr = slopes_dict[ws].values
            accel_arr = accels_dict[ws].values
            time_arr = times_dict[ws].values

            # Detect reversals using 5-point pattern (user story spec)
            reversal_events = self.detect_cycle_events(angle_arr, window=5)

            # Get latest values
            latest_idx = len(angle_arr) - 1
            latest_slope = slope_arr[latest_idx]
            latest_accel = accel_arr[latest_idx] if not np.isnan(accel_arr[latest_idx]) else 0
            latest_time = pd.to_datetime(time_arr[latest_idx])

            # Classify acceleration zone using GMM
            accel_zone = self.classify_acceleration(latest_accel, gmm)

            # Get reversal type at latest point (pass angles to determine PEAK vs BOTTOM)
            reversal = self.get_reversal_type_at_index(reversal_events, latest_idx, angle_arr)

            # Determine direction from forward slope (user story spec)
            direction = self.get_direction_from_slope(latest_slope)

            # Signal quality filter: Good signals require distant/very_distant zone + reversal
            is_high_quality = (
                accel_zone in [AccelerationZone.DISTANT, AccelerationZone.VERY_DISTANT]
                and reversal != ReversalType.NONE
            )

            # Calculate signal strength based on new formula
            strength = self._calculate_strength(accel_zone, reversal, direction)

            # Get predictions for this window
            df_window = results.get(f"df{WINDOW_SIZES.index(ws)}" if ws != 30 else "df")
            predictions = {}
            if df_window is not None:
                predictions = PricePredictor.predict_all_metrics(df_window)

            # Create signal
            signal = Signal(
                direction=direction,
                strength=strength,
                timeframe=timeframe,
                window_size=ws,
                timestamp=latest_time,
                reversal_type=reversal,
                acceleration_zone=accel_zone,
                details={
                    'slope_f': latest_slope,
                    'angle': angle_arr[latest_idx],
                    'acceleration': latest_accel,
                    'is_high_quality': is_high_quality,
                    'predictions': predictions
                }
            )
            signals.append(signal)

        # Detect angle crossings (use shortest time series)
        shortest_ws = min(times_dict.keys(), key=lambda k: len(times_dict[k]))
        crossings = self.detect_angle_crossings(angles_dict, times_dict[shortest_ws])

        # Add crossing signals (most recent only) with timeframe weighting
        if crossings:
            latest_crossing = crossings[-1]
            # Apply timeframe weight to crossing strength (base 0.6, max ~1.0 for 3D)
            tf_weight = self.TIMEFRAME_WEIGHTS.get(timeframe, 1.0)
            weighted_strength = min(0.6 + (tf_weight / 10), 1.0)

            cross_signal = Signal(
                direction=latest_crossing['direction'],
                strength=weighted_strength,  # Weighted by timeframe importance
                timeframe=timeframe,
                window_size=latest_crossing['windows'][0],
                timestamp=pd.to_datetime(latest_crossing['time']),
                reversal_type=ReversalType.NONE,
                acceleration_zone=AccelerationZone.DISTANT,
                is_crossing=True,
                crossing_windows=latest_crossing['windows'],
                details={
                    'angle_diff': latest_crossing['angle_diff'],
                    'tf_weight': tf_weight
                }
            )
            signals.append(cross_signal)

        return signals

    def _calculate_strength(
        self,
        accel_zone: AccelerationZone,
        reversal: ReversalType,
        direction: SignalDirection
    ) -> float:
        """
        Calculate signal strength based on user story formula:
        signal = reversal_event + GMM_quality + direction(slope_f)

        High quality signals: distant/very_distant + reversal + clear direction
        """
        strength = 0.0

        # Acceleration zone contribution (0.0 - 0.4)
        # Good signals require distant/very_distant (per user story)
        zone_scores = {
            AccelerationZone.VERY_DISTANT: 0.4,
            AccelerationZone.DISTANT: 0.3,
            AccelerationZone.BASELINE: 0.1,
            AccelerationZone.CLOSE: 0.05,
            AccelerationZone.VERY_CLOSE: 0.0
        }
        strength += zone_scores.get(accel_zone, 0.1)

        # Reversal contribution (0.0 - 0.4)
        # Using 5-point pattern: PEAK and BOTTOM
        reversal_scores = {
            ReversalType.PEAK: 0.4,    # [a<b<c<d>e] pattern
            ReversalType.BOTTOM: 0.4,  # [a>b>c>d<e] pattern
            ReversalType.NONE: 0.0
        }
        strength += reversal_scores.get(reversal, 0.0)

        # Direction clarity (0.0 - 0.2)
        # Based on forward slope direction
        if direction != SignalDirection.NEUTRAL:
            strength += 0.2

        return min(strength, 1.0)

    def aggregate_signals(self, signals: List[Signal]) -> AggregatedSignal:
        """
        Aggregate signals using count-based scoring.

        Args:
            signals: List of all signals

        Returns:
            AggregatedSignal with final decision
        """
        if not signals:
            return AggregatedSignal(
                final_direction=SignalDirection.NEUTRAL,
                convergence_score=0,
                total_signals=0,
                long_count=0,
                short_count=0,
                neutral_count=0,
                quality="NONE",
                signals=[],
                timestamp=datetime.utcnow()
            )

        # Count by direction (weighted by timeframe)
        long_score = 0
        short_score = 0
        neutral_count = 0

        for signal in signals:
            weight = self.TIMEFRAME_WEIGHTS.get(signal.timeframe, 1.0)
            weighted_strength = signal.strength * weight

            if signal.direction == SignalDirection.LONG:
                long_score += weighted_strength
            elif signal.direction == SignalDirection.SHORT:
                short_score += weighted_strength
            else:
                neutral_count += 1

        # Determine final direction
        long_count = sum(1 for s in signals if s.direction == SignalDirection.LONG)
        short_count = sum(1 for s in signals if s.direction == SignalDirection.SHORT)

        if long_score > short_score and long_count > short_count:
            final_direction = SignalDirection.LONG
            convergence_score = long_count
        elif short_score > long_score and short_count > long_count:
            final_direction = SignalDirection.SHORT
            convergence_score = short_count
        else:
            final_direction = SignalDirection.NEUTRAL
            convergence_score = neutral_count

        # Determine quality based on convergence
        total = len(signals)
        ratio = convergence_score / total if total > 0 else 0

        if ratio >= 0.7:
            quality = "VERY_HIGH"
        elif ratio >= 0.5:
            quality = "HIGH"
        elif ratio >= 0.3:
            quality = "MEDIUM"
        else:
            quality = "LOW"

        return AggregatedSignal(
            final_direction=final_direction,
            convergence_score=convergence_score,
            total_signals=total,
            long_count=long_count,
            short_count=short_count,
            neutral_count=neutral_count,
            quality=quality,
            signals=signals,
            timestamp=datetime.utcnow()
        )

    def run_analysis(self) -> AggregatedSignal:
        """
        Run full analysis across all timeframes.

        Returns:
            AggregatedSignal with final result
        """
        # Reuse existing processor if provided, otherwise create new one
        if self.processor is None:
            logger.info("Loading and processing data...")
            self.processor = TimeframeProcessor()
            self.processor.load_all_data()
            self.processor.process_all()
        else:
            logger.info("Reusing existing processor (data already loaded)")

        logger.info("Analyzing signals...")
        all_signals = []

        for timeframe in TIMEFRAME_ORDER:
            if timeframe not in self.processor.results:
                continue

            tf_signals = self.analyze_timeframe(timeframe, self.processor.results[timeframe])
            all_signals.extend(tf_signals)
            logger.info(f"  {timeframe}: {len(tf_signals)} signals")

        self.signals = all_signals
        logger.info(f"Total signals: {len(all_signals)}")

        # Aggregate
        aggregated = self.aggregate_signals(all_signals)

        return aggregated

    def get_report(self, aggregated: AggregatedSignal) -> str:
        """Generate detailed report."""
        lines = [
            "=" * 70,
            "ADVANCED SIGNAL ANALYSIS REPORT",
            "=" * 70,
            f"Timestamp: {aggregated.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            f"FINAL SIGNAL: {aggregated.final_direction.value}",
            f"Quality: {aggregated.quality}",
            f"Convergence: {aggregated.convergence_score}/{aggregated.total_signals}",
            "",
            f"Direction Counts:",
            f"  LONG:    {aggregated.long_count}",
            f"  SHORT:   {aggregated.short_count}",
            f"  NEUTRAL: {aggregated.neutral_count}",
            "",
            "=" * 70,
            "SIGNALS BY TIMEFRAME",
            "=" * 70,
        ]

        # Group by timeframe
        for tf in TIMEFRAME_ORDER:
            tf_signals = [s for s in aggregated.signals if s.timeframe == tf]
            if not tf_signals:
                continue

            lines.append(f"\n{tf} (weight: {self.TIMEFRAME_WEIGHTS.get(tf, 1.0)}x):")

            for sig in tf_signals:
                cross = " [CROSSING]" if sig.is_crossing else ""
                lines.append(
                    f"  w{sig.window_size}: {sig.direction.value:7} | "
                    f"Str: {sig.strength:.2f} | {sig.reversal_type.value:12} | "
                    f"{sig.acceleration_zone.value}{cross}"
                )

        # Strong signals summary
        strong_signals = [s for s in aggregated.signals if s.strength >= 0.5]
        if strong_signals:
            lines.append("\n" + "=" * 70)
            lines.append("STRONG SIGNALS (strength >= 0.5)")
            lines.append("=" * 70)
            for sig in strong_signals:
                lines.append(f"  {sig}")

        # Crossing signals
        crossing_signals = [s for s in aggregated.signals if s.is_crossing]
        if crossing_signals:
            lines.append("\n" + "=" * 70)
            lines.append("CROSSING SIGNALS")
            lines.append("=" * 70)
            for sig in crossing_signals:
                lines.append(f"  {sig.timeframe}: {sig.crossing_windows} -> {sig.direction.value}")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)


class CalendarDataBuilder:
    """
    Builds the signal prediction calendar table for visualization.

    Calendar format (from user story):
    - Rows: timeframe-window (55 combinations: 11 TF × 5 windows)
    - Columns: Rolling last 3 hours in 15min intervals + 4h forward prediction
    - Cells: Signal + convergence count + color coding
    """

    def __init__(
        self,
        history_hours: int = 3,
        forward_hours: int = 4,
        interval_minutes: int = 15
    ):
        """
        Initialize calendar builder.

        Args:
            history_hours: Hours of historical data to show
            forward_hours: Hours of forward prediction
            interval_minutes: Time interval in minutes
        """
        self.history_hours = history_hours
        self.forward_hours = forward_hours
        self.interval_minutes = interval_minutes

    def get_time_columns(self, now: datetime) -> List[datetime]:
        """
        Generate time columns from -3h to +4h.

        Args:
            now: Current datetime

        Returns:
            List of datetime objects for column headers
        """
        times = []
        start = now - timedelta(hours=self.history_hours)
        end = now + timedelta(hours=self.forward_hours)

        current = start
        while current <= end:
            times.append(current)
            current += timedelta(minutes=self.interval_minutes)

        return times

    def get_row_labels(self) -> List[str]:
        """
        Generate row labels for all timeframe-window combinations.

        Returns:
            List of 55 row labels like "15M-df", "30M-df1", etc.
        """
        labels = []
        window_labels = ["df", "df1", "df2", "df3", "df4"]

        for tf in TIMEFRAME_ORDER:
            for wl in window_labels:
                labels.append(f"{tf}-{wl}")

        return labels

    def _get_direction_from_slope(self, slope_f: float) -> str:
        """Get direction character from slope value."""
        if slope_f > 0.01:
            return "L"  # LONG
        elif slope_f < -0.01:
            return "S"  # SHORT
        else:
            return "N"  # NEUTRAL

    def _find_data_at_time(
        self,
        df: pd.DataFrame,
        target_time: datetime,
        tolerance_minutes: int = 60
    ) -> Optional[pd.Series]:
        """Find data row nearest to target time within tolerance."""
        if df is None or 'time' not in df.columns or len(df) == 0:
            return None

        # Convert to pandas timestamps (timezone-naive for comparison)
        df_times = pd.to_datetime(df['time']).dt.tz_localize(None)
        target_ts = pd.Timestamp(target_time)
        if target_ts.tzinfo is not None:
            target_ts = target_ts.tz_localize(None)

        # Find nearest time
        time_diffs = abs(df_times - target_ts)
        min_idx = time_diffs.idxmin()

        try:
            min_diff = time_diffs.loc[min_idx]
            min_diff_minutes = min_diff.total_seconds() / 60
        except Exception:
            min_diff_minutes = float('inf')

        if min_diff_minutes <= tolerance_minutes:
            return df.loc[min_idx]
        return None

    def _get_data_time_range(self, processor_results: Dict) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get the actual time range from the data."""
        min_time = None
        max_time = None

        if not processor_results:
            return None, None

        for tf in processor_results:
            for ws_label in processor_results[tf]:
                df = processor_results[tf][ws_label]
                if df is not None and 'time' in df.columns and len(df) > 0:
                    df_times = pd.to_datetime(df['time'])
                    if min_time is None or df_times.min() < min_time:
                        min_time = df_times.min()
                    if max_time is None or df_times.max() > max_time:
                        max_time = df_times.max()

        return min_time, max_time

    def _detect_reversal_at_index(self, angles: np.ndarray, idx: int, window: int = 5) -> str:
        """Detect reversal at specific index using 5-point pattern."""
        if idx < window - 1 or idx >= len(angles):
            return ""

        start_idx = idx - window + 1
        window_vals = angles[start_idx:idx + 1]

        if len(window_vals) < window:
            return ""

        # Bottom: [a>b>c>d<e] - decreasing then reversal up
        if (window_vals[0] > window_vals[1] > window_vals[2] > window_vals[3]
            and window_vals[3] < window_vals[4]):
            return "*"  # Bottom detected (potential LONG)

        # Peak: [a<b<c<d>e] - increasing then reversal down
        if (window_vals[0] < window_vals[1] < window_vals[2] < window_vals[3]
            and window_vals[3] > window_vals[4]):
            return "*"  # Peak detected (potential SHORT)

        return ""

    def _predict_angle_crossings(
        self,
        processor_results: Dict,
        timeframe: str,
        steps: int = 2
    ) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Predict angle crossings between window pairs for a timeframe.

        The "Mistique Oracul Concept" - predicting when angles from different windows will cross.

        Args:
            processor_results: Dict of processor results
            timeframe: Timeframe to analyze
            steps: Number of prediction steps (cc, ccc)

        Returns:
            Dict with step number as key and list of (window1, window2, direction) crossings
        """
        crossings = {f"step_{i+1}": [] for i in range(steps)}

        if not processor_results or timeframe not in processor_results:
            return crossings

        window_labels = ["df", "df1", "df2", "df3", "df4"]
        window_sizes = [30, 60, 100, 120, 160]
        size_to_label = dict(zip(window_sizes, window_labels))

        # Get current and predicted angles for each window
        angle_predictions = {}
        for ws_label in window_labels:
            if ws_label not in processor_results[timeframe]:
                continue
            df = processor_results[timeframe][ws_label]
            if 'angle' not in df.columns or len(df) < 2:
                continue

            angles = df['angle'].dropna().values
            if len(angles) < 2:
                continue

            a, b = angles[-2], angles[-1]
            # Predict cc and ccc using simple linear extrapolation
            cc = PricePredictor.predict_next(a, b)
            ccc = PricePredictor.predict_next(b, cc)
            angle_predictions[ws_label] = {
                'current': b,
                'cc': cc,
                'ccc': ccc
            }

        # Use the full CROSSING_PAIRS from SignalLogic (converted to labels)
        # Pairs: (30,60), (30,100), (60,120), (100,160), (60,100), (100,120), (120,160)
        crossing_pairs_sizes = [
            (30, 60),    # df - df1
            (30, 100),   # df - df2
            (60, 100),   # df1 - df2
            (60, 120),   # df1 - df3
            (100, 120),  # df2 - df3
            (100, 160),  # df2 - df4
            (120, 160),  # df3 - df4
        ]

        for ws_size1, ws_size2 in crossing_pairs_sizes:
            ws1 = size_to_label.get(ws_size1)
            ws2 = size_to_label.get(ws_size2)

            if not ws1 or not ws2:
                continue
            if ws1 not in angle_predictions or ws2 not in angle_predictions:
                continue

            pred1 = angle_predictions[ws1]
            pred2 = angle_predictions[ws2]

            # Check for crossing at step 1 (current → cc)
            curr_diff = pred1['current'] - pred2['current']
            step1_diff = pred1['cc'] - pred2['cc']

            if curr_diff * step1_diff < 0:  # Sign change = crossing
                # Determine direction: smaller window crossing above larger = LONG
                # (df up relative to df1 = bullish divergence)
                direction = "L" if step1_diff > 0 else "S"
                crossings["step_1"].append((ws1, ws2, direction))

            # Check for crossing at step 2 (cc → ccc)
            step2_diff = pred1['ccc'] - pred2['ccc']

            if step1_diff * step2_diff < 0:  # Sign change = crossing
                direction = "L" if step2_diff > 0 else "S"
                crossings["step_2"].append((ws1, ws2, direction))

        return crossings

    def _compute_cell_score(
        self,
        has_reversal: bool,
        has_crossing: bool,
        accel_quality: bool,
        direction: str
    ) -> str:
        """
        Compute cell score based on user story requirements.

        Score system:
        - Angle reversal = +1 point
        - Angle crossing = +1 point
        - Acceleration quality (distant/very_distant) = +1 point
        - Max = 3 points

        Returns: "direction:score:flags" e.g., "L:3:RCA", "S:2:RC", "N:0:"
        Flags: R=Reversal, C=Crossing, A=Accel Quality
        """
        score = 0
        flags = ""
        if has_reversal:
            score += 1
            flags += "R"
        if has_crossing:
            score += 1
            flags += "C"
        if accel_quality:
            score += 1
            flags += "A"

        return f"{direction}:{score}:{flags}"

    def _check_acceleration_quality(
        self,
        accel_value: float,
        all_accels: np.ndarray
    ) -> bool:
        """
        Check if acceleration is in distant/very_distant zone (top 40% by absolute value).
        Simplified GMM approximation using percentiles.
        """
        if np.isnan(accel_value) or len(all_accels) == 0:
            return False

        abs_accel = abs(accel_value)
        abs_all = np.abs(all_accels[~np.isnan(all_accels)])

        if len(abs_all) == 0:
            return False

        # Top 40% by absolute value = distant/very_distant
        threshold = np.percentile(abs_all, 60)
        return abs_accel >= threshold

    def _detect_historical_crossing(
        self,
        processor_results: Dict,
        timeframe: str,
        ws_label: str,
        idx: int
    ) -> bool:
        """Check if there's an angle crossing at a specific index."""
        window_labels = ["df", "df1", "df2", "df3", "df4"]

        if ws_label not in window_labels or timeframe not in processor_results:
            return False

        # Define crossing pairs that involve this window
        # Full crossing pairs: (df,df1), (df,df2), (df1,df2), (df1,df3), (df2,df3), (df2,df4), (df3,df4)
        crossing_pairs = {
            "df": ["df1", "df2"],
            "df1": ["df", "df2", "df3"],
            "df2": ["df", "df1", "df3", "df4"],
            "df3": ["df1", "df2", "df4"],
            "df4": ["df2", "df3"],
        }

        partners = crossing_pairs.get(ws_label, [])

        for partner_label in partners:
            if partner_label not in processor_results[timeframe]:
                continue

            df_curr = processor_results[timeframe][ws_label]
            df_partner = processor_results[timeframe][partner_label]

            if 'angle' not in df_curr.columns or 'angle' not in df_partner.columns:
                continue

            angles_curr = df_curr['angle'].values
            angles_partner = df_partner['angle'].values

            min_len = min(len(angles_curr), len(angles_partner))
            if idx <= 0 or idx >= min_len:
                continue

            # Align indices from the end
            curr_idx = len(angles_curr) - (min_len - idx)
            partner_idx = len(angles_partner) - (min_len - idx)

            if curr_idx <= 0 or partner_idx <= 0:
                continue

            # Check for crossing (sign change in difference)
            prev_diff = angles_curr[curr_idx - 1] - angles_partner[partner_idx - 1]
            curr_diff = angles_curr[curr_idx] - angles_partner[partner_idx]

            if prev_diff * curr_diff < 0:
                return True

        return False

    def build_calendar_df(
        self,
        signals: List[Signal],
        now: Optional[datetime] = None,
        processor_results: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Build the calendar DataFrame with scoring system.

        Cell format: "direction:score" where score = reversal(+1) + crossing(+1) + accel_quality(+1)

        User story requirements:
        - Predict angle, slope_f, acceleration using linear extrapolation
        - Show convergence event count per cell
        - Detect reversals and crossings in predictions

        Args:
            signals: List of Signal objects from analysis
            now: Current datetime (defaults to UTC now)
            processor_results: Dict of processor results {timeframe: {window_label: DataFrame}}

        Returns:
            DataFrame with rows = TF-window, columns = time, values = "direction:score"
        """
        # Use actual data time range instead of UTC now
        _, data_max_time = self._get_data_time_range(processor_results)

        if now is None:
            if data_max_time is not None:
                # Use the last data point time as "now"
                now = pd.Timestamp(data_max_time).to_pydatetime()
                if hasattr(now, 'tzinfo') and now.tzinfo is not None:
                    now = now.replace(tzinfo=None)
            else:
                now = datetime.utcnow()

        time_cols = self.get_time_columns(now)
        row_labels = self.get_row_labels()
        window_labels = ["df", "df1", "df2", "df3", "df4"]

        # Pre-compute crossing predictions for all timeframes
        crossing_cache = {}
        if processor_results:
            for tf in TIMEFRAME_ORDER:
                crossing_cache[tf] = self._predict_angle_crossings(processor_results, tf, steps=2)

        # Collect all acceleration values for quality threshold
        all_accels = []
        if processor_results:
            for tf in processor_results:
                for ws_label in processor_results[tf]:
                    df = processor_results[tf][ws_label]
                    if 'acceleration' in df.columns:
                        all_accels.extend(df['acceleration'].dropna().values)
        all_accels = np.array(all_accels) if all_accels else np.array([])

        # Initialize DataFrame
        data = []

        for row_label in row_labels:
            parts = row_label.split("-")
            tf = parts[0]
            ws_label = parts[1]

            row_data = {"TF-Window": row_label}

            # Get the DataFrame for this TF-window combination
            df = None
            if processor_results and tf in processor_results and ws_label in processor_results[tf]:
                df = processor_results[tf][ws_label]

            # Get crossing predictions for this timeframe/window
            tf_crossings = crossing_cache.get(tf, {"step_1": [], "step_2": []})

            for time_col in time_cols:
                col_name = time_col.strftime("%H:%M")

                if time_col <= now:
                    # Historical: look up actual data from processor results
                    if df is not None and len(df) > 0:
                        data_row = self._find_data_at_time(df, time_col)
                        if data_row is not None and 'slope_f' in data_row:
                            direction = self._get_direction_from_slope(data_row['slope_f'])

                            # Check for reversal
                            has_reversal = False
                            pos_idx = -1
                            if 'angle' in df.columns:
                                angles = df['angle'].values
                                try:
                                    df_times = pd.to_datetime(df['time'])
                                    target_ts = pd.Timestamp(time_col)
                                    time_diffs = abs(df_times - target_ts)
                                    nearest_idx = time_diffs.idxmin()
                                    pos_idx = df.index.get_loc(nearest_idx) if nearest_idx in df.index else -1
                                    if pos_idx >= 0:
                                        reversal_marker = self._detect_reversal_at_index(angles, pos_idx)
                                        has_reversal = reversal_marker == "*"
                                except Exception:
                                    pass

                            # Check for crossing
                            has_crossing = self._detect_historical_crossing(
                                processor_results, tf, ws_label, pos_idx
                            ) if pos_idx >= 0 else False

                            # Check acceleration quality
                            accel_value = data_row.get('acceleration', np.nan)
                            accel_quality = self._check_acceleration_quality(accel_value, all_accels)

                            # Compute score
                            row_data[col_name] = self._compute_cell_score(
                                has_reversal, has_crossing, accel_quality, direction
                            )
                        else:
                            row_data[col_name] = "-"
                    else:
                        row_data[col_name] = "-"
                else:
                    # Future: predict angle, slope_f, acceleration using linear extrapolation
                    if df is not None and len(df) >= 2:
                        time_diff_hours = (time_col - now).total_seconds() / 3600

                        # Predict slope_f for direction
                        direction = "N"
                        if 'slope_f' in df.columns:
                            slopes = df['slope_f'].dropna().values
                            if len(slopes) >= 2:
                                a, b = slopes[-2], slopes[-1]
                                pred_cc = PricePredictor.predict_next(a, b)
                                pred_ccc = PricePredictor.predict_next(b, pred_cc)
                                pred_slope = pred_cc if time_diff_hours <= 2 else pred_ccc
                                direction = self._get_direction_from_slope(pred_slope)

                        # Predict angle for reversal detection
                        has_reversal = False
                        if 'angle' in df.columns:
                            angles = df['angle'].dropna().values
                            if len(angles) >= 5:
                                # Predict next 2 angles
                                a, b = angles[-2], angles[-1]
                                pred_angle_cc = PricePredictor.predict_next(a, b)
                                pred_angle_ccc = PricePredictor.predict_next(b, pred_angle_cc)

                                # Check for predicted reversal in extended sequence
                                extended = list(angles[-4:]) + [pred_angle_cc, pred_angle_ccc]
                                # Check pattern at predicted points
                                for check_idx in [4, 5]:
                                    if check_idx < len(extended):
                                        window_vals = extended[check_idx-4:check_idx+1]
                                        if len(window_vals) == 5:
                                            # Bottom pattern
                                            if (window_vals[0] > window_vals[1] > window_vals[2] > window_vals[3]
                                                and window_vals[3] < window_vals[4]):
                                                has_reversal = True
                                                break
                                            # Peak pattern
                                            if (window_vals[0] < window_vals[1] < window_vals[2] < window_vals[3]
                                                and window_vals[3] > window_vals[4]):
                                                has_reversal = True
                                                break

                        # Check for predicted crossing
                        has_crossing = False
                        step_key = "step_1" if time_diff_hours <= 2 else "step_2"
                        for ws1, ws2, cross_dir in tf_crossings.get(step_key, []):
                            if ws_label == ws1 or ws_label == ws2:
                                has_crossing = True
                                break

                        # Predict acceleration for quality
                        accel_quality = False
                        if 'acceleration' in df.columns:
                            accels = df['acceleration'].dropna().values
                            if len(accels) >= 2:
                                a, b = accels[-2], accels[-1]
                                pred_accel_cc = PricePredictor.predict_next(a, b)
                                pred_accel_ccc = PricePredictor.predict_next(b, pred_accel_cc)
                                pred_accel = pred_accel_cc if time_diff_hours <= 2 else pred_accel_ccc
                                accel_quality = self._check_acceleration_quality(pred_accel, all_accels)

                        # Compute score with prediction marker
                        score_str = self._compute_cell_score(
                            has_reversal, has_crossing, accel_quality, direction
                        )
                        row_data[col_name] = f"{score_str}~"
                    else:
                        row_data[col_name] = "?"

            data.append(row_data)

        return pd.DataFrame(data)

    def count_convergence(
        self,
        calendar_df: pd.DataFrame,
        time_col: str
    ) -> Dict[str, int]:
        """
        Count signal convergence at a specific time column.

        Cell format: "direction:score" or "direction:score~"

        Args:
            calendar_df: Calendar DataFrame
            time_col: Column name (time)

        Returns:
            Dict with counts: {'LONG': n, 'SHORT': n, 'NEUTRAL': n, 'TOTAL_SCORE': n}
        """
        counts = {'LONG': 0, 'SHORT': 0, 'NEUTRAL': 0, 'TOTAL_SCORE': 0}

        if time_col not in calendar_df.columns:
            return counts

        for val in calendar_df[time_col]:
            if isinstance(val, str) and val not in ['-', '?']:
                val_clean = val.replace('~', '')

                # Parse new format: "direction:score:flags" e.g., "L:2:RC"
                parts = val_clean.split(':')
                if len(parts) >= 1:
                    direction = parts[0]
                else:
                    direction = 'N'

                if len(parts) >= 2:
                    try:
                        score = int(parts[1])
                    except ValueError:
                        score = 0
                else:
                    score = 0

                if direction == 'L':
                    counts['LONG'] += 1
                    counts['TOTAL_SCORE'] += score
                elif direction == 'S':
                    counts['SHORT'] += 1
                    counts['TOTAL_SCORE'] += score
                elif direction == 'N':
                    counts['NEUTRAL'] += 1

        return counts

    def get_convergence_summary(
        self,
        calendar_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get convergence summary for all time columns.

        Returns:
            DataFrame with convergence counts and scores per time column
        """
        summaries = []
        time_cols = [c for c in calendar_df.columns if c != "TF-Window"]

        for col in time_cols:
            counts = self.count_convergence(calendar_df, col)
            total = counts['LONG'] + counts['SHORT']
            dominant = 'LONG' if counts['LONG'] > counts['SHORT'] else 'SHORT' if counts['SHORT'] > counts['LONG'] else 'NEUTRAL'
            pct = max(counts['LONG'], counts['SHORT']) / max(total, 1) * 100

            summaries.append({
                'Time': col,
                'LONG': counts['LONG'],
                'SHORT': counts['SHORT'],
                'NEUTRAL': counts['NEUTRAL'],
                'Dominant': dominant,
                'Convergence %': f"{pct:.0f}%",
                'Total Score': counts['TOTAL_SCORE']
            })

        return pd.DataFrame(summaries)

    def get_aggregated_score_table(
        self,
        calendar_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get aggregated score breakdown table by time.

        Aggregates all 55 TF-windows per time slot into:
        - Reversal Count (how many reversals detected)
        - Cross Count (how many crossings detected)
        - Accel Quality Count (how many in distant/very_distant zone)
        - Total Score
        - Dominant Direction

        Returns:
            DataFrame with score breakdown per time column
        """
        summaries = []
        time_cols = [c for c in calendar_df.columns if c != "TF-Window"]

        for col in time_cols:
            reversal_count = 0
            cross_count = 0
            accel_count = 0
            total_score = 0
            long_count = 0
            short_count = 0
            neutral_count = 0

            for val in calendar_df[col]:
                if not isinstance(val, str) or val in ['-', '?']:
                    continue

                val_clean = val.replace('~', '')

                # Parse new format: "direction:score:flags" e.g., "L:2:RC"
                # Flags: R=Reversal, C=Crossing, A=Accel Quality
                parts = val_clean.split(':')
                if len(parts) >= 1:
                    direction = parts[0]
                else:
                    direction = 'N'

                if len(parts) >= 2:
                    try:
                        score = int(parts[1])
                    except ValueError:
                        score = 0
                else:
                    score = 0

                # Parse flags from third part
                flags = parts[2] if len(parts) >= 3 else ""

                # Count directions
                if direction == 'L':
                    long_count += 1
                elif direction == 'S':
                    short_count += 1
                else:
                    neutral_count += 1

                # Count components from flags (accurate tracking)
                if 'R' in flags:
                    reversal_count += 1
                if 'C' in flags:
                    cross_count += 1
                if 'A' in flags:
                    accel_count += 1

                total_score += score

            # Determine dominant direction
            if long_count > short_count:
                dominant = 'LONG'
                dominant_pct = long_count / max(long_count + short_count, 1) * 100
            elif short_count > long_count:
                dominant = 'SHORT'
                dominant_pct = short_count / max(long_count + short_count, 1) * 100
            else:
                dominant = 'NEUTRAL'
                dominant_pct = 50

            # Check if this is a prediction (future) column
            is_prediction = any('~' in str(v) for v in calendar_df[col] if isinstance(v, str))

            summaries.append({
                'Time': col,
                'Reversals': reversal_count,
                'Crossings': cross_count,
                'Accel Quality': accel_count,
                'Total Score': total_score,
                'LONG': long_count,
                'SHORT': short_count,
                'Direction': dominant,
                'Strength %': f"{dominant_pct:.0f}%",
                'Is Forecast': is_prediction
            })

        return pd.DataFrame(summaries)


def run_advanced_analysis() -> SignalLogic:
    """Run the advanced signal analysis."""
    logic = SignalLogic()
    aggregated = logic.run_analysis()
    print(logic.get_report(aggregated))
    return logic


if __name__ == "__main__":
    logic = run_advanced_analysis()
