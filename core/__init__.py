"""Core signal processing engine: config, analysis, processor, signal_logic."""

from core.config import (
    TIMEFRAME_ORDER,
    WINDOW_SIZES,
    DATA_DIR,
    OUTPUT_DIR,
    CHART_COLORS,
)
from core.analysis import iterative_regression, calculate_acceleration
from core.processor import TimeframeProcessor
