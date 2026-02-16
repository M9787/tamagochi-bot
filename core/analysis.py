#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core Analysis Module for Cryptocurrency Trading Signal System

Implements the algorithms from Power BI Power Query:
- iterative_regression: Rolling linear regression with backward/forward windows
- angle_between_lines: Calculate angle between two regression lines
- calculate_acceleration: Compute rate of change of angles
- get_signal_summary: Direction summary from slope_f
"""

import math
from typing import Optional, Dict

import numpy as np
import pandas as pd
import scipy.stats as stats


def angle_between_lines(slope1: float, slope2: float) -> float:
    """
    Calculate the angle (in degrees) between two lines given their slopes.

    Args:
        slope1: Slope of the first line (forward regression)
        slope2: Slope of the second line (backward regression)

    Returns:
        Angle in degrees between the two lines
    """
    # Calculate the angles between each line and the positive x-axis
    theta1 = math.atan(slope1)
    theta2 = math.atan(slope2)

    # Calculate the angle between the two lines
    angle = abs(theta1 - theta2)

    # Convert from radians to degrees
    angle_degrees = math.degrees(angle)

    return angle_degrees


def iterative_regression(
    df: pd.DataFrame,
    window_size: int,
    cut_index: Optional[int] = None
) -> pd.DataFrame:
    """
    Perform iterative linear regression with sliding backward/forward windows.

    For each data point, maintains a sliding window of 2 * window_size prices:
    - Backward window: older N points
    - Forward window: newer N points

    Uses sqrt(price) for all calculations as per Power BI implementation.

    Args:
        df: DataFrame with 'Open Time' and 'Close' columns
        window_size: Size of each half-window (total window = 2 * window_size)
        cut_index: Optional index to slice data (for limiting data range)

    Returns:
        DataFrame with regression results: count, intercept_b/f, slope_b/f,
        p_value_b/f, corr, spearman, actual, time, angle
    """
    # Find column names dynamically
    columns = list(df.columns)
    date_column_name = [val for val in columns if 'time' in val.lower()]
    date_column_name = ''.join(date_column_name) if date_column_name else 'Open Time'

    # Get price column (the other column)
    price_column_name = [c for c in columns if c != date_column_name]
    price_column_name = price_column_name[0] if price_column_name else 'Close'

    # Sort by date
    df = df.sort_values(by=date_column_name).copy()

    # Apply cut_index if provided
    if cut_index is not None:
        df = df.iloc[:cut_index, :]

    # Initialize results dictionary
    results = {
        'count': [],
        'intercept_b': [],
        'intercept_f': [],
        'slope_b': [],
        'slope_f': [],
        'p_value_b': [],
        'p_value_f': [],
        'corr': [],
        'spearman': [],
        'actual': [],
        'time': [],
        'angle': []
    }

    # Apply sqrt transformation to prices (as per Power BI implementation)
    price_column = df[price_column_name].map(np.sqrt)
    date_column = df[date_column_name]

    # Initialize the window with first 2*window_size elements
    window = list(price_column.values[:window_size * 2])
    step = np.arange(1, window_size + 1)

    # Counter for results
    count = 0

    # Iterate over remaining data points
    for number, time in zip(
        price_column.values[window_size * 2:],
        date_column.values[window_size * 2:]
    ):
        count += 1

        # Split window into backward (older) and forward (newer) halves
        backward = window[:window_size]
        forward = window[window_size:]

        # Perform linear regression on backward window
        slope_b, intercept_b, r_value_b, p_value_b, std_err_b = stats.linregress(step, backward)

        # Perform linear regression on forward window
        slope_f, intercept_f, r_value_f, p_value_f, std_err_f = stats.linregress(step, forward)

        # Calculate Spearman correlation between windows
        spearman_corr, _ = stats.spearmanr(backward, forward)

        # Calculate Pearson correlation between windows
        pearson_corr = np.corrcoef(backward, forward)[0][-1]

        # Store results
        results['count'].append(count)
        results['intercept_b'].append(intercept_b)
        results['intercept_f'].append(intercept_f)
        results['slope_b'].append(slope_b)
        results['slope_f'].append(slope_f)
        results['p_value_b'].append(p_value_b)
        results['p_value_f'].append(p_value_f)
        results['corr'].append(pearson_corr)
        results['spearman'].append(spearman_corr)
        results['actual'].append(number)
        results['time'].append(time)
        results['angle'].append(angle_between_lines(slope_f, slope_b))

        # Slide the window: remove oldest, add current
        window.pop(0)
        window.append(number)

    return pd.DataFrame.from_dict(results)


def calculate_acceleration(angles: pd.Series) -> pd.Series:
    """
    Calculate the acceleration (rate of change) of angles.

    acceleration[t] = angle[t] - angle[t-1]

    Args:
        angles: Series of angle values

    Returns:
        Series of acceleration values (first value is NaN)
    """
    return angles.diff()


def get_signal_summary(df: pd.DataFrame) -> Dict[str, any]:
    """
    Get a summary of the current trading signal state based on slope_f direction.

    Args:
        df: DataFrame with regression output columns (slope_f, spearman, angle, time)

    Returns:
        Dictionary with signal summary including latest signal and counts
    """
    # Get latest row
    latest = df.iloc[-1] if len(df) > 0 else None

    # Determine current signal from slope_f direction
    current_signal = 'HOLD'
    if latest is not None and 'slope_f' in df.columns:
        slope_f = latest['slope_f']
        if slope_f > 0:
            current_signal = 'BUY'
        elif slope_f < 0:
            current_signal = 'SELL'

    # Count directions in recent history (last 10 rows)
    recent = df.tail(10) if len(df) > 0 else df
    recent_buy_count = int((recent['slope_f'] > 0).sum()) if 'slope_f' in recent.columns else 0
    recent_sell_count = int((recent['slope_f'] < 0).sum()) if 'slope_f' in recent.columns else 0

    return {
        'current_signal': current_signal,
        'latest_time': latest['time'] if latest is not None and 'time' in df.columns else None,
        'latest_slope_f': latest['slope_f'] if latest is not None else None,
        'latest_spearman': latest['spearman'] if latest is not None else None,
        'latest_angle': latest['angle'] if latest is not None else None,
        'recent_buy_count': recent_buy_count,
        'recent_sell_count': recent_sell_count,
    }


if __name__ == '__main__':
    print("Analysis module loaded successfully.")
    print(f"Available functions: angle_between_lines, iterative_regression, calculate_acceleration, get_signal_summary")

    # Quick test of angle calculation
    test_angle = angle_between_lines(0.5, -0.5)
    print(f"Test angle between slopes 0.5 and -0.5: {test_angle:.2f} degrees")
