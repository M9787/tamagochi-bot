#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-Timeframe Processor for Cryptocurrency Trading Signal System

Processes all 11 timeframes with all 5 window sizes:
- Loads CSV data from Binance kline exports
- Runs iterative regression analysis
- Generates trading signals
- Outputs 55 result DataFrames (11 timeframes x 5 window sizes)
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np

from core.analysis import (
    iterative_regression,
    calculate_acceleration,
    get_signal_summary
)
from core.config import (
    TIMEFRAME_ORDER,
    WINDOW_SIZES,
    WINDOW_LABELS,
    get_csv_path,
    get_output_path,
    DATA_DIR
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _process_timeframe_worker(args: Tuple[str, pd.DataFrame, List[int]]) -> Tuple[str, Dict[str, pd.DataFrame]]:
    """
    Worker function for parallel timeframe processing.
    Must be defined at module level for pickling.

    Args:
        args: Tuple of (timeframe, dataframe, window_sizes)

    Returns:
        Tuple of (timeframe, results_dict)
    """
    timeframe, df, window_sizes = args
    results = {}

    for i, ws in enumerate(window_sizes):
        label = f"df{i}" if i > 0 else "df"

        # Check if we have enough data
        min_required = ws * 2 + 1
        if len(df) < min_required:
            continue

        try:
            # Run iterative regression
            result = iterative_regression(df, ws, len(df['Close']))

            # Add acceleration
            result['acceleration'] = calculate_acceleration(result['angle'])

            # Store window size metadata
            result.attrs['window_size'] = ws
            result.attrs['timeframe'] = timeframe

            results[label] = result

        except Exception as e:
            logger.error(f"Error processing {timeframe}/{label}: {e}")

    return timeframe, results


class TimeframeProcessor:
    """Processes multiple timeframes with multiple window sizes."""

    def __init__(
        self,
        timeframes: Optional[List[str]] = None,
        window_sizes: Optional[List[int]] = None,
        data_dir: Optional[Path] = None
    ):
        """
        Initialize the processor.

        Args:
            timeframes: List of timeframes to process (default: all 11)
            window_sizes: List of window sizes (default: [30, 60, 100, 120, 160])
            data_dir: Directory containing CSV files
        """
        self.timeframes = timeframes or TIMEFRAME_ORDER
        self.window_sizes = window_sizes or WINDOW_SIZES
        self.data_dir = data_dir or DATA_DIR

        # Storage for results
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.results: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.summaries: Dict[str, Dict[str, dict]] = {}

    def load_data(self, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Load CSV data for a single timeframe.

        Args:
            timeframe: Timeframe label (e.g., '3D', '1H', '5M')

        Returns:
            DataFrame with 'Open Time' and 'Close' columns, or None if file not found
        """
        csv_path = self.data_dir / f"testing_data_{timeframe}.csv"

        if not csv_path.exists():
            logger.warning(f"CSV file not found: {csv_path}")
            return None

        try:
            df = pd.read_csv(csv_path)

            # Handle potential index column from reset_index().to_csv()
            if 'index' in df.columns:
                df = df.drop(columns=['index'])

            # Ensure proper column names
            if 'Open Time' not in df.columns:
                # Try to find datetime column
                for col in df.columns:
                    if 'time' in col.lower() or 'date' in col.lower():
                        df = df.rename(columns={col: 'Open Time'})
                        break

            # Parse datetime
            df['Open Time'] = pd.to_datetime(df['Open Time'], utc=True)

            # Ensure Close is numeric
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

            # Remove any NaN values
            df = df.dropna()

            logger.info(f"Loaded {len(df)} rows for {timeframe}")
            return df

        except Exception as e:
            logger.error(f"Error loading {timeframe}: {e}")
            return None

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load data for all configured timeframes.

        Returns:
            Dictionary mapping timeframe labels to DataFrames
        """
        for tf in self.timeframes:
            df = self.load_data(tf)
            if df is not None:
                self.raw_data[tf] = df

        logger.info(f"Loaded data for {len(self.raw_data)}/{len(self.timeframes)} timeframes")
        return self.raw_data

    def process_timeframe(
        self,
        timeframe: str,
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Process a single timeframe with all window sizes.

        Args:
            timeframe: Timeframe label
            df: Optional DataFrame (if not provided, uses cached raw_data)

        Returns:
            Dictionary mapping window labels to processed DataFrames
        """
        if df is None:
            df = self.raw_data.get(timeframe)

        if df is None:
            logger.warning(f"No data available for {timeframe}")
            return {}

        results = {}

        for i, ws in enumerate(self.window_sizes):
            label = f"df{i}" if i > 0 else "df"

            # Check if we have enough data
            min_required = ws * 2 + 1
            if len(df) < min_required:
                logger.warning(
                    f"Insufficient data for {timeframe} window {ws}: "
                    f"need {min_required}, have {len(df)}"
                )
                continue

            try:
                # Run iterative regression
                result = iterative_regression(df, ws, len(df['Close']))

                # Add acceleration
                result['acceleration'] = calculate_acceleration(result['angle'])

                # Store window size metadata
                result.attrs['window_size'] = ws
                result.attrs['timeframe'] = timeframe

                results[label] = result
                logger.debug(f"Processed {timeframe}/{label} ({ws}): {len(result)} rows")

            except Exception as e:
                logger.error(f"Error processing {timeframe}/{label}: {e}")

        return results

    def process_all(self, parallel: bool = True, max_workers: Optional[int] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Process all timeframes with all window sizes.

        Args:
            parallel: Use parallel processing (default: True)
            max_workers: Max parallel workers (default: min(cpu_count-1, 8))

        Returns:
            Nested dictionary: results[timeframe][window_label] = DataFrame
        """
        if not self.raw_data:
            self.load_all_data()

        if parallel and len(self.raw_data) > 1:
            self._process_all_parallel(max_workers)
        else:
            self._process_all_sequential()

        total = sum(len(r) for r in self.results.values())
        logger.info(f"Total processed: {total} DataFrames")

        return self.results

    def _process_all_sequential(self) -> None:
        """Process all timeframes sequentially (fallback)."""
        for tf in self.timeframes:
            if tf in self.raw_data:
                self.results[tf] = self.process_timeframe(tf)
                logger.info(f"Processed {tf}: {len(self.results[tf])} window sizes")

    def _process_all_parallel(self, max_workers: Optional[int] = None) -> None:
        """Process all timeframes in parallel using ProcessPoolExecutor."""
        # Determine optimal worker count: min(cpu_count - 1, 8, num_timeframes)
        cpu_count = os.cpu_count() or 4
        num_timeframes = len([tf for tf in self.timeframes if tf in self.raw_data])
        workers = min(cpu_count - 1, 8, num_timeframes) if max_workers is None else max_workers

        logger.info(f"Starting parallel processing with {workers} workers for {num_timeframes} timeframes")

        # Prepare worker arguments: (timeframe, dataframe, window_sizes)
        worker_args = [
            (tf, self.raw_data[tf], self.window_sizes)
            for tf in self.timeframes
            if tf in self.raw_data
        ]

        # Process in parallel
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_tf = {
                executor.submit(_process_timeframe_worker, args): args[0]
                for args in worker_args
            }

            # Collect results as they complete
            for future in as_completed(future_to_tf):
                tf = future_to_tf[future]
                try:
                    timeframe, results = future.result()
                    self.results[timeframe] = results
                    logger.info(f"Processed {timeframe}: {len(results)} window sizes")
                except Exception as e:
                    logger.error(f"Failed to process {tf}: {e}")

    def get_latest_signals(self) -> Dict[str, Dict[str, dict]]:
        """
        Get the latest signal summary for all timeframes and window sizes.

        Returns:
            Nested dictionary: summaries[timeframe][window_label] = signal_summary
        """
        for tf, tf_results in self.results.items():
            self.summaries[tf] = {}
            for label, df in tf_results.items():
                self.summaries[tf][label] = get_signal_summary(df)

        return self.summaries

    def get_signals_table(self) -> pd.DataFrame:
        """
        Create a summary table of all current signals.

        Returns:
            DataFrame with timeframe, window, signal, and key metrics
        """
        rows = []

        for tf in self.timeframes:
            if tf not in self.results:
                continue

            for label, df in self.results[tf].items():
                if len(df) == 0:
                    continue

                summary = get_signal_summary(df)
                ws = df.attrs.get('window_size', 0)

                rows.append({
                    'timeframe': tf,
                    'window': ws,
                    'window_label': label,
                    'signal': summary['current_signal'],
                    'time': summary['latest_time'],
                    'slope_f': summary['latest_slope_f'],
                    'spearman': summary['latest_spearman'],
                    'angle': summary['latest_angle'],
                    'recent_buys': summary['recent_buy_count'],
                    'recent_sells': summary['recent_sell_count']
                })

        return pd.DataFrame(rows)

    def export_results(self, output_dir: Optional[Path] = None) -> List[Path]:
        """
        Export all results to CSV files.

        Args:
            output_dir: Directory to save files (default: configured OUTPUT_DIR)

        Returns:
            List of saved file paths
        """
        output_dir = output_dir or get_output_path("").parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []

        for tf, tf_results in self.results.items():
            for label, df in tf_results.items():
                filename = f"analysis_{tf}_{label}.csv"
                filepath = output_dir / filename
                df.to_csv(filepath, index=False)
                saved_files.append(filepath)
                logger.debug(f"Saved {filepath}")

        # Also save the summary table
        summary_df = self.get_signals_table()
        summary_path = output_dir / "signals_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        saved_files.append(summary_path)

        logger.info(f"Exported {len(saved_files)} files to {output_dir}")
        return saved_files

    def get_combined_angles(self, timeframe: str) -> pd.DataFrame:
        """
        Get combined angles from all window sizes for a timeframe.

        Args:
            timeframe: Timeframe label

        Returns:
            DataFrame with time and angle columns for each window size
        """
        if timeframe not in self.results:
            return pd.DataFrame()

        dfs = []
        for label, df in self.results[timeframe].items():
            ws = df.attrs.get('window_size', label)
            subset = df[['time', 'angle']].copy()
            subset = subset.rename(columns={'angle': f'angle_{ws}'})
            dfs.append(subset)

        if not dfs:
            return pd.DataFrame()

        # Merge all on time
        result = dfs[0]
        for df in dfs[1:]:
            result = pd.merge(result, df, on='time', how='outer')

        return result.sort_values('time')

    def get_combined_slopes(self, timeframe: str) -> pd.DataFrame:
        """
        Get combined forward slopes from all window sizes for a timeframe.

        Args:
            timeframe: Timeframe label

        Returns:
            DataFrame with time and slope_f columns for each window size
        """
        if timeframe not in self.results:
            return pd.DataFrame()

        dfs = []
        for label, df in self.results[timeframe].items():
            ws = df.attrs.get('window_size', label)
            subset = df[['time', 'slope_f', 'p_value_f']].copy()
            subset = subset.rename(columns={
                'slope_f': f'slope_f_{ws}',
                'p_value_f': f'p_value_f_{ws}'
            })
            dfs.append(subset)

        if not dfs:
            return pd.DataFrame()

        # Merge all on time
        result = dfs[0]
        for df in dfs[1:]:
            result = pd.merge(result, df, on='time', how='outer')

        return result.sort_values('time')


def run_analysis(
    timeframes: Optional[List[str]] = None,
    window_sizes: Optional[List[int]] = None,
    export: bool = True
) -> TimeframeProcessor:
    """
    Run the full analysis pipeline.

    Args:
        timeframes: List of timeframes to process
        window_sizes: List of window sizes
        export: Whether to export results to CSV

    Returns:
        TimeframeProcessor instance with results
    """
    processor = TimeframeProcessor(timeframes, window_sizes)

    logger.info("Loading data...")
    processor.load_all_data()

    logger.info("Processing all timeframes...")
    processor.process_all()

    logger.info("Generating signal summaries...")
    processor.get_latest_signals()

    if export:
        logger.info("Exporting results...")
        processor.export_results()

    return processor


if __name__ == "__main__":
    print("=" * 70)
    print("Multi-Timeframe Processor")
    print("=" * 70)

    # Run full analysis
    processor = run_analysis(export=True)

    # Print summary
    print("\nSignal Summary:")
    print("-" * 70)

    signals_df = processor.get_signals_table()
    if len(signals_df) > 0:
        print(signals_df.to_string(index=False))
    else:
        print("No signals generated. Check if data files exist in:", DATA_DIR)

    print("\n" + "=" * 70)
