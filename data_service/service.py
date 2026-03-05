"""Data service main entry point — startup, gap detection, incremental cycles.

Usage:
    python -m data_service.service --data-dir ./persistent_data --interval 300
    python -m data_service.service --threshold 0.75 --debug
"""

import argparse
import json
import logging
import os
import signal as signal_module
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force unbuffered stdout
os.environ["PYTHONUNBUFFERED"] = "1"

from data_service.layers import PersistentPipeline

logger = logging.getLogger("data_service")

# Graceful shutdown flag
_shutdown_requested = False
MAX_CONSECUTIVE_ERRORS = 10


def _sigterm_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("SIGTERM received — shutting down after current cycle")


signal_module.signal(signal_module.SIGTERM, _sigterm_handler)


def write_status(data_dir: Path, status: dict):
    """Write status.json atomically for Docker healthcheck."""
    import tempfile
    status_path = data_dir / "status.json"
    status["timestamp"] = datetime.now(timezone.utc).isoformat()
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(data_dir), suffix=".tmp")
        os.close(tmp_fd)
        with open(tmp_path, "w") as f:
            json.dump(status, f, indent=2)
        os.replace(tmp_path, str(status_path))
    except Exception as e:
        logger.warning(f"Failed to write status.json: {e}")
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def run_service(args):
    global _shutdown_requested

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Data service starting")
    logger.info(f"  data_dir: {data_dir}")
    logger.info(f"  interval: {args.interval}s")
    logger.info(f"  threshold: {args.threshold}")

    # Initialize pipeline (loads models)
    pipeline = PersistentPipeline(
        data_dir=data_dir,
        threshold=args.threshold,
        model_dir=args.model_dir,
    )

    # Write initial status
    write_status(data_dir, {"state": "starting", "cycle": 0})

    consecutive_errors = 0
    cycle = 0

    while not _shutdown_requested:
        cycle += 1
        logger.info(f"--- Cycle {cycle} ---")

        try:
            status = pipeline.run_cycle()
            consecutive_errors = 0  # Reset on success

            # Log summary
            pred = status.get("prediction")
            if status.get("skipped"):
                logger.info(f"  Cycle {cycle}: SKIPPED (no new 5M) "
                            f"[L1={status['l1_time']}s]")
            elif pred:
                logger.info(f"  Cycle {cycle}: {pred['signal']} "
                            f"(conf={pred['confidence']:.3f}) "
                            f"[L1={status['l1_time']}s L2={status['l2_time']}s "
                            f"L3={status['l3_time']}s total={status['total_time']}s]")
            else:
                logger.info(f"  Cycle {cycle}: no new prediction "
                            f"[L1={status['l1_time']}s L2={status['l2_time']}s "
                            f"L3={status['l3_time']}s total={status['total_time']}s]")

            # Update status file
            write_status(data_dir, {
                "state": "running",
                "cycle": cycle,
                "last_cycle": status,
            })

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            break

        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Cycle {cycle} failed ({consecutive_errors}/"
                         f"{MAX_CONSECUTIVE_ERRORS}): {e}", exc_info=True)

            write_status(data_dir, {
                "state": "error",
                "cycle": cycle,
                "error": str(e),
                "consecutive_errors": consecutive_errors,
            })

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logger.critical(f"Too many consecutive errors "
                                f"({consecutive_errors}) — exiting")
                sys.exit(1)

        # Sleep with 1-second granularity for SIGTERM responsiveness
        sleep_until = time.time() + args.interval
        while time.time() < sleep_until and not _shutdown_requested:
            time.sleep(1)

    # Shutdown
    logger.info("Data service shutting down")
    write_status(data_dir, {"state": "stopped", "cycle": cycle})


def main():
    parser = argparse.ArgumentParser(
        description="Persistent incremental data pipeline for V10 predictions")
    parser.add_argument("--data-dir", type=str, default="./persistent_data",
                        help="Base directory for persistent data (default: ./persistent_data)")
    parser.add_argument("--interval", type=int, default=300,
                        help="Seconds between cycles (default: 300)")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Prediction confidence threshold (default: 0.75)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory with production models")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    from core.structured_log import setup_logging
    setup_logging(
        "data_service",
        log_dir=str(Path(args.data_dir) / "logs"),
        debug=args.debug,
    )

    run_service(args)


if __name__ == "__main__":
    main()
