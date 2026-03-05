"""Structured JSONL logging — adds rotating file handler alongside stdout.

Usage:
    from core.structured_log import setup_logging, log_structured_event
    setup_logging("trading_bot", log_dir="logs/bot", debug=False)
    log_structured_event(logger, "TRADE_OPEN", signal="LONG", price=95000)

Queryability:
    grep '"TRADE_OPEN"' logs/bot/trading_bot.jsonl | jq .
    pd.read_json("logs/bot/trading_bot.jsonl", lines=True).query("level == 'ERROR'")
"""

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record):
        entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        # Include any extra structured fields
        for key in getattr(record, "_structured_fields", {}):
            entry[key] = record._structured_fields[key]
        return json.dumps(entry, default=str)


def setup_logging(service_name, log_dir=None, debug=False):
    """Configure root logger with stdout + optional JSONL file handler.

    Args:
        service_name: Used for JSONL filename ({service_name}.jsonl)
        log_dir: Directory for JSONL file. None = stdout only.
        debug: If True, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if debug else logging.INFO

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers (prevents duplicate handlers on re-init)
    root.handlers.clear()

    # Stdout handler — preserves existing format
    stdout = logging.StreamHandler()
    stdout.setLevel(level)
    stdout.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    root.addHandler(stdout)

    # JSONL file handler — rotating, 10MB, 5 backups
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        jsonl_path = os.path.join(log_dir, f"{service_name}.jsonl")
        file_handler = RotatingFileHandler(
            jsonl_path, maxBytes=10 * 1024 * 1024, backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)

        logging.getLogger(service_name).info(
            f"JSONL logging enabled: {jsonl_path}"
        )


def log_structured_event(logger, event_type, **kwargs):
    """Log a structured event with typed fields for easy querying.

    Args:
        logger: Logger instance
        event_type: Event type string (e.g., "TRADE_OPEN", "SL_TP_HIT")
        **kwargs: Arbitrary key-value pairs included in the JSON record
    """
    record = logger.makeRecord(
        name=logger.name,
        level=logging.INFO,
        fn="",
        lno=0,
        msg=event_type,
        args=(),
        exc_info=None,
    )
    record._structured_fields = {"event_type": event_type, **kwargs}
    logger.handle(record)
