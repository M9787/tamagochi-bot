"""Subscriber persistence — atomic JSON store for Telegram chat IDs + settings."""

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.70


class SubscriberStore:
    """Persists subscriber chat IDs and per-user settings as a JSON file."""

    def __init__(self, path: str = "/app/telegram_data/subscribers.json"):
        self._path = Path(path)
        self._subscribers: dict[int, dict] = {}
        self._load()

    def _load(self):
        """Load subscribers from disk. Handles old and new formats."""
        if not self._path.exists():
            self._subscribers = {}
            return
        try:
            data = json.loads(self._path.read_text())

            # New format: {"subscribers": {"123": {"threshold": 0.70}}}
            if "subscribers" in data:
                self._subscribers = {
                    int(k): v for k, v in data["subscribers"].items()
                }
            # Old format: {"chat_ids": [123, 456]} — migrate
            elif "chat_ids" in data:
                self._subscribers = {
                    int(cid): {"threshold": DEFAULT_THRESHOLD}
                    for cid in data["chat_ids"]
                }
                self._save()  # persist migration
                logger.info("Migrated subscribers from old format")
            else:
                self._subscribers = {}

            logger.info(f"Loaded {len(self._subscribers)} subscriber(s)")
        except Exception as e:
            logger.warning(f"Failed to load subscribers: {e}")
            self._subscribers = {}

    def _save(self):
        """Save subscribers atomically (temp file + os.replace)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self._path.parent), suffix=".tmp")
        os.close(tmp_fd)
        try:
            payload = {
                "subscribers": {
                    str(k): v for k, v in sorted(self._subscribers.items())
                }
            }
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, str(self._path))
        except Exception as e:
            logger.error(f"Failed to save subscribers: {e}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def add(self, chat_id: int) -> bool:
        """Add a subscriber. Returns True if new, False if already subscribed."""
        if chat_id in self._subscribers:
            return False
        self._subscribers[chat_id] = {"threshold": DEFAULT_THRESHOLD}
        self._save()
        logger.info(f"Subscriber added: {chat_id} (total: {len(self._subscribers)})")
        return True

    def remove(self, chat_id: int) -> bool:
        """Remove a subscriber. Returns True if removed, False if not found."""
        if chat_id not in self._subscribers:
            return False
        del self._subscribers[chat_id]
        self._save()
        logger.info(f"Subscriber removed: {chat_id} (total: {len(self._subscribers)})")
        return True

    def get_all(self) -> list[int]:
        """Return all subscriber chat IDs."""
        return list(self._subscribers.keys())

    def get_all_with_settings(self) -> dict[int, dict]:
        """Return all subscribers with their settings."""
        return dict(self._subscribers)

    def get_threshold(self, chat_id: int) -> float:
        """Get a subscriber's alert threshold."""
        sub = self._subscribers.get(chat_id, {})
        return sub.get("threshold", DEFAULT_THRESHOLD)

    def set_threshold(self, chat_id: int, threshold: float) -> bool:
        """Set a subscriber's alert threshold. Returns False if not subscribed."""
        if chat_id not in self._subscribers:
            return False
        self._subscribers[chat_id]["threshold"] = threshold
        self._save()
        logger.info(f"Threshold set: {chat_id} → {threshold}")
        return True

    def count(self) -> int:
        return len(self._subscribers)
