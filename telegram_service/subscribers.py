"""Subscriber persistence — atomic JSON store for Telegram chat IDs."""

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class SubscriberStore:
    """Persists subscriber chat IDs as a JSON file with atomic writes."""

    def __init__(self, path: str = "/app/telegram_data/subscribers.json"):
        self._path = Path(path)
        self._chat_ids: set[int] = set()
        self._load()

    def _load(self):
        """Load subscribers from disk."""
        if not self._path.exists():
            self._chat_ids = set()
            return
        try:
            data = json.loads(self._path.read_text())
            self._chat_ids = set(data.get("chat_ids", []))
            logger.info(f"Loaded {len(self._chat_ids)} subscriber(s)")
        except Exception as e:
            logger.warning(f"Failed to load subscribers: {e}")
            self._chat_ids = set()

    def _save(self):
        """Save subscribers atomically (temp file + os.replace)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self._path.parent), suffix=".tmp")
        os.close(tmp_fd)
        try:
            with open(tmp_path, "w") as f:
                json.dump({"chat_ids": sorted(self._chat_ids)}, f)
            os.replace(tmp_path, str(self._path))
        except Exception as e:
            logger.error(f"Failed to save subscribers: {e}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def add(self, chat_id: int) -> bool:
        """Add a subscriber. Returns True if new, False if already subscribed."""
        if chat_id in self._chat_ids:
            return False
        self._chat_ids.add(chat_id)
        self._save()
        logger.info(f"Subscriber added: {chat_id} (total: {len(self._chat_ids)})")
        return True

    def remove(self, chat_id: int) -> bool:
        """Remove a subscriber. Returns True if removed, False if not found."""
        if chat_id not in self._chat_ids:
            return False
        self._chat_ids.discard(chat_id)
        self._save()
        logger.info(f"Subscriber removed: {chat_id} (total: {len(self._chat_ids)})")
        return True

    def get_all(self) -> list[int]:
        """Return all subscriber chat IDs."""
        return list(self._chat_ids)

    def count(self) -> int:
        return len(self._chat_ids)
