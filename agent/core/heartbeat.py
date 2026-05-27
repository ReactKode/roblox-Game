"""Heartbeat daemon - background thread for continuous agent operation."""
import json
import threading
from datetime import datetime
from pathlib import Path


class Heartbeat:
    def __init__(self, data_dir: Path, interval: int = 300, on_beat=None):
        self._data_dir = data_dir
        self._interval = interval
        self._on_beat = on_beat  # Optional callback(beat_count)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._beat_count = 0
        self._started_at: str | None = None
        self._status_path = data_dir / "status.json"

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._started_at = datetime.utcnow().isoformat()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="NexusHeartbeat")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self):
        while not self._stop_event.wait(timeout=self._interval):
            self._beat()

    def _beat(self):
        self._beat_count += 1
        self._write_status()
        if self._on_beat:
            try:
                self._on_beat(self._beat_count)
            except Exception:
                pass

    def _write_status(self):
        try:
            status = {
                "agent": "NexusAgent",
                "version": "1.0.0",
                "status": "running",
                "heartbeat_count": self._beat_count,
                "started_at": self._started_at,
                "last_beat": datetime.utcnow().isoformat(),
                "interval_seconds": self._interval,
            }
            self._status_path.parent.mkdir(parents=True, exist_ok=True)
            self._status_path.write_text(json.dumps(status, indent=2))
        except Exception:
            pass

    @property
    def beat_count(self) -> int:
        return self._beat_count

    @property
    def uptime_seconds(self) -> float:
        if not self._started_at:
            return 0.0
        from datetime import timezone
        started = datetime.fromisoformat(self._started_at)
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        return (now - started).total_seconds()
