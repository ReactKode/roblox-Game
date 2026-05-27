"""Heartbeat daemon — background intelligence loop.

Every beat:
  1. Consolidates short-term memory into semantic store
  2. Re-learns low-confidence high-usage skills
  3. Stores lessons from recent episodes into semantic memory
  4. Writes agent status file
"""
import json
import threading
from datetime import datetime
from pathlib import Path


class Heartbeat:
    def __init__(self, data_dir: Path, interval: int = 300,
                 memory_manager=None, skill_registry=None, skill_learner=None):
        self._data_dir = data_dir
        self._interval = interval
        self._memory = memory_manager
        self._skills = skill_registry
        self._learner = skill_learner
        self._status_path = data_dir / "status.json"

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._beat_count = 0
        self._started_at: str | None = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._started_at = datetime.utcnow().isoformat()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="NexusHeartbeat")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def beat_count(self) -> int:
        return self._beat_count

    @property
    def uptime_seconds(self) -> float:
        if not self._started_at:
            return 0.0
        from datetime import timezone
        s = datetime.fromisoformat(self._started_at).replace(tzinfo=timezone.utc)
        return (datetime.now(tz=timezone.utc) - s).total_seconds()

    # ── loop ─────────────────────────────────────────────────────────────────

    def _loop(self):
        while not self._stop.wait(timeout=self._interval):
            self._beat()

    def _beat(self):
        self._beat_count += 1
        try:
            self._consolidate_memory()
        except Exception:
            pass
        try:
            if self._beat_count % 3 == 0:  # every 3rd beat (15 min at 5-min interval)
                self._promote_lessons()
        except Exception:
            pass
        try:
            if self._beat_count % 6 == 0:  # every 6th beat (30 min)
                self._refresh_weak_skills()
        except Exception:
            pass
        self._write_status()

    def _consolidate_memory(self):
        if self._memory:
            self._memory.consolidate()

    def _promote_lessons(self):
        """Move high-value lessons from episodic to semantic memory."""
        if not self._memory:
            return
        reflections = self._memory.episodic.recall_reflections(limit=10)
        for r in reflections:
            obs = r.get("observation", "")
            if obs and "lesson=" in obs:
                lesson_text = obs.split("lesson=")[-1].split("|")[0].strip()
                if len(lesson_text) > 20:
                    self._memory.store_lesson(lesson_text)

    def _refresh_weak_skills(self):
        """Re-learn skills that are heavily used but have low confidence."""
        if not self._skills or not self._learner:
            return
        for skill in self._skills.list_all():
            uses = skill.get("use_count", 0)
            conf = skill.get("confidence", 1.0)
            if uses >= 3 and conf < 0.5:
                try:
                    self._learner.learn(skill["name"], skill["domain"], force=True)
                except Exception:
                    pass

    def _write_status(self):
        try:
            mem_count = self._memory.semantic.count() if self._memory else 0
            skill_count = len(self._skills.list_all()) if self._skills else 0
            status = {
                "agent": "NexusAgent",
                "version": "2.0.0",
                "status": "running",
                "heartbeat_count": self._beat_count,
                "started_at": self._started_at,
                "last_beat": datetime.utcnow().isoformat(),
                "interval_seconds": self._interval,
                "semantic_memories": mem_count,
                "skills_learned": skill_count,
            }
            self._status_path.parent.mkdir(parents=True, exist_ok=True)
            self._status_path.write_text(json.dumps(status, indent=2))
        except Exception:
            pass
