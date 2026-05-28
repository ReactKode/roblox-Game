"""Heartbeat daemon — background intelligence loop.

Every beat:
  1. Consolidates short-term memory into semantic store
  2. Re-learns low-confidence high-usage skills (with 24h cooldown per skill)
  3. Stores lessons from recent episodes into semantic memory
  4. Writes agent status file
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from .logger import get_logger

log = get_logger(__name__)


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

        # Cooldown tracking: skill_name → ISO timestamp of last re-learn attempt
        self._relearn_cooldowns: dict[str, str] = {}
        self._RELEARN_COOLDOWN_SECONDS = 86400  # 24 hours

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._started_at = datetime.now(tz=timezone.utc).isoformat()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="NexusHeartbeat")
        self._thread.start()
        log.info("Heartbeat started (interval=%ds)", self._interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        log.info("Heartbeat stopped after %d beats", self._beat_count)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def beat_count(self) -> int:
        return self._beat_count

    @property
    def uptime_seconds(self) -> float:
        if not self._started_at:
            return 0.0
        s = datetime.fromisoformat(self._started_at)
        if s.tzinfo is None:
            s = s.replace(tzinfo=timezone.utc)
        return (datetime.now(tz=timezone.utc) - s).total_seconds()

    # ── loop ─────────────────────────────────────────────────────────────────

    def _loop(self):
        while not self._stop.wait(timeout=self._interval):
            self._beat()

    def _beat(self):
        self._beat_count += 1
        log.debug("Heartbeat #%d", self._beat_count)

        self._safe_run("consolidate_memory", self._consolidate_memory)

        if self._beat_count % 3 == 0:
            self._safe_run("promote_lessons", self._promote_lessons)

        if self._beat_count % 6 == 0:
            self._safe_run("refresh_weak_skills", self._refresh_weak_skills)

        self._safe_run("write_status", self._write_status)

    def _safe_run(self, name: str, fn):
        try:
            fn()
        except Exception:
            log.exception("Heartbeat task '%s' raised an exception", name)

    def _consolidate_memory(self):
        if not self._memory:
            return
        moved = self._memory.consolidate()
        if moved:
            log.debug("Consolidated %d short-term messages into semantic memory", moved)

    def _promote_lessons(self):
        """Move high-value lessons from episodic to semantic memory."""
        if not self._memory:
            return
        reflections = self._memory.episodic.recall_reflections(limit=10)
        promoted = 0
        for r in reflections:
            obs = r.get("observation", "")
            # Observation format: "lesson=<text> | next_time=<text>"
            if "lesson=" not in obs:
                continue
            try:
                lesson_text = obs.split("lesson=", 1)[1].split("|")[0].strip()
            except (IndexError, ValueError):
                continue
            if len(lesson_text) > 20:
                self._memory.store_lesson(lesson_text)
                promoted += 1
        if promoted:
            log.debug("Promoted %d lessons to semantic memory", promoted)

    def _refresh_weak_skills(self):
        """Re-learn skills that are heavily used but have low confidence.

        A 24-hour cooldown prevents infinite re-learning loops when a skill
        stubbornly stays below the confidence threshold.
        """
        if not self._skills or not self._learner:
            return
        now = datetime.now(tz=timezone.utc)
        for skill in self._skills.list_all():
            uses = skill.get("use_count", 0)
            conf = skill.get("confidence", 1.0)
            if uses < 3 or conf >= 0.5:
                continue
            name = skill["name"]
            # Check cooldown
            last_str = self._relearn_cooldowns.get(name)
            if last_str:
                try:
                    last = datetime.fromisoformat(last_str)
                    if last.tzinfo is None:
                        last = last.replace(tzinfo=timezone.utc)
                    if (now - last).total_seconds() < self._RELEARN_COOLDOWN_SECONDS:
                        log.debug("Skipping re-learn for '%s' (cooldown active)", name)
                        continue
                except (ValueError, TypeError):
                    pass
            log.info("Re-learning weak skill: '%s' (conf=%.2f, uses=%d)", name, conf, uses)
            self._relearn_cooldowns[name] = now.isoformat()
            try:
                self._learner.learn(name, skill.get("domain", "general"), force=True)
            except Exception:
                log.exception("Re-learn failed for skill '%s'", name)

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
                "last_beat": datetime.now(tz=timezone.utc).isoformat(),
                "interval_seconds": self._interval,
                "semantic_memories": mem_count,
                "skills_learned": skill_count,
            }
            self._status_path.parent.mkdir(parents=True, exist_ok=True)
            self._status_path.write_text(json.dumps(status, indent=2))
        except Exception:
            log.exception("Failed to write status file")
