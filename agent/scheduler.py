"""Natural language task scheduler backed by APScheduler."""
import re
import threading
import uuid
from datetime import datetime
from typing import Callable

from .core.logger import get_logger

log = get_logger(__name__)

_APSCHEDULER_AVAILABLE = False
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    _APSCHEDULER_AVAILABLE = True
except ImportError:
    pass

# ── Pattern tables ─────────────────────────────────────────────────────────

_INTERVAL_PATTERNS: list[tuple[str, Callable]] = [
    (r"every\s+(\d+)\s+seconds?", lambda m: {"seconds": int(m.group(1))}),
    (r"every\s+(\d+)\s+minutes?", lambda m: {"minutes": int(m.group(1))}),
    (r"every\s+(\d+)\s+hours?",   lambda m: {"hours": int(m.group(1))}),
    (r"every\s+(\d+)\s+days?",    lambda m: {"days": int(m.group(1))}),
    (r"every\s+minute",           lambda m: {"minutes": 1}),
    (r"every\s+hour",             lambda m: {"hours": 1}),
    (r"every\s+day",              lambda m: {"days": 1}),
    (r"\bhourly\b",               lambda m: {"hours": 1}),
    (r"\bdaily\b(?!\s+at)",       lambda m: {"days": 1}),
    (r"\bweekly\b",               lambda m: {"weeks": 1}),
]

_DOW_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4,  "saturday": 5, "sunday": 6,
}
_DOW_PATTERN = r"every\s+(" + "|".join(_DOW_MAP) + r")"


# ── Data class ─────────────────────────────────────────────────────────────

class ScheduledTask:
    def __init__(self, task_id: str, schedule_str: str, task_text: str):
        self.id = task_id
        self.schedule_str = schedule_str
        self.task_text = task_text
        self.created_at = datetime.utcnow().isoformat()
        self.run_count = 0
        self.last_run: str | None = None


# ── Scheduler ──────────────────────────────────────────────────────────────

class TaskScheduler:
    """Schedule recurring tasks using natural language triggers.

    Requires APScheduler: pip install apscheduler
    Falls back to a no-op with a clear error if not installed.
    """

    def __init__(self, agent_fn: Callable[[str], str]):
        self._agent_fn = agent_fn
        self._tasks: dict[str, ScheduledTask] = {}
        self._lock = threading.Lock()
        self._scheduler = None

        if _APSCHEDULER_AVAILABLE:
            self._scheduler = BackgroundScheduler(daemon=True)
            self._scheduler.start()
            log.info("[scheduler] APScheduler started.")
        else:
            log.warning(
                "[scheduler] APScheduler not installed — scheduling disabled. "
                "Install with: pip install apscheduler"
            )

    # ── Public API ───────────────────────────────────────────────────────

    def schedule(self, schedule_desc: str, task_text: str) -> str:
        """Schedule a recurring task. Returns task_id on success or an error string."""
        if not _APSCHEDULER_AVAILABLE:
            return "Scheduling unavailable — install: pip install apscheduler"

        trigger = self._parse_trigger(schedule_desc)
        if trigger is None:
            return (
                f"Could not parse schedule '{schedule_desc}'. "
                "Try: 'every 5 minutes', 'daily at 09:00', 'every Monday at 08:00'."
            )

        task_id = str(uuid.uuid4())[:8]
        task = ScheduledTask(task_id=task_id, schedule_str=schedule_desc, task_text=task_text)

        def _run() -> None:
            task.run_count += 1
            task.last_run = datetime.utcnow().isoformat()
            log.info("[scheduler] Running task %s: %s", task_id, task_text[:80])
            try:
                self._agent_fn(task_text)
            except Exception:
                log.exception("[scheduler] Task %s raised an exception", task_id)

        with self._lock:
            self._scheduler.add_job(_run, trigger=trigger, id=task_id, replace_existing=True)
            self._tasks[task_id] = task

        log.info("[scheduler] Scheduled task %s (%s): '%s'", task_id, schedule_desc, task_text[:60])
        return task_id

    def cancel(self, task_id: str) -> bool:
        """Cancel a task by ID. Returns True if found and removed."""
        if not _APSCHEDULER_AVAILABLE:
            return False
        with self._lock:
            if task_id not in self._tasks:
                return False
            try:
                self._scheduler.remove_job(task_id)
            except Exception:
                pass
            del self._tasks[task_id]
        log.info("[scheduler] Cancelled task %s", task_id)
        return True

    def list_tasks(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "id": t.id,
                    "schedule": t.schedule_str,
                    "task": t.task_text[:100],
                    "run_count": t.run_count,
                    "last_run": t.last_run,
                    "created_at": t.created_at,
                }
                for t in self._tasks.values()
            ]

    def shutdown(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            log.info("[scheduler] Scheduler stopped.")

    @property
    def available(self) -> bool:
        return _APSCHEDULER_AVAILABLE

    # ── Private ──────────────────────────────────────────────────────────

    def _parse_trigger(self, desc: str):
        """Parse a natural language description → APScheduler trigger, or None."""
        d = desc.lower().strip()

        # Interval patterns
        for pattern, extractor in _INTERVAL_PATTERNS:
            m = re.search(pattern, d)
            if m:
                return IntervalTrigger(**extractor(m))

        # "at HH:MM [am|pm]" — possibly preceded by day-of-week or "daily"
        time_match = re.search(r"at\s+(\d{1,2}):(\d{2})(?:\s*([ap]m))?", d)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            ampm = time_match.group(3)
            if ampm == "pm" and hour < 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0

            dow_m = re.search(_DOW_PATTERN, d)
            if dow_m:
                return CronTrigger(day_of_week=_DOW_MAP[dow_m.group(1)], hour=hour, minute=minute)
            return CronTrigger(hour=hour, minute=minute)

        # Day of week without time — default 09:00
        dow_m = re.search(_DOW_PATTERN, d)
        if dow_m:
            return CronTrigger(day_of_week=_DOW_MAP[dow_m.group(1)], hour=9, minute=0)

        return None
