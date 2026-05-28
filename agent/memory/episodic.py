"""Episodic memory — SQLite log of every task, action, reflection, and outcome."""
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from ..core.logger import get_logger

log = get_logger(__name__)


class EpisodicMemory:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()   # serializes all DB writes across threads
        self._init_db()

    def _init_db(self):
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT NOT NULL,
                    task        TEXT DEFAULT '',
                    action      TEXT DEFAULT '',
                    observation TEXT DEFAULT '',
                    outcome     TEXT DEFAULT '',
                    skill_used  TEXT DEFAULT '',
                    success     INTEGER DEFAULT 1,
                    is_reflection INTEGER DEFAULT 0,
                    tags        TEXT DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS idx_ts   ON episodes(timestamp);
                CREATE INDEX IF NOT EXISTS idx_refl ON episodes(is_reflection);
                CREATE INDEX IF NOT EXISTS idx_skill ON episodes(skill_used);
            """)
            self._conn.commit()

    def log(self, task: str = "", action: str = "", observation: str = "",
            outcome: str = "", skill_used: str = "", success: bool = True,
            is_reflection: bool = False, tags: list | None = None) -> int:
        try:
            with self._lock:
                cur = self._conn.execute(
                    "INSERT INTO episodes "
                    "(timestamp,task,action,observation,outcome,skill_used,success,is_reflection,tags) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        datetime.utcnow().isoformat(),
                        task[:2000],       # was 500 — tasks can be long
                        action[:1000],
                        observation[:4000],  # was 2000 — reflections need space
                        outcome[:2000],      # was 500
                        skill_used,
                        int(success),
                        int(is_reflection),
                        json.dumps(tags or []),
                    ),
                )
                self._conn.commit()
                return cur.lastrowid
        except Exception:
            log.exception("EpisodicMemory.log failed")
            return -1

    def recall(self, query: str = "", limit: int = 10,
               include_reflections: bool = True) -> list[dict]:
        try:
            base = "SELECT * FROM episodes"
            conditions: list[str] = []
            params: list = []
            if not include_reflections:
                conditions.append("is_reflection = 0")
            if query:
                # Escape LIKE wildcards in user query
                safe_q = query.replace("%", r"\%").replace("_", r"\_")
                conditions.append(
                    "(task LIKE ? ESCAPE '\\' OR action LIKE ? ESCAPE '\\' "
                    "OR observation LIKE ? ESCAPE '\\' OR outcome LIKE ? ESCAPE '\\')"
                )
                params.extend([f"%{safe_q}%"] * 4)
            where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
            with self._lock:
                rows = self._conn.execute(
                    f"{base}{where} ORDER BY timestamp DESC LIMIT ?", params + [limit]
                ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except Exception:
            log.exception("EpisodicMemory.recall failed")
            return []

    def recall_reflections(self, query: str = "", limit: int = 5) -> list[dict]:
        try:
            with self._lock:
                if query:
                    safe_q = query.replace("%", r"\%").replace("_", r"\_")
                    rows = self._conn.execute(
                        "SELECT * FROM episodes WHERE is_reflection=1 "
                        "AND (task LIKE ? ESCAPE '\\' OR observation LIKE ? ESCAPE '\\') "
                        "ORDER BY timestamp DESC LIMIT ?",
                        (f"%{safe_q}%", f"%{safe_q}%", limit),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT * FROM episodes WHERE is_reflection=1 "
                        "ORDER BY timestamp DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except Exception:
            log.exception("EpisodicMemory.recall_reflections failed")
            return []

    def stats(self) -> dict:
        try:
            with self._lock:
                total = self._conn.execute(
                    "SELECT COUNT(*) FROM episodes WHERE is_reflection=0"
                ).fetchone()[0]
                successful = self._conn.execute(
                    "SELECT COUNT(*) FROM episodes WHERE is_reflection=0 AND success=1"
                ).fetchone()[0]
                reflections = self._conn.execute(
                    "SELECT COUNT(*) FROM episodes WHERE is_reflection=1"
                ).fetchone()[0]
                skills_used = self._conn.execute(
                    "SELECT skill_used, COUNT(*) cnt FROM episodes WHERE skill_used != '' "
                    "GROUP BY skill_used ORDER BY cnt DESC LIMIT 8"
                ).fetchall()
            return {
                "total_tasks": total,
                "successful": successful,
                "failed": total - successful,
                "reflections": reflections,
                "top_skills": [{"skill": s, "count": c} for s, c in skills_used],
            }
        except Exception:
            log.exception("EpisodicMemory.stats failed")
            return {"total_tasks": 0, "successful": 0, "failed": 0,
                    "reflections": 0, "top_skills": []}

    def _row_to_dict(self, row) -> dict:
        cols = ["id", "timestamp", "task", "action", "observation", "outcome",
                "skill_used", "success", "is_reflection", "tags"]
        d = dict(zip(cols, row))
        try:
            d["tags"] = json.loads(d.get("tags", "[]"))
        except Exception:
            d["tags"] = []
        return d

    def close(self):
        try:
            with self._lock:
                self._conn.close()
        except Exception:
            log.warning("EpisodicMemory.close: connection already closed")
