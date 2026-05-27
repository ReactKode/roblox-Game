"""Episodic memory — SQLite log of every task, action, reflection, and outcome."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path


class EpisodicMemory:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_db()

    def _init_db(self):
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
            CREATE INDEX IF NOT EXISTS idx_ts  ON episodes(timestamp);
            CREATE INDEX IF NOT EXISTS idx_refl ON episodes(is_reflection);
            CREATE INDEX IF NOT EXISTS idx_skill ON episodes(skill_used);
        """)
        self._conn.commit()

    def log(self, task: str = "", action: str = "", observation: str = "",
            outcome: str = "", skill_used: str = "", success: bool = True,
            is_reflection: bool = False, tags: list | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO episodes (timestamp,task,action,observation,outcome,skill_used,success,is_reflection,tags) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                datetime.utcnow().isoformat(),
                task[:500], action[:1000], observation[:2000],
                outcome[:500], skill_used, int(success),
                int(is_reflection), json.dumps(tags or []),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def recall(self, query: str = "", limit: int = 10, include_reflections: bool = True) -> list[dict]:
        base = "SELECT * FROM episodes"
        conditions = []
        params: list = []
        if not include_reflections:
            conditions.append("is_reflection = 0")
        if query:
            conditions.append("(task LIKE ? OR action LIKE ? OR observation LIKE ? OR outcome LIKE ?)")
            params.extend([f"%{query}%"] * 4)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        rows = self._conn.execute(
            f"{base}{where} ORDER BY timestamp DESC LIMIT ?", params + [limit]
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def recall_reflections(self, query: str = "", limit: int = 5) -> list[dict]:
        if query:
            rows = self._conn.execute(
                "SELECT * FROM episodes WHERE is_reflection=1 AND (task LIKE ? OR observation LIKE ?) "
                "ORDER BY timestamp DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM episodes WHERE is_reflection=1 ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM episodes WHERE is_reflection=0").fetchone()[0]
        successful = self._conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE is_reflection=0 AND success=1"
        ).fetchone()[0]
        reflections = self._conn.execute("SELECT COUNT(*) FROM episodes WHERE is_reflection=1").fetchone()[0]
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
        self._conn.close()
