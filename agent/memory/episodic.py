"""Episodic memory - SQLite log of agent actions and outcomes."""
import sqlite3
from datetime import datetime
from pathlib import Path


class EpisodicMemory:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                task TEXT,
                action TEXT,
                observation TEXT,
                outcome TEXT,
                skill_used TEXT,
                success INTEGER DEFAULT 1
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_skill ON episodes(skill_used)")
        self._conn.commit()

    def log(self, task: str = "", action: str = "", observation: str = "",
            outcome: str = "", skill_used: str = "", success: bool = True) -> int:
        cur = self._conn.execute(
            "INSERT INTO episodes (timestamp, task, action, observation, outcome, skill_used, success) VALUES (?,?,?,?,?,?,?)",
            (datetime.utcnow().isoformat(), task[:500], action[:1000], observation[:2000], outcome[:500], skill_used, int(success))
        )
        self._conn.commit()
        return cur.lastrowid

    def recall(self, query: str = "", limit: int = 10) -> list[dict]:
        if query:
            rows = self._conn.execute(
                "SELECT * FROM episodes WHERE task LIKE ? OR action LIKE ? OR outcome LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", f"%{query}%", limit)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        cols = ["id", "timestamp", "task", "action", "observation", "outcome", "skill_used", "success"]
        return [dict(zip(cols, row)) for row in rows]

    def stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        successful = self._conn.execute("SELECT COUNT(*) FROM episodes WHERE success=1").fetchone()[0]
        skills = self._conn.execute(
            "SELECT skill_used, COUNT(*) as cnt FROM episodes WHERE skill_used != '' GROUP BY skill_used ORDER BY cnt DESC LIMIT 10"
        ).fetchall()
        return {
            "total_episodes": total,
            "successful": successful,
            "failed": total - successful,
            "top_skills": [{"skill": s, "count": c} for s, c in skills],
        }

    def close(self):
        self._conn.close()
