"""Agent self-model — persistent identity, capability tracking, lessons learned."""
import json
from datetime import datetime
from pathlib import Path


_DEFAULT = {
    "name": "NexusAgent",
    "version": "2.0.0",
    "created_at": "",
    "stats": {
        "tasks_completed": 0,
        "tasks_failed": 0,
        "skills_learned": 0,
        "reflections_stored": 0,
        "total_iterations": 0,
    },
    "strengths": [],
    "weaknesses": [],
    "skill_confidence": {},
    "domain_experience": {},
    "lessons_learned": [],
    "last_updated": "",
}


class SelfModel:
    """Persists to data/profile.json. Injected into every agent system prompt."""

    def __init__(self, data_dir: Path):
        self._path = data_dir / "profile.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except Exception:
                pass
        d = dict(_DEFAULT)
        d["created_at"] = datetime.utcnow().isoformat()
        return d

    def _save(self):
        self._data["last_updated"] = datetime.utcnow().isoformat()
        self._path.write_text(json.dumps(self._data, indent=2))

    # --- mutation methods ---

    def record_task(self, success: bool, iterations: int, domain: str = "general"):
        s = self._data["stats"]
        if success:
            s["tasks_completed"] += 1
        else:
            s["tasks_failed"] += 1
        s["total_iterations"] += iterations
        exp = self._data["domain_experience"]
        exp[domain] = exp.get(domain, 0) + 1
        self._save()

    def record_skill(self, name: str, domain: str, confidence: float):
        self._data["stats"]["skills_learned"] += 1
        self._data["skill_confidence"][name] = round(confidence, 2)
        self._save()

    def apply_reflection(self, lesson: str, what_failed: str, domain: str, confidence: float):
        lessons = self._data["lessons_learned"]
        if lesson and lesson not in lessons:
            lessons.append(lesson)
            if len(lessons) > 25:
                lessons.pop(0)
        self._data["stats"]["reflections_stored"] += 1

        if confidence >= 0.8 and domain and domain != "general":
            s = self._data["strengths"]
            if domain not in s:
                s.append(domain)
                if len(s) > 10:
                    s.pop(0)

        if confidence < 0.5 and what_failed and what_failed != "nothing":
            w = self._data["weaknesses"]
            if domain not in w:
                w.append(domain)
                if len(w) > 10:
                    w.pop(0)

        self._save()

    # --- read methods ---

    def get_prompt_context(self) -> str:
        d = self._data
        s = d["stats"]
        total = s["tasks_completed"] + s["tasks_failed"]
        rate = f"{100*s['tasks_completed']//max(total,1)}%" if total else "no history yet"
        lines = [f"Tasks: {s['tasks_completed']} completed, {s['tasks_failed']} failed ({rate} success rate)"]
        if d["strengths"]:
            lines.append(f"Proven strengths: {', '.join(d['strengths'][:6])}")
        if d["weaknesses"]:
            lines.append(f"Known weaknesses (extra care needed): {', '.join(d['weaknesses'][:4])}")
        recent = d["lessons_learned"][-4:]
        if recent:
            lines.append("Key lessons:\n" + "\n".join(f"  • {l}" for l in recent))
        return "\n".join(lines)

    def skill_confidence(self, name: str) -> float:
        return self._data["skill_confidence"].get(name, 0.5)

    @property
    def stats(self) -> dict:
        return self._data["stats"]

    @property
    def profile(self) -> dict:
        return self._data
