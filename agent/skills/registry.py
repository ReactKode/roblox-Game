"""Skill registry - persistent storage and retrieval of learned skills."""
import json
from datetime import datetime
from pathlib import Path


SKILL_SCHEMA = {
    "name": "",
    "description": "",
    "domain": "general",
    "version": "1.0.0",
    "created_at": "",
    "last_used": "",
    "use_count": 0,
    "procedure": "",
    "code_template": "",
    "tools_required": [],
    "examples": [],
    "tags": [],
    "metadata": {},
}

DOMAIN_KEYWORDS = {
    "blender": ["blender", "3d modeling", "mesh", "bpy", "sculpt", "rigging", "animation3d"],
    "unity": ["unity", "unity3d", "c# game", "monobehaviour", "unity engine"],
    "unreal_engine": ["unreal", "ue5", "ue4", "blueprint", "unreal engine", "cpp game"],
    "godot": ["godot", "gdscript", "godot engine", "godot4"],
    "game_dev": ["game development", "game design", "game mechanics", "level design"],
    "python": ["python", "pandas", "numpy", "flask", "fastapi", "django"],
    "javascript": ["javascript", "typescript", "nodejs", "react", "vue", "svelte"],
    "machine_learning": ["machine learning", "ml", "neural network", "pytorch", "tensorflow", "ai model"],
    "business": ["business", "startup", "company", "revenue", "marketing", "saas"],
    "research": ["research", "analysis", "summarize", "study"],
    "web": ["web scraping", "html", "css", "api design", "rest api"],
}


class SkillRegistry:
    def __init__(self, skill_dir: Path):
        self._dir = Path(skill_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        safe = name.lower().replace(" ", "_").replace("/", "_")
        return self._dir / f"{safe}.json"

    def register(self, skill: dict) -> dict:
        skill = {**SKILL_SCHEMA, **skill}
        if not skill["created_at"]:
            skill["created_at"] = datetime.utcnow().isoformat()
        skill["last_used"] = datetime.utcnow().isoformat()
        self._path(skill["name"]).write_text(json.dumps(skill, indent=2))
        return skill

    def get(self, name: str) -> dict | None:
        p = self._path(name)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def update_usage(self, name: str) -> None:
        skill = self.get(name)
        if skill:
            skill["use_count"] += 1
            skill["last_used"] = datetime.utcnow().isoformat()
            self._path(name).write_text(json.dumps(skill, indent=2))

    def list_all(self, domain: str = None) -> list[dict]:
        skills = []
        for p in sorted(self._dir.glob("*.json")):
            try:
                s = json.loads(p.read_text())
                if domain is None or s.get("domain") == domain:
                    skills.append(s)
            except Exception:
                continue
        return skills

    def search(self, query: str, domain: str = None) -> list[dict]:
        query_lower = query.lower()
        all_skills = self.list_all(domain=domain)
        scored = []
        for s in all_skills:
            score = 0
            text = f"{s['name']} {s['description']} {s['domain']} {' '.join(s.get('tags', []))}".lower()
            for word in query_lower.split():
                if word in text:
                    score += text.count(word)
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored]

    def detect_domain(self, topic: str) -> str:
        topic_lower = topic.lower()
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(kw in topic_lower for kw in keywords):
                return domain
        return "general"

    def delete(self, name: str) -> bool:
        p = self._path(name)
        if p.exists():
            p.unlink()
            return True
        return False
