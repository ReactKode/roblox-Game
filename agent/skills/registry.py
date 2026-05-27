"""Skill registry — persistent JSON skill library with confidence, versioning, and search."""
import json
from datetime import datetime
from pathlib import Path

DOMAIN_KEYWORDS = {
    "blender":         ["blender", "3d model", "mesh", "bpy", "sculpt", "rig", "animation3d", "uv map"],
    "unity":           ["unity", "unity3d", "monobehaviour", "c# game", "unity engine", "prefab"],
    "unreal_engine":   ["unreal", "ue5", "ue4", "blueprint", "unreal engine", "nanite", "lumen"],
    "godot":           ["godot", "gdscript", "godot engine", "godot4", "nodes"],
    "game_dev":        ["game development", "game design", "game mechanics", "level design", "game loop"],
    "python":          ["python", "pandas", "numpy", "flask", "fastapi", "django", "pip"],
    "javascript":      ["javascript", "typescript", "nodejs", "node.js", "react", "vue", "svelte", "npm"],
    "machine_learning":["machine learning", "neural network", "pytorch", "tensorflow", "llm", "fine-tun"],
    "business":        ["startup", "company", "revenue", "marketing", "saas", "monetize", "business plan"],
    "research":        ["research", "analysis", "summarize", "literature review", "fact-check"],
    "web":             ["web scraping", "html", "css", "rest api", "graphql", "http"],
    "data":            ["sql", "database", "data analysis", "visualization", "csv", "pandas"],
}

_SCHEMA = {
    "name": "", "description": "", "domain": "general",
    "version": 1, "created_at": "", "last_used": "", "use_count": 0,
    "procedure": "", "code_template": "", "tools_required": [],
    "examples": [], "tags": [],
    "confidence": 0.5, "validated": False, "validation_issues": [],
    "metadata": {},
}


class SkillRegistry:
    def __init__(self, skill_dir: Path):
        self._dir = Path(skill_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _safe_name(self, name: str) -> str:
        return name.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")[:80]

    def _path(self, name: str) -> Path:
        return self._dir / f"{self._safe_name(name)}.json"

    # ── write ──────────────────────────────────────────────────────────────

    def register(self, skill: dict) -> dict:
        existing = self.get(skill.get("name", ""))
        if existing:
            skill["version"] = existing.get("version", 1) + 1
            skill["created_at"] = existing.get("created_at", datetime.utcnow().isoformat())
        else:
            skill.setdefault("created_at", datetime.utcnow().isoformat())
            skill.setdefault("version", 1)

        full = {**_SCHEMA, **skill, "last_used": datetime.utcnow().isoformat()}
        self._path(full["name"]).write_text(json.dumps(full, indent=2))
        return full

    def update_usage(self, name: str):
        s = self.get(name)
        if s:
            s["use_count"] = s.get("use_count", 0) + 1
            s["last_used"] = datetime.utcnow().isoformat()
            self._path(name).write_text(json.dumps(s, indent=2))

    def update_confidence(self, name: str, confidence: float, issues: list = None):
        s = self.get(name)
        if s:
            s["confidence"] = round(confidence, 2)
            s["validated"] = True
            s["validation_issues"] = issues or []
            self._path(name).write_text(json.dumps(s, indent=2))

    def delete(self, name: str) -> bool:
        p = self._path(name)
        if p.exists():
            p.unlink()
            return True
        return False

    # ── read ───────────────────────────────────────────────────────────────

    def get(self, name: str) -> dict | None:
        p = self._path(name)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
        return None

    def list_all(self, domain: str = None) -> list[dict]:
        skills = []
        for p in sorted(self._dir.glob("*.json")):
            try:
                s = json.loads(p.read_text())
                if domain is None or s.get("domain") == domain:
                    skills.append(s)
            except Exception:
                continue
        return sorted(skills, key=lambda s: -s.get("use_count", 0))

    def search(self, query: str, domain: str = None, min_confidence: float = 0.0) -> list[dict]:
        q = query.lower()
        candidates = self.list_all(domain=domain)
        scored = []
        for s in candidates:
            if s.get("confidence", 0.5) < min_confidence:
                continue
            text = f"{s['name']} {s['description']} {s['domain']} {' '.join(s.get('tags', []))}".lower()
            score = sum(text.count(w) for w in q.split() if len(w) > 2)
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored]

    # ── helpers ────────────────────────────────────────────────────────────

    def detect_domain(self, topic: str) -> str:
        t = topic.lower()
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(kw in t for kw in keywords):
                return domain
        return "general"

    def confidence_summary(self) -> dict:
        skills = self.list_all()
        if not skills:
            return {}
        return {s["name"]: s.get("confidence", 0.5) for s in skills}
