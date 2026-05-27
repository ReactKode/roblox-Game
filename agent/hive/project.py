"""Project data model — the shared state all agents read and write to."""
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Deliverable:
    role: str
    title: str
    content: str
    approved: bool = False
    revision_notes: str = ""
    word_count: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ProjectTask:
    role: str
    description: str
    status: str = "pending"   # pending | running | complete | failed | revision
    deliverable: Deliverable | None = None
    started_at: str = ""
    completed_at: str = ""


@dataclass
class Project:
    id: str
    name: str
    tagline: str
    type: str                      # game | app | saas | website | tool
    genre: str
    vision: str
    target_audience: str
    tech_stack: list[str]
    unique_selling_points: list[str]
    core_features: list[str]
    revenue_model: str
    scope: str                     # indie | mid | large | AAA
    roles_needed: list[str]
    goal: str                      # original user request
    tasks: list[ProjectTask] = field(default_factory=list)
    deliverables: dict[str, Deliverable] = field(default_factory=dict)
    status: str = "planning"       # planning | active | review | complete
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: str = ""

    def summary(self) -> str:
        """Short context string injected into every specialist's prompt."""
        usps = "\n".join(f"  • {u}" for u in self.unique_selling_points[:3])
        feats = ", ".join(self.core_features[:5])
        return (
            f"**{self.name}** — {self.tagline}\n"
            f"Type: {self.type} | Genre: {self.genre} | Scope: {self.scope}\n"
            f"Vision: {self.vision}\n"
            f"Audience: {self.target_audience}\n"
            f"Tech stack: {', '.join(self.tech_stack)}\n"
            f"Core features: {feats}\n"
            f"Revenue: {self.revenue_model}\n"
            f"Unique selling points:\n{usps}"
        )

    def team_context(self, exclude_role: str = "") -> str:
        """All completed deliverables as context for later specialists."""
        done = {r: d for r, d in self.deliverables.items()
                if d.approved and r != exclude_role}
        if not done:
            return "No team output yet — you are first."
        parts = []
        for role, deliv in done.items():
            preview = deliv.content[:800]
            parts.append(f"### {deliv.title}\n{preview}\n[...{deliv.word_count} words total...]")
        return "\n\n---\n\n".join(parts)

    def save(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save each deliverable as its own markdown file
        for role, deliv in self.deliverables.items():
            safe_role = role.replace("_", "-")
            (output_dir / f"{safe_role}.md").write_text(
                f"# {deliv.title}\n\n> Project: {self.name}\n\n{deliv.content}"
            )
        # Save project manifest
        manifest = {
            "id": self.id,
            "name": self.name,
            "tagline": self.tagline,
            "type": self.type,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "roles": list(self.deliverables.keys()),
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "tagline": self.tagline,
            "type": self.type,
            "genre": self.genre,
            "vision": self.vision,
            "status": self.status,
            "deliverables": {r: {"title": d.title, "word_count": d.word_count, "approved": d.approved}
                             for r, d in self.deliverables.items()},
        }
