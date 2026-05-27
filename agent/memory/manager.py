"""Memory manager — coordinates short-term, semantic long-term, and episodic memory."""
from pathlib import Path
from .short_term import ShortTermMemory
from .semantic import SemanticMemory
from .episodic import EpisodicMemory


class MemoryManager:
    def __init__(self, data_dir: Path, short_term_limit: int = 20, backend: str = "auto"):
        self.short = ShortTermMemory(limit=short_term_limit)
        self.semantic = SemanticMemory(data_dir / "memory", backend=backend)
        self.episodic = EpisodicMemory(data_dir / "memory" / "episodes.db")
        self._embed_fn = None

    def set_embed_fn(self, fn):
        self._embed_fn = fn

    def _embed(self, text: str) -> list[float]:
        if self._embed_fn:
            try:
                return self._embed_fn(text)
            except Exception:
                pass
        return []

    # ── write ──────────────────────────────────────────────────────────────

    def remember(self, content: str, role: str = "assistant",
                 memory_type: str = "conversation", importance: float = 0.4,
                 metadata: dict = None) -> None:
        self.short.add(role, content)
        if len(content) > 20:
            emb = self._embed(content)
            self.semantic.add(content, memory_type=memory_type, importance=importance,
                              embedding=emb, metadata=metadata or {})

    def store_fact(self, text: str, importance: float = 0.7, metadata: dict = None) -> str:
        emb = self._embed(text)
        return self.semantic.add(text, memory_type="fact", importance=importance,
                                 embedding=emb, metadata=metadata or {})

    def store_lesson(self, text: str) -> str:
        emb = self._embed(text)
        return self.semantic.add(text, memory_type="lesson", importance=0.9, embedding=emb)

    def store_research(self, text: str, source: str = "") -> str:
        emb = self._embed(text)
        return self.semantic.add(text, memory_type="research", importance=0.6,
                                 embedding=emb, metadata={"source": source})

    def log_episode(self, **kwargs) -> int:
        return self.episodic.log(**kwargs)

    # ── read ───────────────────────────────────────────────────────────────

    def recall(self, query: str, k: int = 5, memory_type: str = None):
        emb = self._embed(query)
        return self.semantic.search(query, emb, k, memory_type)

    def recall_lessons(self, query: str = "", k: int = 3):
        emb = self._embed(query) if query else []
        return self.semantic.search(query or "lesson learned", emb, k, memory_type="lesson")

    def recall_reflections(self, query: str, limit: int = 3) -> list[str]:
        episodes = self.episodic.recall_reflections(query, limit)
        return [e["observation"] for e in episodes if e.get("observation")]

    # ── consolidation ──────────────────────────────────────────────────────

    def consolidate(self) -> int:
        """Move important short-term messages into semantic memory."""
        moved = 0
        for m in self.short.get_messages(include_system=False):
            content = m["content"]
            if len(content) > 80 and m["role"] == "assistant":
                emb = self._embed(content)
                self.semantic.add(content, memory_type="conversation",
                                  importance=0.3, embedding=emb)
                moved += 1
        return moved

    def build_context(self, system_prompt: str) -> list[dict]:
        msgs = [{"role": "system", "content": system_prompt}]
        msgs.extend(self.short.get_messages(include_system=False))
        return msgs

    def get_relevant_context(self, query: str, k: int = 4) -> str:
        entries = self.recall(query, k)
        if not entries:
            return "None"
        lines = []
        for e in entries:
            tag = f"[{e.memory_type}|importance={e.importance:.1f}]"
            lines.append(f"{tag} {e.text[:250]}")
        return "\n".join(lines)

    def stats(self) -> dict:
        return {
            "short_term_messages": len(self.short),
            "semantic_entries": self.semantic.count(),
            "semantic_backend": self.semantic.backend_name,
            "episodes": self.episodic.stats(),
        }
