"""Memory manager - coordinates all memory subsystems."""
from pathlib import Path
from .short_term import ShortTermMemory
from .long_term import LongTermMemory
from .episodic import EpisodicMemory


class MemoryManager:
    def __init__(self, data_dir: Path, short_term_limit: int = 20, long_term_backend: str = "chromadb"):
        self.short = ShortTermMemory(limit=short_term_limit)
        self.long = LongTermMemory(data_dir / "memory", backend=long_term_backend)
        self.episodic = EpisodicMemory(data_dir / "memory" / "episodes.db")
        self._embed_fn = None  # injected by agent after model is ready

    def set_embed_fn(self, fn):
        self._embed_fn = fn

    def _embed(self, text: str) -> list[float]:
        if self._embed_fn:
            try:
                return self._embed_fn(text)
            except Exception:
                pass
        return []

    def remember(self, content: str, role: str = "assistant", store_long_term: bool = True, metadata: dict = None) -> None:
        self.short.add(role, content)
        if store_long_term and len(content) > 20:
            embedding = self._embed(content)
            self.long.add(content, embedding, metadata or {})

    def recall(self, query: str, k: int = 5) -> list[dict]:
        embedding = self._embed(query)
        return self.long.search(query, embedding, k)

    def recall_recent_episodes(self, query: str = "", limit: int = 5) -> list[dict]:
        return self.episodic.recall(query, limit)

    def log_episode(self, **kwargs) -> int:
        return self.episodic.log(**kwargs)

    def consolidate(self) -> int:
        """Move recent short-term messages into long-term storage."""
        msgs = self.short.get_messages(include_system=False)
        moved = 0
        for m in msgs:
            if len(m["content"]) > 50:
                emb = self._embed(m["content"])
                self.long.add(m["content"], emb, {"role": m["role"], "source": "short_term_consolidation"})
                moved += 1
        return moved

    def build_context_messages(self, system_prompt: str) -> list[dict]:
        msgs = [{"role": "system", "content": system_prompt}]
        msgs.extend(self.short.get_messages(include_system=False))
        return msgs

    def stats(self) -> dict:
        return {
            "short_term_messages": len(self.short),
            "long_term_entries": self.long.count(),
            "episodes": self.episodic.stats(),
        }
