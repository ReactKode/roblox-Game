"""Semantic long-term memory.
Primary backend: ChromaDB (vector search).
Fallback: JSON file with TF-IDF keyword similarity — works fully offline with no embedding model.
"""
import json
import math
import uuid
from collections import Counter
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


MemoryType = Literal["fact", "skill", "lesson", "reflection", "conversation", "research"]


@dataclass
class MemoryEntry:
    id: str
    text: str
    memory_type: MemoryType
    importance: float
    score: float = 0.0
    metadata: dict = field(default_factory=dict)
    created_at: str = ""
    access_count: int = 0


# ── TF-IDF helpers ──────────────────────────────────────────────────────────

_STOP_WORDS = {"the","a","an","is","are","was","were","be","been","to","of","and",
               "in","that","it","for","on","with","as","by","at","from","this","have",
               "has","had","not","but","or","if","so","do","did","can","will","just","i"}

def _tokenize(text: str) -> list[str]:
    import re
    tokens = re.findall(r"\b[a-z][a-z0-9_]{2,}\b", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS]

def _tfidf_score(query: str, text: str) -> float:
    q_tokens = set(_tokenize(query))
    t_tokens = _tokenize(text)
    if not q_tokens or not t_tokens:
        return 0.0
    t_freq = Counter(t_tokens)
    total = len(t_tokens)
    score = 0.0
    for term in q_tokens:
        if term in t_freq:
            tf = t_freq[term] / total
            # Boost longer terms (more specific)
            idf_boost = min(len(term) / 5.0, 2.0)
            score += tf * idf_boost
    return round(score, 4)

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def _is_duplicate(new_text: str, existing_texts: list[str], threshold: float = 0.85) -> bool:
    new_tokens = set(_tokenize(new_text))
    for existing in existing_texts[-50:]:  # Only check recent 50
        ex_tokens = set(_tokenize(existing))
        if not new_tokens or not ex_tokens:
            continue
        overlap = len(new_tokens & ex_tokens) / len(new_tokens | ex_tokens)
        if overlap >= threshold:
            return True
    return False


# ── JSON fallback store ─────────────────────────────────────────────────────

class _JSONStore:
    def __init__(self, path: Path):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict] = []
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                self._records = json.loads(self._path.read_text())
            except Exception:
                self._records = []

    def _save(self):
        self._path.write_text(json.dumps(self._records, indent=2))

    def add(self, text: str, embedding: list[float], metadata: dict) -> str:
        existing_texts = [r["text"] for r in self._records]
        if _is_duplicate(text, existing_texts):
            return "duplicate"
        rid = str(uuid.uuid4())
        self._records.append({
            "id": rid,
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
            "created_at": datetime.utcnow().isoformat(),
            "access_count": 0,
        })
        self._save()
        return rid

    def search(self, query: str, query_embedding: list[float], k: int,
               memory_type: str | None) -> list[dict]:
        candidates = self._records
        if memory_type:
            candidates = [r for r in candidates if r.get("metadata", {}).get("memory_type") == memory_type]

        scored = []
        use_vectors = bool(query_embedding)
        for r in candidates:
            if use_vectors and r.get("embedding"):
                score = _cosine(query_embedding, r["embedding"])
            else:
                score = _tfidf_score(query, r["text"])
            # Boost by importance
            importance = r.get("metadata", {}).get("importance", 0.5)
            score = score * (0.7 + 0.3 * importance)
            scored.append((score, r))

        scored.sort(key=lambda x: -x[0])
        results = []
        for score, r in scored[:k]:
            r["access_count"] = r.get("access_count", 0) + 1
            results.append({**r, "score": round(score, 4)})
        if results:
            self._save()
        return results

    def count(self) -> int:
        return len(self._records)


# ── ChromaDB store ──────────────────────────────────────────────────────────

class _ChromaStore:
    def __init__(self, persist_dir: Path):
        import chromadb
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._col = self._client.get_or_create_collection(
            "semantic_memory",
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, text: str, embedding: list[float], metadata: dict) -> str:
        rid = str(uuid.uuid4())
        kwargs: dict = {"documents": [text], "ids": [rid], "metadatas": [metadata]}
        if embedding:
            kwargs["embeddings"] = [embedding]
        self._col.add(**kwargs)
        return rid

    def search(self, query: str, query_embedding: list[float], k: int,
               memory_type: str | None) -> list[dict]:
        n = min(k, max(self._col.count(), 1))
        where = {"memory_type": memory_type} if memory_type else None
        try:
            if query_embedding:
                res = self._col.query(query_embeddings=[query_embedding], n_results=n, where=where)
            else:
                res = self._col.query(query_texts=[query], n_results=n, where=where)
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            return [{"text": d, "metadata": m, "score": round(1 - dist, 4)}
                    for d, m, dist in zip(docs, metas, dists)]
        except Exception:
            return []

    def count(self) -> int:
        return self._col.count()


# ── Public interface ─────────────────────────────────────────────────────────

class SemanticMemory:
    """Long-term semantic memory with automatic backend selection."""

    def __init__(self, data_dir: Path, backend: str = "auto"):
        self._data_dir = data_dir
        self._store = self._init(backend)

    def _init(self, backend: str):
        if backend in ("chromadb", "auto"):
            try:
                return _ChromaStore(self._data_dir / "chroma")
            except (ImportError, Exception):
                pass
        return _JSONStore(self._data_dir / "vectors.json")

    @property
    def backend_name(self) -> str:
        return "chromadb" if isinstance(self._store, _ChromaStore) else "json+tfidf"

    def add(self, text: str, memory_type: MemoryType = "fact",
            importance: float = 0.5, embedding: list[float] = None,
            metadata: dict = None) -> str:
        if not text or len(text.strip()) < 10:
            return ""
        meta = {
            "memory_type": memory_type,
            "importance": round(importance, 2),
            "timestamp": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }
        return self._store.add(text, embedding or [], meta)

    def search(self, query: str, embedding: list[float] = None, k: int = 5,
               memory_type: MemoryType | None = None) -> list[MemoryEntry]:
        raw = self._store.search(query, embedding or [], k, memory_type)
        entries = []
        for r in raw:
            meta = r.get("metadata", {})
            entries.append(MemoryEntry(
                id=r.get("id", ""),
                text=r.get("text", ""),
                memory_type=meta.get("memory_type", "fact"),
                importance=meta.get("importance", 0.5),
                score=r.get("score", 0.0),
                metadata=meta,
                created_at=meta.get("timestamp", ""),
                access_count=r.get("access_count", 0),
            ))
        return entries

    def count(self) -> int:
        return self._store.count()
