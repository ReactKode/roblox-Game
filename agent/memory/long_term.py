"""Long-term vector memory. Uses ChromaDB when available, falls back to JSON."""
import json
import math
import os
import uuid
from datetime import datetime
from pathlib import Path


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class JSONVectorStore:
    """Fallback vector store using JSON file."""

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
        record_id = str(uuid.uuid4())
        self._records.append({
            "id": record_id,
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
            "created_at": datetime.utcnow().isoformat(),
        })
        self._save()
        return record_id

    def search(self, query_embedding: list[float], k: int = 5) -> list[dict]:
        if not query_embedding:
            return [{"text": r["text"], "metadata": r["metadata"], "score": 0.0}
                    for r in self._records[-k:]]
        scored = []
        for r in self._records:
            score = _cosine_similarity(query_embedding, r["embedding"])
            scored.append({"text": r["text"], "metadata": r["metadata"], "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def count(self) -> int:
        return len(self._records)


class ChromaVectorStore:
    def __init__(self, persist_dir: Path):
        import chromadb
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection("long_term_memory")

    def add(self, text: str, embedding: list[float], metadata: dict) -> str:
        record_id = str(uuid.uuid4())
        kwargs = {"documents": [text], "ids": [record_id], "metadatas": [metadata]}
        if embedding:
            kwargs["embeddings"] = [embedding]
        self._collection.add(**kwargs)
        return record_id

    def search(self, query_embedding: list[float], k: int = 5, query_text: str = "") -> list[dict]:
        try:
            if query_embedding:
                results = self._collection.query(query_embeddings=[query_embedding], n_results=min(k, self._collection.count()))
            else:
                results = self._collection.query(query_texts=[query_text], n_results=min(k, self._collection.count()))
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            return [{"text": d, "metadata": m, "score": 1 - dist}
                    for d, m, dist in zip(docs, metas, distances)]
        except Exception:
            return []

    def count(self) -> int:
        return self._collection.count()


class LongTermMemory:
    def __init__(self, data_dir: Path, backend: str = "chromadb"):
        self._data_dir = data_dir
        self._store = self._init_store(backend)

    def _init_store(self, backend: str):
        if backend == "chromadb":
            try:
                return ChromaVectorStore(self._data_dir / "chroma")
            except ImportError:
                pass
        return JSONVectorStore(self._data_dir / "vectors.json")

    def add(self, text: str, embedding: list[float] = None, metadata: dict = None) -> str:
        meta = {"timestamp": datetime.utcnow().isoformat(), **(metadata or {})}
        return self._store.add(text, embedding or [], meta)

    def search(self, query: str, embedding: list[float] = None, k: int = 5) -> list[dict]:
        if hasattr(self._store, "search"):
            if isinstance(self._store, ChromaVectorStore):
                return self._store.search(embedding or [], k, query_text=query)
            return self._store.search(embedding or [], k)
        return []

    def count(self) -> int:
        return self._store.count()
