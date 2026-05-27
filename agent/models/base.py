"""Abstract model client interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ModelClient(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        """Send messages and return assistant reply."""

    def embed(self, text: str) -> list[float]:
        """Return embedding vector. Override if model supports it."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support embeddings")

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend is reachable."""
