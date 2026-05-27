"""Ollama local model client."""
import json
import requests
from .base import ModelClient


class OllamaClient(ModelClient):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2", embed_model: str = "nomic-embed-text"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_model = embed_model

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            msg = data.get("message", {})
            # Handle tool_calls returned by Ollama
            if msg.get("tool_calls"):
                return json.dumps({"tool_calls": msg["tool_calls"]})
            return msg.get("content", "")
        except requests.exceptions.Timeout:
            return "Error: Ollama request timed out."
        except Exception as e:
            return f"Error: Ollama chat failed - {e}"

    def embed(self, text: str) -> list[float]:
        payload = {"model": self.embed_model, "input": text}
        try:
            r = requests.post(f"{self.base_url}/api/embed", json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            embeddings = data.get("embeddings", [[]])
            return embeddings[0] if embeddings else []
        except Exception:
            return []

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []
