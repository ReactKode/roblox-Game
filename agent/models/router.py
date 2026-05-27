"""Model router - auto-selects available backend."""
import os
from .base import ModelClient
from .ollama import OllamaClient
from .lmstudio import LMStudioClient
from .cloud import OpenAIClient, AnthropicClient


class ModelRouter(ModelClient):
    """Tries backends in order: preferred → ollama → lmstudio → openai → anthropic."""

    BACKENDS = ["ollama", "lmstudio", "openai", "anthropic"]

    def __init__(self, config: dict):
        self._config = config
        self._active: ModelClient | None = None
        self._active_name: str = ""
        self._clients: dict[str, ModelClient] = self._build_clients()

    def _build_clients(self) -> dict[str, ModelClient]:
        cfg = self._config
        return {
            "ollama": OllamaClient(
                base_url=os.getenv("NEXUS_OLLAMA_URL", cfg.get("ollama", {}).get("base_url", "http://localhost:11434")),
                model=os.getenv("NEXUS_OLLAMA_MODEL", cfg.get("ollama", {}).get("model", "llama3.2")),
                embed_model=cfg.get("ollama", {}).get("embed_model", "nomic-embed-text"),
            ),
            "lmstudio": LMStudioClient(
                base_url=os.getenv("NEXUS_LMSTUDIO_URL", cfg.get("lmstudio", {}).get("base_url", "http://localhost:1234/v1")),
                model=cfg.get("lmstudio", {}).get("model", "local-model"),
            ),
            "openai": OpenAIClient(model=cfg.get("openai", {}).get("model", "gpt-4o-mini")),
            "anthropic": AnthropicClient(model=cfg.get("anthropic", {}).get("model", "claude-sonnet-4-6")),
        }

    def _resolve(self) -> tuple[str, ModelClient]:
        if self._active and self._active.is_available():
            return self._active_name, self._active

        preferred = os.getenv("NEXUS_MODEL_BACKEND", self._config.get("preferred_backend", "auto"))
        order = [preferred] + [b for b in self.BACKENDS if b != preferred] if preferred != "auto" else self.BACKENDS

        for name in order:
            client = self._clients.get(name)
            if client and client.is_available():
                self._active = client
                self._active_name = name
                return name, client

        raise RuntimeError(
            "No model backend available. Start Ollama (`ollama serve`) or LM Studio, "
            "or set OPENAI_API_KEY / ANTHROPIC_API_KEY environment variable."
        )

    def is_available(self) -> bool:
        try:
            self._resolve()
            return True
        except RuntimeError:
            return False

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        _, client = self._resolve()
        return client.chat(messages, tools)

    def embed(self, text: str) -> list[float]:
        _, client = self._resolve()
        try:
            return client.embed(text)
        except NotImplementedError:
            return []

    @property
    def active_backend(self) -> str:
        try:
            name, _ = self._resolve()
            return name
        except RuntimeError:
            return "none"
