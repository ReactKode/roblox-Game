"""LM Studio model client (OpenAI-compatible API)."""
import json
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

import requests
from .base import ModelClient


class LMStudioClient(ModelClient):
    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str = "local-model"):
        self.base_url = base_url
        self.model = model
        self._client = None
        if HAS_OPENAI:
            self._client = OpenAI(base_url=base_url, api_key="lmstudio")

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/models", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def _get_model(self) -> str:
        """Use first available model if 'local-model' placeholder."""
        if self.model != "local-model":
            return self.model
        try:
            r = requests.get(f"{self.base_url}/models", timeout=5)
            models = r.json().get("data", [])
            return models[0]["id"] if models else self.model
        except Exception:
            return self.model

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        if not HAS_OPENAI:
            return self._chat_raw(messages, tools)
        try:
            model = self._get_model()
            kwargs = {"model": model, "messages": messages}
            if tools:
                kwargs["tools"] = tools
            response = self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            msg = choice.message
            if msg.tool_calls:
                tool_calls = [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
                return json.dumps({"tool_calls": tool_calls})
            return msg.content or ""
        except Exception as e:
            return f"Error: LM Studio chat failed - {e}"

    def _chat_raw(self, messages: list[dict], tools: list[dict] | None) -> str:
        payload = {"model": self._get_model(), "messages": messages}
        if tools:
            payload["tools"] = tools
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: LM Studio raw chat failed - {e}"

    def embed(self, text: str) -> list[float]:
        if not HAS_OPENAI:
            return []
        try:
            response = self._client.embeddings.create(model=self._get_model(), input=text)
            return response.data[0].embedding
        except Exception:
            return []
