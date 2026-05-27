"""Cloud API model clients (OpenAI, Anthropic)."""
import json
import os
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic as _anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from .base import ModelClient


class OpenAIClient(ModelClient):
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if HAS_OPENAI else None

    def is_available(self) -> bool:
        return HAS_OPENAI and bool(os.getenv("OPENAI_API_KEY"))

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        if not self._client:
            return "Error: openai package not installed."
        try:
            kwargs = {"model": self.model, "messages": messages}
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
            return f"Error: OpenAI chat failed - {e}"

    def embed(self, text: str) -> list[float]:
        if not self._client:
            return []
        try:
            response = self._client.embeddings.create(model="text-embedding-3-small", input=text)
            return response.data[0].embedding
        except Exception:
            return []


class AnthropicClient(ModelClient):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self._client = _anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) if HAS_ANTHROPIC else None

    def is_available(self) -> bool:
        return HAS_ANTHROPIC and bool(os.getenv("ANTHROPIC_API_KEY"))

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        if not self._client:
            return "Error: anthropic package not installed."
        try:
            system_msg = ""
            filtered = []
            for m in messages:
                if m["role"] == "system":
                    system_msg = m["content"]
                else:
                    filtered.append(m)

            kwargs = {"model": self.model, "max_tokens": 4096, "messages": filtered}
            if system_msg:
                kwargs["system"] = system_msg
            if tools:
                # Convert OpenAI tool format to Anthropic format
                ant_tools = []
                for t in tools:
                    fn = t.get("function", t)
                    ant_tools.append({
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                    })
                kwargs["tools"] = ant_tools

            response = self._client.messages.create(**kwargs)
            parts = []
            for block in response.content:
                if block.type == "text":
                    parts.append(block.text)
                elif block.type == "tool_use":
                    parts.append(json.dumps({"tool_calls": [{"id": block.id, "function": {"name": block.name, "arguments": json.dumps(block.input)}}]}))
            return "\n".join(parts)
        except Exception as e:
            return f"Error: Anthropic chat failed - {e}"

    def embed(self, text: str) -> list[float]:
        return []  # Anthropic does not expose embeddings API
