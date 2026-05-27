"""Short-term context window memory."""
from collections import deque
from datetime import datetime


class ShortTermMemory:
    def __init__(self, limit: int = 20):
        self.limit = limit
        self._messages: deque[dict] = deque()

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content, "_ts": datetime.utcnow().isoformat()})
        while len(self._messages) > self.limit:
            self._messages.popleft()

    def get_messages(self, include_system: bool = True) -> list[dict]:
        return [{"role": m["role"], "content": m["content"]} for m in self._messages
                if include_system or m["role"] != "system"]

    def get_recent(self, n: int = 5) -> list[dict]:
        msgs = list(self._messages)
        return [{"role": m["role"], "content": m["content"]} for m in msgs[-n:]]

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    def to_text(self) -> str:
        lines = []
        for m in self._messages:
            lines.append(f"[{m['role']}]: {m['content'][:200]}")
        return "\n".join(lines)
