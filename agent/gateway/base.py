"""Base gateway — defines the interface all platform connectors implement."""
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Callable

from ..core.logger import get_logger

_log = get_logger(__name__)


@dataclass
class Message:
    """Unified message envelope shared by every gateway."""

    id: str
    platform: str
    user_id: str
    user_name: str
    text: str
    chat_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    attachments: list = field(default_factory=list)


class BaseGateway(ABC):
    """Abstract base that every platform connector must subclass."""

    name: str = "base"

    def __init__(self) -> None:
        self._handler: Callable[[Message], str] | None = None

    # ------------------------------------------------------------------
    # Handler wiring
    # ------------------------------------------------------------------

    def set_handler(self, fn: Callable[[Message], str]) -> None:
        """Store the callable that processes an incoming Message and returns a reply."""
        self._handler = fn

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def start(self) -> None:
        """Blocking start — meant to be run inside a dedicated thread."""

    @abstractmethod
    def stop(self) -> None:
        """Signal the gateway to shut down cleanly."""

    @abstractmethod
    def send(self, chat_id: str, text: str) -> None:
        """Push an outbound message to the given chat/channel/user."""

    # ------------------------------------------------------------------
    # Dispatch helper
    # ------------------------------------------------------------------

    def handle(self, msg: Message) -> str:
        """Route an inbound Message through the registered handler.

        Returns the handler's reply string, or a safe fallback on error.
        """
        if self._handler is None:
            _log.warning("[%s] No handler set — dropping message id=%s", self.name, msg.id)
            return "No handler set"

        try:
            return self._handler(msg)
        except Exception as exc:  # noqa: BLE001
            _log.error(
                "[%s] Handler raised an exception for message id=%s: %s",
                self.name,
                msg.id,
                exc,
                exc_info=True,
            )
            return f"An error occurred while processing your request: {exc}"
