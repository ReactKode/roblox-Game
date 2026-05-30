"""CLI gateway — wraps the interactive terminal loop as a gateway."""
import threading
import uuid

from .base import BaseGateway, Message
from ..core.logger import get_logger

_log = get_logger(__name__)

# Optional rich console — graceful fallback to plain print
try:
    from rich.console import Console as _RichConsole

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


class CLIGateway(BaseGateway):
    """Interactive terminal gateway — reads from stdin, writes to stdout."""

    name = "cli"

    def __init__(self) -> None:
        super().__init__()
        self._stop_flag = threading.Event()
        self._console = _RichConsole() if _RICH_AVAILABLE else None

    # ------------------------------------------------------------------
    # BaseGateway interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Run the REPL loop until '/quit' or stop() is called."""
        _log.info("[cli] Starting CLI gateway. Type /quit to exit.")
        self._print("Hermes is ready. Type your message (or /quit to exit).")

        while not self._stop_flag.is_set():
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                _log.info("[cli] Input stream closed or KeyboardInterrupt — stopping.")
                break

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                _log.info("[cli] /quit received — stopping CLI gateway.")
                break

            msg = Message(
                id=str(uuid.uuid4()),
                platform="cli",
                user_id="local_user",
                user_name="User",
                text=user_input,
                chat_id="cli",
            )

            response = self.handle(msg)
            self.send("cli", response)

        self._stop_flag.set()
        _log.info("[cli] CLI gateway stopped.")

    def stop(self) -> None:
        """Signal the REPL loop to exit."""
        self._stop_flag.set()

    def send(self, chat_id: str, text: str) -> None:  # noqa: ARG002
        """Print a response to stdout, using rich markup if available."""
        self._print(f"Hermes: {text}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _print(self, text: str) -> None:
        """Print using rich Console when available, plain print otherwise."""
        if self._console is not None:
            self._console.print(text)
        else:
            print(text)
