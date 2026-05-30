"""Slack gateway — responds to DMs and @mentions via Slack Bolt."""
import uuid
from typing import Optional

from .base import BaseGateway, Message
from ..core.logger import get_logger

_log = get_logger(__name__)

_SLACK_AVAILABLE = False
try:
    from slack_bolt import App as _BoltApp  # type: ignore[import]
    from slack_bolt.adapter.socket_mode import SocketModeHandler as _SocketModeHandler  # type: ignore[import]

    _SLACK_AVAILABLE = True
except ImportError:
    pass


class SlackGateway(BaseGateway):
    """Slack bot gateway using Slack Bolt with Socket Mode."""

    name = "slack"

    def __init__(self, bot_token: str, app_token: str) -> None:
        super().__init__()
        self._bot_token = bot_token
        self._app_token = app_token
        self._app = None             # slack_bolt App
        self._socket_handler = None  # SocketModeHandler

    # ------------------------------------------------------------------
    # BaseGateway interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Register Bolt event handlers and start Socket Mode (blocking)."""
        if not _SLACK_AVAILABLE:
            _log.error(
                "[slack] slack-bolt is not installed. "
                "Install it with: pip install slack-bolt"
            )
            return

        _log.info("[slack] Starting Slack gateway (Socket Mode)…")
        self._app = _BoltApp(token=self._bot_token)

        gateway_self = self

        # ------ DM / regular messages ------
        @self._app.message()
        def on_message(message: dict, say) -> None:  # noqa: ANN001
            _handle_event(gateway_self, message, say)

        # ------ @mention ------
        @self._app.event("app_mention")
        def on_mention(event: dict, say) -> None:  # noqa: ANN001
            _handle_event(gateway_self, event, say)

        self._socket_handler = _SocketModeHandler(self._app, self._app_token)
        self._socket_handler.start()  # blocking
        _log.info("[slack] Slack gateway stopped.")

    def stop(self) -> None:
        """Close the Socket Mode handler."""
        if self._socket_handler is not None:
            _log.info("[slack] Requesting Slack gateway shutdown…")
            try:
                self._socket_handler.close()
            except Exception as exc:  # noqa: BLE001
                _log.warning("[slack] Error during stop: %s", exc)

    def send(self, chat_id: str, text: str) -> None:
        """Post a message to a Slack channel or DM by channel ID."""
        if not _SLACK_AVAILABLE or self._app is None:
            _log.warning("[slack] Cannot send — gateway is not running.")
            return

        try:
            self._app.client.chat_postMessage(channel=chat_id, text=text)
        except Exception as exc:  # noqa: BLE001
            _log.error("[slack] Failed to post to %s: %s", chat_id, exc)


# ------------------------------------------------------------------
# Module-level helper (avoids closure issues with Bolt decorators)
# ------------------------------------------------------------------

def _handle_event(gateway: SlackGateway, event: dict, say) -> None:  # noqa: ANN001
    """Build a Message from a Slack event dict and dispatch it."""
    user_id = event.get("user", "unknown")
    channel = event.get("channel", "unknown")
    text = event.get("text", "")

    msg = Message(
        id=str(uuid.uuid4()),
        platform="slack",
        user_id=user_id,
        user_name=user_id,  # Slack doesn't always include display name in payload
        text=text,
        chat_id=channel,
    )

    _log.debug("[slack] Message from %s in %s: %s", msg.user_id, msg.chat_id, msg.text)
    response = gateway.handle(msg)

    try:
        say(response)
    except Exception as exc:  # noqa: BLE001
        _log.error("[slack] Failed to send reply: %s", exc)
