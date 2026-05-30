"""Telegram gateway — bot that routes messages to the agent."""
import asyncio
import uuid
from typing import Optional

from .base import BaseGateway, Message
from ..core.logger import get_logger

_log = get_logger(__name__)

_TELEGRAM_AVAILABLE = False
try:
    from telegram import Update
    from telegram.ext import (
        Application,
        ContextTypes,
        MessageHandler,
        filters,
    )

    _TELEGRAM_AVAILABLE = True
except ImportError:
    pass

# Telegram hard limit for a single message
_MAX_MSG_LEN = 4096


class TelegramGateway(BaseGateway):
    """Telegram bot gateway using python-telegram-bot v20+."""

    name = "telegram"

    def __init__(self, token: str, allowed_users: Optional[list[str]] = None) -> None:
        super().__init__()
        self._token = token
        self._allowed_users: list[str] = allowed_users or []
        self._application = None  # set in start()

    # ------------------------------------------------------------------
    # BaseGateway interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Build the Application and start long-polling (blocking)."""
        if not _TELEGRAM_AVAILABLE:
            _log.error(
                "[telegram] python-telegram-bot is not installed. "
                "Install it with: pip install python-telegram-bot"
            )
            return

        _log.info("[telegram] Starting Telegram gateway (polling)…")
        self._application = (
            Application.builder().token(self._token).build()
        )

        self._application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )

        # run_polling is blocking; it manages its own event loop
        self._application.run_polling(stop_signals=None)
        _log.info("[telegram] Telegram gateway stopped.")

    def stop(self) -> None:
        """Request the running Application to stop polling."""
        if self._application is not None:
            _log.info("[telegram] Requesting Telegram gateway shutdown…")
            # Schedule the stop inside the bot's own loop
            try:
                loop = self._application.update_queue._loop  # type: ignore[attr-defined]
                if loop and loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self._application.updater.stop(), loop
                    )
                    asyncio.run_coroutine_threadsafe(
                        self._application.stop(), loop
                    )
            except Exception as exc:  # noqa: BLE001
                _log.warning("[telegram] Error during stop: %s", exc)

    def send(self, chat_id: str, text: str) -> None:
        """Send a message to the given chat_id (sync wrapper)."""
        if not _TELEGRAM_AVAILABLE or self._application is None:
            _log.warning("[telegram] Cannot send — gateway is not running.")
            return

        try:
            loop = self._application.update_queue._loop  # type: ignore[attr-defined]
            if loop and loop.is_running():
                for chunk in _split_text(text):
                    asyncio.run_coroutine_threadsafe(
                        self._application.bot.send_message(
                            chat_id=int(chat_id), text=chunk
                        ),
                        loop,
                    ).result(timeout=15)
        except Exception as exc:  # noqa: BLE001
            _log.error("[telegram] Failed to send message to %s: %s", chat_id, exc)

    # ------------------------------------------------------------------
    # Internal async handler
    # ------------------------------------------------------------------

    async def _on_message(
        self,
        update: "Update",  # noqa: F821
        context: "ContextTypes.DEFAULT_TYPE",  # noqa: F821
    ) -> None:
        """Async handler called by python-telegram-bot for each text message."""
        tg_user = update.effective_user
        if tg_user is None or update.message is None:
            return

        user_id = str(tg_user.id)

        # Access control
        if self._allowed_users and user_id not in self._allowed_users:
            _log.warning(
                "[telegram] Rejected message from unauthorised user %s", user_id
            )
            await update.message.reply_text(
                "Sorry, you are not authorised to use this bot."
            )
            return

        msg = Message(
            id=str(uuid.uuid4()),
            platform="telegram",
            user_id=user_id,
            user_name=tg_user.full_name or tg_user.username or user_id,
            text=update.message.text or "",
            chat_id=str(update.effective_chat.id),
        )

        _log.debug("[telegram] Received message from %s: %s", msg.user_name, msg.text)
        response = self.handle(msg)

        # Split long replies to respect Telegram's 4096-char limit
        for chunk in _split_text(response):
            await update.message.reply_text(chunk)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _split_text(text: str, max_len: int = _MAX_MSG_LEN) -> list[str]:
    """Split text into chunks that fit within Telegram's message size limit."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:max_len])
        text = text[max_len:]
    return chunks
