"""Discord gateway — bot that routes DMs and @mentions to the agent."""
import asyncio
import uuid
from typing import Optional

from .base import BaseGateway, Message
from ..core.logger import get_logger

_log = get_logger(__name__)

_DISCORD_AVAILABLE = False
try:
    import discord  # type: ignore[import]

    _DISCORD_AVAILABLE = True
except ImportError:
    pass


class DiscordGateway(BaseGateway):
    """Discord bot gateway responding to DMs and @mentions."""

    name = "discord"

    def __init__(
        self,
        token: str,
        allowed_guilds: Optional[list[int]] = None,
        command_prefix: str = "!",
    ) -> None:
        super().__init__()
        self._token = token
        self._allowed_guilds: list[int] = allowed_guilds or []
        self._command_prefix = command_prefix
        self._client = None  # set in start()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    # BaseGateway interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Create the Discord client and block until it disconnects."""
        if not _DISCORD_AVAILABLE:
            _log.error(
                "[discord] discord.py is not installed. "
                "Install it with: pip install discord.py"
            )
            return

        intents = discord.Intents.default()
        intents.message_content = True
        self._client = discord.Client(intents=intents)

        # Capture the loop so send() can schedule coroutines onto it
        gateway_self = self

        @self._client.event
        async def on_ready() -> None:
            gateway_self._loop = asyncio.get_event_loop()
            _log.info(
                "[discord] Logged in as %s (id=%s)",
                gateway_self._client.user,
                gateway_self._client.user.id,
            )

        @self._client.event
        async def on_message(message: "discord.Message") -> None:
            # Ignore messages from the bot itself
            if message.author == gateway_self._client.user:
                return

            # Guild filter
            if (
                gateway_self._allowed_guilds
                and message.guild
                and message.guild.id not in gateway_self._allowed_guilds
            ):
                return

            is_dm = isinstance(message.channel, discord.DMChannel)
            is_mention = (
                gateway_self._client.user is not None
                and gateway_self._client.user in message.mentions
            )

            if not (is_dm or is_mention):
                return

            # Strip the @mention prefix if present
            text = message.content
            if is_mention and gateway_self._client.user is not None:
                text = text.replace(
                    f"<@{gateway_self._client.user.id}>", ""
                ).replace(
                    f"<@!{gateway_self._client.user.id}>", ""
                ).strip()

            msg = Message(
                id=str(uuid.uuid4()),
                platform="discord",
                user_id=str(message.author.id),
                user_name=str(message.author),
                text=text,
                chat_id=str(message.channel.id),
            )

            _log.debug(
                "[discord] Message from %s in channel %s: %s",
                msg.user_name,
                msg.chat_id,
                msg.text,
            )

            response = gateway_self.handle(msg)

            # Discord has a 2000-char message limit
            for chunk in _split_text(response, max_len=2000):
                await message.channel.send(chunk)

        _log.info("[discord] Starting Discord gateway…")
        self._client.run(self._token)
        _log.info("[discord] Discord gateway stopped.")

    def stop(self) -> None:
        """Close the Discord client connection."""
        if self._client is not None and self._loop is not None:
            _log.info("[discord] Requesting Discord gateway shutdown…")
            try:
                asyncio.run_coroutine_threadsafe(
                    self._client.close(), self._loop
                ).result(timeout=10)
            except Exception as exc:  # noqa: BLE001
                _log.warning("[discord] Error during stop: %s", exc)

    def send(self, chat_id: str, text: str) -> None:
        """Send a message to a channel by ID (sync wrapper)."""
        if not _DISCORD_AVAILABLE or self._client is None or self._loop is None:
            _log.warning("[discord] Cannot send — gateway is not running.")
            return

        async def _send() -> None:
            channel = self._client.get_channel(int(chat_id))
            if channel is None:
                try:
                    channel = await self._client.fetch_channel(int(chat_id))
                except Exception as exc:  # noqa: BLE001
                    _log.error(
                        "[discord] Could not fetch channel %s: %s", chat_id, exc
                    )
                    return
            for chunk in _split_text(text, max_len=2000):
                await channel.send(chunk)

        try:
            asyncio.run_coroutine_threadsafe(_send(), self._loop).result(timeout=15)
        except Exception as exc:  # noqa: BLE001
            _log.error("[discord] Failed to send to channel %s: %s", chat_id, exc)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _split_text(text: str, max_len: int = 2000) -> list[str]:
    """Split text into chunks that fit within Discord's message size limit."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:max_len])
        text = text[max_len:]
    return chunks
