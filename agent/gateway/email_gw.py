"""Email gateway — polls IMAP for new messages, responds via SMTP."""
import email as _email_stdlib
import imaplib
import smtplib
import threading
import time
import uuid
from email.mime.text import MIMEText
from typing import Optional

from .base import BaseGateway, Message
from ..core.logger import get_logger

_log = get_logger(__name__)


class EmailGateway(BaseGateway):
    """IMAP-polling / SMTP-replying email gateway."""

    name = "email"

    def __init__(
        self,
        imap_host: str,
        imap_port: int,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        poll_interval: int = 60,
        allowed_senders: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self._imap_host = imap_host
        self._imap_port = imap_port
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._username = username
        self._password = password
        self._poll_interval = poll_interval
        self._allowed_senders: list[str] = allowed_senders or []
        self._stop_flag = threading.Event()

    # ------------------------------------------------------------------
    # BaseGateway interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Poll the inbox every poll_interval seconds until stop() is called."""
        _log.info(
            "[email] Starting email gateway — polling every %ds.", self._poll_interval
        )
        while not self._stop_flag.is_set():
            try:
                self._check_inbox()
            except Exception as exc:  # noqa: BLE001
                _log.error("[email] Error while checking inbox: %s", exc, exc_info=True)

            # Sleep in small increments so stop() is responsive
            for _ in range(self._poll_interval):
                if self._stop_flag.is_set():
                    break
                time.sleep(1)

        _log.info("[email] Email gateway stopped.")

    def stop(self) -> None:
        """Signal the polling loop to exit."""
        self._stop_flag.set()

    def send(self, chat_id: str, text: str) -> None:
        """Send an email reply; chat_id is treated as the recipient address."""
        self._send_reply(
            to_addr=chat_id,
            subject="Re: Hermes Response",
            body=text,
        )

    # ------------------------------------------------------------------
    # IMAP inbox check
    # ------------------------------------------------------------------

    def _check_inbox(self) -> None:
        """Connect to IMAP, fetch UNSEEN messages, process each, mark as read."""
        _log.debug("[email] Connecting to IMAP %s:%d…", self._imap_host, self._imap_port)
        with imaplib.IMAP4_SSL(self._imap_host, self._imap_port) as imap:
            imap.login(self._username, self._password)
            imap.select("INBOX")

            status, data = imap.search(None, "UNSEEN")
            if status != "OK":
                _log.warning("[email] IMAP SEARCH returned status: %s", status)
                return

            msg_ids: list[bytes] = data[0].split() if data[0] else []
            if not msg_ids:
                _log.debug("[email] No new messages.")
                return

            _log.info("[email] Found %d unread message(s).", len(msg_ids))
            for msg_id in msg_ids:
                self._process_email(imap, msg_id)

    def _process_email(self, imap: imaplib.IMAP4_SSL, msg_id: bytes) -> None:
        """Fetch, parse, handle, and reply to a single email."""
        status, data = imap.fetch(msg_id, "(RFC822)")
        if status != "OK" or not data or data[0] is None:
            _log.warning("[email] Could not fetch message %s", msg_id)
            return

        raw_bytes = data[0][1]  # type: ignore[index]
        parsed = _email_stdlib.message_from_bytes(raw_bytes)

        sender = _decode_header_value(parsed.get("From", ""))
        subject = _decode_header_value(parsed.get("Subject", "(no subject)"))
        body = _extract_body(parsed)

        _log.debug("[email] Email from=%s subject=%s", sender, subject)

        # Sender allow-list check
        sender_addr = _extract_address(sender)
        if self._allowed_senders and sender_addr not in self._allowed_senders:
            _log.warning("[email] Rejected email from unauthorised sender %s", sender_addr)
            # Still mark as read so we don't keep seeing it
            imap.store(msg_id, "+FLAGS", "\\Seen")
            return

        msg = Message(
            id=str(uuid.uuid4()),
            platform="email",
            user_id=sender_addr,
            user_name=sender,
            text=f"Subject: {subject}\n\n{body}",
            chat_id=sender_addr,
        )

        response = self.handle(msg)
        self._send_reply(
            to_addr=sender_addr,
            subject=f"Re: {subject}",
            body=response,
        )

        # Mark as read
        imap.store(msg_id, "+FLAGS", "\\Seen")

    # ------------------------------------------------------------------
    # SMTP reply
    # ------------------------------------------------------------------

    def _send_reply(self, to_addr: str, subject: str, body: str) -> None:
        """Send a reply email via SMTP SSL."""
        _log.debug("[email] Sending reply to %s (subject=%s)", to_addr, subject)
        mime = MIMEText(body, "plain", "utf-8")
        mime["From"] = self._username
        mime["To"] = to_addr
        mime["Subject"] = subject

        try:
            with smtplib.SMTP_SSL(self._smtp_host, self._smtp_port) as smtp:
                smtp.login(self._username, self._password)
                smtp.sendmail(self._username, [to_addr], mime.as_string())
            _log.info("[email] Reply sent to %s.", to_addr)
        except Exception as exc:  # noqa: BLE001
            _log.error("[email] Failed to send reply to %s: %s", to_addr, exc, exc_info=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _decode_header_value(value: str) -> str:
    """Decode an RFC-2047-encoded email header into a plain string."""
    parts = _email_stdlib.header.decode_header(value)
    decoded_parts = []
    for fragment, charset in parts:
        if isinstance(fragment, bytes):
            decoded_parts.append(fragment.decode(charset or "utf-8", errors="replace"))
        else:
            decoded_parts.append(fragment)
    return "".join(decoded_parts)


def _extract_address(from_header: str) -> str:
    """Extract the bare email address from a 'Display Name <addr>' header."""
    if "<" in from_header and ">" in from_header:
        return from_header.split("<")[1].split(">")[0].strip()
    return from_header.strip()


def _extract_body(parsed_msg: "_email_stdlib.message.Message") -> str:  # type: ignore[name-defined]
    """Walk a MIME message and return the first text/plain part."""
    if parsed_msg.is_multipart():
        for part in parsed_msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    return payload.decode(
                        part.get_content_charset() or "utf-8", errors="replace"
                    )
    else:
        payload = parsed_msg.get_payload(decode=True)
        if isinstance(payload, bytes):
            return payload.decode(
                parsed_msg.get_content_charset() or "utf-8", errors="replace"
            )
    return ""
