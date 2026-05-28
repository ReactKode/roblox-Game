"""Centralized logger factory.

Usage in any module:
    from .logger import get_logger
    log = get_logger(__name__)
    log.info("message")
    log.warning("something odd")
    log.error("failure", exc_info=True)   # includes traceback

Log level is controlled via env var: NEXUS_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR
Default: INFO
"""
import logging
import os
import sys
from pathlib import Path


def _setup_root_logger() -> None:
    """Configure root NexusAgent logger once at import time."""
    root = logging.getLogger("nexus")
    if root.handlers:
        return  # already configured

    level_name = os.getenv("NEXUS_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler (stderr, so it doesn't pollute stdout responses)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Optional file handler
    log_dir = Path(os.getenv("NEXUS_DATA_DIR", "./data"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "nexus.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)  # file always gets full detail
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception:
        pass  # if data dir isn't ready yet, file logging is optional


_setup_root_logger()


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'nexus' namespace."""
    if not name.startswith("nexus"):
        # Strip leading 'agent.' so logs read as nexus.core.loop etc.
        short = name.replace("agent.", "")
        name = f"nexus.{short}"
    return logging.getLogger(name)
