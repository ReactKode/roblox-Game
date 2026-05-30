"""GatewayManager — starts all configured gateways in daemon threads."""
import threading
from typing import Callable

from .base import BaseGateway, Message
from ..core.logger import get_logger

_log = get_logger(__name__)


class GatewayManager:
    """Owns a collection of gateways, wires them to the agent, and manages
    their lifecycle (start / stop) using daemon threads so the main process
    can exit cleanly."""

    def __init__(self, agent_fn: Callable[[str], str]) -> None:
        """
        Parameters
        ----------
        agent_fn:
            A callable that accepts a plain-text user message and returns
            a plain-text reply string (typically ``agent.chat``).
        """
        self.agent_fn = agent_fn
        self._gateways: dict[str, BaseGateway] = {}
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, gateway: BaseGateway) -> None:
        """Add a gateway and wire it to the manager's internal handler.

        Must be called before ``start_all()``.
        """
        self._gateways[gateway.name] = gateway
        gateway.set_handler(self._handle)
        _log.debug("[manager] Registered gateway: %s", gateway.name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_all(self) -> None:
        """Launch every registered gateway in its own daemon thread."""
        if not self._gateways:
            _log.warning("[manager] start_all() called but no gateways are registered.")
            return

        for name, gateway in self._gateways.items():
            thread = threading.Thread(
                target=gateway.start,
                name=f"gateway-{name}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()
            _log.info("[manager] Started gateway: %s (thread=%s)", name, thread.name)

    def stop_all(self) -> None:
        """Signal every gateway to stop and wait up to 5 s for each thread."""
        _log.info("[manager] Stopping all gateways…")
        for gateway in self._gateways.values():
            try:
                gateway.stop()
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "[manager] Error while stopping gateway %s: %s", gateway.name, exc
                )

        for thread in self._threads:
            thread.join(timeout=5)
            if thread.is_alive():
                _log.warning(
                    "[manager] Thread %s did not stop within 5 s.", thread.name
                )

        _log.info("[manager] All gateways stopped.")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def active_gateways(self) -> list[str]:
        """Return the names of all registered gateways."""
        return list(self._gateways.keys())

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _handle(self, message: Message) -> str:
        """Route an inbound Message through ``agent_fn`` and return the reply.

        Any exception raised by the agent is caught here so individual
        platform threads cannot crash the whole process.
        """
        _log.debug(
            "[manager] Dispatching message id=%s from platform=%s user=%s",
            message.id,
            message.platform,
            message.user_name,
        )
        try:
            return self.agent_fn(message.text)
        except Exception as exc:  # noqa: BLE001
            _log.error(
                "[manager] agent_fn raised an exception for message id=%s: %s",
                message.id,
                exc,
                exc_info=True,
            )
            return f"An internal error occurred: {exc}"
