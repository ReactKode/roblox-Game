"""Isolated subagent — runs tasks with a fresh conversation context."""
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

from .loop import AgentLoop
from .logger import get_logger

log = get_logger(__name__)


class SubAgent:
    """Runs isolated AgentLoop instances with their own conversation histories.

    Each call to run() or run_async() gets a brand-new AgentLoop so its
    conversation context is completely independent of every other call.
    """

    def __init__(
        self,
        model_client,
        tool_registry,
        memory_manager,
        skill_registry,
        reflector=None,
        self_model=None,
        max_iterations: int = 10,
        max_workers: int = 4,
    ):
        self._model = model_client
        self._tools = tool_registry
        self._memory = memory_manager
        self._skills = skill_registry
        self._reflector = reflector
        self._self_model = self_model
        self._max_iterations = max_iterations
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="subagent"
        )

    # ── Public API ────────────────────────────────────────────────────────

    def run(self, task: str, verbose: bool = False) -> str:
        """Run a task synchronously in a fresh, isolated context."""
        loop = self._new_loop()
        log.info("[subagent] Starting task: %s", task[:100])
        try:
            result = loop.run(task, verbose=verbose)
            log.info("[subagent] Task complete (%d chars): %s…", len(result), task[:60])
            return result
        except Exception as exc:
            log.exception("[subagent] Task failed: %s", task[:60])
            return f"Subagent error: {exc}"

    def run_async(
        self,
        task: str,
        callback: Callable[[str], None] | None = None,
        verbose: bool = False,
    ) -> Future:
        """Run a task in a background thread; call *callback* on completion."""

        def _worker() -> str:
            result = self.run(task, verbose=verbose)
            if callback:
                try:
                    callback(result)
                except Exception:
                    log.exception("[subagent] Callback raised for task: %s", task[:60])
            return result

        return self._executor.submit(_worker)

    def run_parallel(self, tasks: list[str], verbose: bool = False) -> list[str]:
        """Run multiple tasks concurrently; return results in the same order."""
        futures = [self.run_async(t, verbose=verbose) for t in tasks]
        results: list[str] = []
        for f in futures:
            try:
                results.append(f.result(timeout=300))
            except Exception as exc:
                log.exception("[subagent] Parallel task failed")
                results.append(f"Error: {exc}")
        return results

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
        log.info("[subagent] Thread pool shut down.")

    # ── Private ───────────────────────────────────────────────────────────

    def _new_loop(self) -> AgentLoop:
        """Return a fresh AgentLoop with an empty conversation history."""
        return AgentLoop(
            model=self._model,
            tool_registry=self._tools,
            memory_manager=self._memory,
            skill_registry=self._skills,
            reflector=self._reflector,
            self_model=self._self_model,
            max_iterations=self._max_iterations,
        )
