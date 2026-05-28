"""Task planner — decomposes complex goals into ordered subtasks."""
import json
import re
from dataclasses import dataclass, field

from .logger import get_logger

log = get_logger(__name__)

COMPLEXITY_SIGNALS = [
    "build", "create a full", "develop", "start a", "make a complete",
    "and then", "after that", "step by step", "from scratch", "entire",
    "launch", "deploy", "implement", "design and", "research and",
]

DECOMPOSE_PROMPT = """Break this complex task into a clear, ordered list of subtasks.

Task: {task}
Available tools: {tools}

Rules:
- 3 to 7 subtasks maximum
- Each subtask is ONE specific, actionable step
- List only direct dependencies (which step must complete first)
- Keep descriptions under 20 words

Output ONLY a JSON array:
[
  {{"id": 1, "description": "...", "depends_on": []}},
  {{"id": 2, "description": "...", "depends_on": [1]}},
  {{"id": 3, "description": "...", "depends_on": [1, 2]}}
]"""


@dataclass
class SubTask:
    id: int
    description: str
    depends_on: list[int] = field(default_factory=list)
    completed: bool = False
    result: str = ""


class TaskPlanner:
    def __init__(self, model_client=None):
        self._model = model_client

    def set_model(self, m):
        self._model = m

    def is_complex(self, task: str) -> bool:
        t = task.lower()
        word_count = len(task.split())
        signal_match = sum(1 for s in COMPLEXITY_SIGNALS if s in t)
        return word_count > 25 or signal_match >= 2

    def decompose(self, task: str, tools: list[str] = None) -> list[SubTask]:
        if not self._model:
            return [SubTask(id=1, description=task)]

        prompt = DECOMPOSE_PROMPT.format(
            task=task,
            tools=", ".join(tools or []),
        )
        try:
            response = self._model.chat([
                {"role": "system", "content": "You are a precise task planner. Output only valid JSON array."},
                {"role": "user", "content": prompt},
            ])
        except Exception:
            log.exception("TaskPlanner: model call failed for task '%s'", task[:60])
            return [SubTask(id=1, description=task)]

        try:
            # Extract JSON array — try direct parse first, then extract from text
            raw = response.strip()
            if not raw.startswith("["):
                m = re.search(r"\[.*\]", raw, re.DOTALL)
                raw = m.group() if m else "[]"
            items = json.loads(raw)
            subtasks = []
            known_ids: set[int] = set()
            for item in items:
                if "id" not in item or "description" not in item:
                    continue
                tid = int(item["id"])
                deps = [d for d in item.get("depends_on", []) if d in known_ids]
                subtasks.append(SubTask(id=tid, description=item["description"], depends_on=deps))
                known_ids.add(tid)
            if subtasks:
                return subtasks
        except Exception:
            log.warning("TaskPlanner: could not parse subtask JSON for task '%s'", task[:60])

        return [SubTask(id=1, description=task)]

    def ready_tasks(self, tasks: list[SubTask]) -> list[SubTask]:
        done_ids = {t.id for t in tasks if t.completed}
        return [t for t in tasks if not t.completed and all(d in done_ids for d in t.depends_on)]
