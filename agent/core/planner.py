"""Task planner - decomposes complex tasks into subtasks."""
from dataclasses import dataclass, field


@dataclass
class SubTask:
    id: int
    description: str
    depends_on: list[int] = field(default_factory=list)
    completed: bool = False
    result: str = ""


PLAN_PROMPT = """Break down the following complex task into a series of clear, ordered subtasks.

Task: {task}

Available tools: {tools}

Rules:
- Create 3-7 subtasks maximum
- Each subtask should be a single actionable step
- List dependencies between tasks
- Think about what information is needed before taking actions

Respond as JSON array:
[
  {{"id": 1, "description": "First step", "depends_on": []}},
  {{"id": 2, "description": "Second step", "depends_on": [1]}},
  ...
]

Only output the JSON array, nothing else."""


class TaskPlanner:
    def __init__(self, model_client=None):
        self._model = model_client

    def set_model(self, model_client):
        self._model = model_client

    def decompose(self, task: str, available_tools: list[str] = None) -> list[SubTask]:
        if not self._model:
            return [SubTask(id=1, description=task)]

        tools_str = ", ".join(available_tools or [])
        prompt = PLAN_PROMPT.format(task=task, tools=tools_str)
        try:
            response = self._model.chat([
                {"role": "system", "content": "You are a precise task planner. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ])
            import json, re
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                items = json.loads(json_match.group())
                return [
                    SubTask(id=item["id"], description=item["description"], depends_on=item.get("depends_on", []))
                    for item in items
                ]
        except Exception:
            pass
        return [SubTask(id=1, description=task)]

    def get_ready_tasks(self, tasks: list[SubTask]) -> list[SubTask]:
        completed_ids = {t.id for t in tasks if t.completed}
        return [t for t in tasks if not t.completed and all(dep in completed_ids for dep in t.depends_on)]
