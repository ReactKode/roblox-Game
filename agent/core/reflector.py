"""Reflexion system — post-task self-analysis that makes the agent improve over time."""
import json
import re
from dataclasses import dataclass, field
from datetime import datetime


REFLECT_PROMPT = """You just completed a task. Critically analyze your own performance.

Task: {task}
Tools used: {tools_used}
Number of steps: {iterations}
Success: {success}
Final answer preview: {answer_preview}

Output ONLY this JSON — no other text:
{{
  "what_worked": "what approach or tool was most effective",
  "what_failed": "what went wrong or was inefficient (write 'nothing' if fully successful)",
  "lesson": "one concrete actionable lesson for future similar tasks",
  "next_time": "specific change to make next time",
  "confidence": 0.85,
  "tags": ["keyword1", "keyword2"],
  "skill_domain": "primary domain: coding/research/blender/unity/godot/business/general"
}}"""


@dataclass
class Reflection:
    task: str
    what_worked: str
    what_failed: str
    lesson: str
    next_time: str
    confidence: float
    tags: list[str]
    skill_domain: str
    tools_used: list[str] = field(default_factory=list)
    success: bool = True
    iterations: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class Reflector:
    def __init__(self, model_client=None, episodic_memory=None):
        self._model = model_client
        self._episodic = episodic_memory

    def set_model(self, model):
        self._model = model

    def reflect(self, task: str, tools_used: list[str], final_answer: str,
                success: bool, iterations: int) -> "Reflection | None":
        if not self._model:
            return None
        prompt = REFLECT_PROMPT.format(
            task=task[:400],
            tools_used=", ".join(tools_used) or "none",
            iterations=iterations,
            success=success,
            answer_preview=final_answer[:300],
        )
        try:
            response = self._model.chat([
                {"role": "system", "content": "You are critically analyzing AI agent performance. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ])
            data = _parse_json(response)
            if not data:
                return None
            r = Reflection(
                task=task,
                what_worked=data.get("what_worked", ""),
                what_failed=data.get("what_failed", "nothing"),
                lesson=data.get("lesson", ""),
                next_time=data.get("next_time", ""),
                confidence=float(data.get("confidence", 0.7)),
                tags=data.get("tags", []),
                skill_domain=data.get("skill_domain", "general"),
                tools_used=tools_used,
                success=success,
                iterations=iterations,
            )
            if self._episodic:
                self._episodic.log(
                    task=task,
                    action="reflection",
                    observation=f"lesson={r.lesson} | next_time={r.next_time}",
                    outcome=f"confidence={r.confidence:.2f} | domain={r.skill_domain}",
                    skill_used=r.skill_domain,
                    success=success,
                    is_reflection=True,
                    tags=r.tags,
                )
            return r
        except Exception:
            return None

    def get_relevant_reflections(self, task: str, limit: int = 3) -> list[str]:
        if not self._episodic:
            return []
        episodes = self._episodic.recall_reflections(task, limit)
        return [e["observation"] for e in episodes if e.get("observation")]


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None
