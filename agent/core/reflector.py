"""Reflexion system — post-task self-analysis that makes the agent improve over time."""
import json
import re
from dataclasses import dataclass, field
from datetime import datetime

from .logger import get_logger

log = get_logger(__name__)

REFLECT_PROMPT = """Analyze your own performance on this task. Be honest and specific.

Task: {task}
Tools used: {tools_used}
Steps taken: {iterations}
Success: {success}
Answer preview: {answer_preview}

What actually worked? What was inefficient or wrong? What would you do differently?
Don't be generic. If you used too many steps, say why. If a tool failed, name it.

Output ONLY this JSON:
{{
  "what_worked": "the specific approach or tool that was most effective",
  "what_failed": "what went wrong or was inefficient — 'nothing' only if truly flawless",
  "lesson": "one concrete, actionable lesson for next time — not a platitude",
  "next_time": "the specific change you would make if given this task again",
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
                {"role": "system",
                 "content": "You are critically analyzing AI agent performance. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ])
        except Exception:
            log.exception("Reflector: model call failed for task '%s'", task[:60])
            return None

        data = _parse_json(response)
        if not data:
            log.warning("Reflector: could not parse JSON from model response for task '%s'", task[:60])
            return None

        confidence = float(data.get("confidence", 0.7))
        # Clamp to valid range
        confidence = max(0.0, min(1.0, confidence))

        r = Reflection(
            task=task,
            what_worked=data.get("what_worked", ""),
            what_failed=data.get("what_failed", "nothing"),
            lesson=data.get("lesson", ""),
            next_time=data.get("next_time", ""),
            confidence=confidence,
            tags=data.get("tags", []),
            skill_domain=data.get("skill_domain", "general"),
            tools_used=tools_used,
            success=success,
            iterations=iterations,
        )

        if self._episodic:
            try:
                self._episodic.log(
                    task=task,
                    action="reflection",
                    # Store full lesson + next_time so get_relevant_reflections can return them
                    observation=f"lesson={r.lesson} | next_time={r.next_time} | worked={r.what_worked}",
                    outcome=f"confidence={r.confidence:.2f} | domain={r.skill_domain} | failed={r.what_failed}",
                    skill_used=r.skill_domain,
                    success=success,
                    is_reflection=True,
                    tags=r.tags,
                )
            except Exception:
                log.exception("Reflector: failed to log reflection to episodic memory")

        log.debug("Reflection stored (domain=%s, confidence=%.2f)", r.skill_domain, r.confidence)
        return r

    def get_relevant_reflections(self, task: str, limit: int = 3) -> list[str]:
        if not self._episodic:
            return []
        try:
            episodes = self._episodic.recall_reflections(task, limit)
        except Exception:
            log.exception("Reflector: failed to recall reflections for task '%s'", task[:60])
            return []
        summaries = []
        for e in episodes:
            obs = e.get("observation", "")
            outcome = e.get("outcome", "")
            if obs:
                summaries.append(f"{obs} | {outcome}".strip(" |"))
        return summaries


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try markdown code fences first (more specific than bare regex)
    for pattern in [r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    # Last resort: find first {...} containing known keys
    for m in re.finditer(r"\{[^{}]{10,}\}", text, re.DOTALL):
        try:
            obj = json.loads(m.group())
            if any(k in obj for k in ("confidence", "lesson", "what_worked")):
                return obj
        except json.JSONDecodeError:
            continue
    return None
