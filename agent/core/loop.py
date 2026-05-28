"""ReAct agent loop with structured-output retry, reflection, and self-model integration."""
import json
import re
from datetime import datetime

from .logger import get_logger

log = get_logger(__name__)


# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are NexusAgent — a self-improving AI agent with persistent memory and a growing skill library.

━━━ SELF-KNOWLEDGE ━━━
{self_context}

━━━ LEARNED SKILLS ━━━
{skills}

━━━ RELEVANT MEMORIES ━━━
{memories}

━━━ PAST EXPERIENCE WITH SIMILAR TASKS ━━━
{reflections}

━━━ DATE ━━━
{date}

━━━ AVAILABLE TOOLS ━━━
{tool_descriptions}

━━━ INSTRUCTIONS ━━━
Think step by step. Use tools to gather real information — never guess or make up facts.

YOUR RESPONSE MUST BE VALID JSON IN EXACTLY ONE OF THESE TWO FORMS:

  Use a tool:      {{"tool": "TOOL_NAME", "args": {{"param": "value"}}}}
  Final answer:    {{"final_answer": "Your complete, well-structured answer"}}

STRATEGY:
• recall_skill first if the task involves a domain you may know
• web_search + web_scrape before answering factual or technical questions
• learn_skill when encountering a new technology or technique
• remember to store important findings for future tasks
• Break complex goals into logical sub-steps using your tools
• Give a thorough, complete final_answer — not just "done"

Maximum {max_iter} tool calls. Budget wisely.\
"""

CORRECTION_MSG = (
    "Your last response was not valid JSON. You MUST output ONLY one of:\n"
    '  {"tool": "tool_name", "args": {"key": "value"}}\n'
    '  {"final_answer": "your complete answer"}\n'
    "No markdown, no explanation before or after. Output the JSON object only."
)


class AgentLoop:
    def __init__(self, model, tool_registry, memory_manager,
                 skill_registry, reflector=None, self_model=None,
                 max_iterations: int = 12):
        self._model = model
        self._tools = tool_registry
        self._memory = memory_manager
        self._skills = skill_registry
        self._reflector = reflector
        self._self_model = self_model
        self.max_iterations = max_iterations

    # ── public ───────────────────────────────────────────────────────────────

    def run(self, task: str, verbose: bool = False) -> str:
        system = self._build_system_prompt(task)
        messages = [{"role": "system", "content": system}]
        messages.extend(self._memory.short.get_messages(include_system=False))
        messages.append({"role": "user", "content": task})

        tool_schemas = self._tools.get_schemas()
        iterations = 0
        tools_used: list[str] = []
        final_answer = ""
        success = True

        while iterations < self.max_iterations:
            iterations += 1

            response, action = self._chat_with_retry(messages, tool_schemas)

            if verbose:
                print(f"  [iter {iterations}] {str(action)[:120]}")

            if action is None:
                # Gave up retrying — treat raw text as answer
                log.warning("All JSON retry attempts exhausted after %d iterations", iterations)
                final_answer = response
                break

            if "final_answer" in action:
                answer = str(action["final_answer"]).strip()
                if answer:
                    final_answer = answer
                else:
                    log.warning("Model returned empty final_answer, continuing loop")
                    continue
                break

            # Handle both {"tool": ..., "args": ...} and OpenAI tool_calls format
            calls = self._extract_tool_calls(action)
            if not calls:
                final_answer = response
                break

            for tool_name, args in calls:
                tools_used.append(tool_name)
                observation = self._tools.execute(tool_name, args)
                if verbose:
                    print(f"  [tool:{tool_name}] → {observation[:100]}")
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user",
                                  "content": f"Tool '{tool_name}' returned:\n{observation}"})

        if not final_answer:
            log.warning("Reached max_iterations=%d without a final answer (task='%s')",
                        self.max_iterations, task[:80])
            final_answer = (
                f"I reached the maximum number of steps ({self.max_iterations}) "
                f"without completing this task. Last response: {response[:400]}"
            )
            success = False

        # Post-task: reflect and update self-model
        self._post_task(task, tools_used, final_answer, success, iterations)

        return final_answer

    # ── internals ────────────────────────────────────────────────────────────

    def _chat_with_retry(self, messages: list[dict], tools: list[dict],
                          max_retries: int = 3) -> tuple[str, dict | None]:
        working = list(messages)
        for attempt in range(max_retries):
            response = self._model.chat(working, tools)
            action = _parse_action(response)
            if action is not None:
                return response, action
            if attempt < max_retries - 1:
                working = working + [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": CORRECTION_MSG},
                ]
        # All retries exhausted — return raw text with None action
        return response, None

    def _extract_tool_calls(self, action: dict) -> list[tuple[str, dict]]:
        """Normalize both {tool/args} and OpenAI {tool_calls} formats."""
        if "tool" in action:
            return [(action["tool"], action.get("args", {}))]
        if "tool_calls" in action:
            calls = []
            for tc in action["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    args = {}
                if name:
                    calls.append((name, args))
            return calls
        return []

    def _post_task(self, task: str, tools_used: list, answer: str,
                   success: bool, iterations: int):
        # Log to episodic memory
        self._memory.episodic.log(
            task=task[:500],
            action=", ".join(tools_used) or "direct_answer",
            observation=f"{iterations} iterations",
            outcome=answer[:500],
            success=success,
        )

        # Reflexion
        if self._reflector:
            reflection = self._reflector.reflect(task, tools_used, answer, success, iterations)
            if reflection and self._self_model:
                self._self_model.apply_reflection(
                    reflection.lesson,
                    reflection.what_failed,
                    reflection.skill_domain,
                    reflection.confidence,
                )

        # Update self-model task stats — infer domain from tools used and task text
        if self._self_model:
            domain = _infer_domain(task, tools_used)
            self._self_model.record_task(success, iterations, domain)

    def _build_system_prompt(self, task: str) -> str:
        skills = self._skills.list_all()
        if skills:
            skill_lines = []
            for s in skills[:12]:
                conf = s.get("confidence", 0.5)
                conf_label = "✓" if conf >= 0.7 else "~" if conf >= 0.5 else "?"
                skill_lines.append(f"  [{conf_label}] {s['name']} ({s['domain']}) — {s['description'][:60]}")
            skills_str = "\n".join(skill_lines)
        else:
            skills_str = "  None learned yet — use learn_skill to build your library"

        # Tool descriptions (one line each)
        tool_desc_lines = []
        for schema in self._tools.get_schemas():
            fn = schema["function"]
            params = list(fn.get("parameters", {}).get("properties", {}).keys())
            tool_desc_lines.append(f"  {fn['name']}({', '.join(params)}) — {fn['description']}")
        tools_str = "\n".join(tool_desc_lines)

        memories = self._memory.get_relevant_context(task, k=4)
        reflections = "\n".join(f"  • {r}" for r in self._memory.recall_reflections(task, 3)) or "  None yet"
        self_ctx = self._self_model.get_prompt_context() if self._self_model else "New agent — no history yet"

        return SYSTEM_PROMPT.format(
            self_context=self_ctx,
            skills=skills_str,
            memories=memories,
            reflections=reflections,
            date=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            tool_descriptions=tools_str,
            max_iter=self.max_iterations,
        )


# ── JSON parsing ─────────────────────────────────────────────────────────────

def _parse_action(text: str) -> dict | None:
    text = text.strip()

    # Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Extract from markdown code block
    for pattern in [r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    # Grab first {...} that contains a known key
    for m in re.finditer(r"\{[^{}]{10,}\}", text, re.DOTALL):
        try:
            obj = json.loads(m.group())
            if any(k in obj for k in ("tool", "final_answer", "tool_calls")):
                return obj
        except json.JSONDecodeError:
            continue

    return None


# ── Domain inference ─────────────────────────────────────────────────────────

_DOMAIN_KEYWORDS: list[tuple[str, list[str]]] = [
    ("blender",       ["blender", "bpy", "3d model", "mesh", "rigging", "sculpt"]),
    ("unity",         ["unity", "c#", "monobehaviour", "prefab", "unityengine"]),
    ("godot",         ["godot", "gdscript", "node3d", "characterbody"]),
    ("unreal_engine", ["unreal", "ue5", "blueprint", "nanite", "lumen"]),
    ("game_dev",      ["game", "level design", "gameplay", "spawn", "hitbox", "physics"]),
    ("python",        ["python", "flask", "fastapi", "django", "pandas", "numpy", "pytest"]),
    ("machine_learning", ["ml", "machine learning", "neural network", "pytorch", "tensorflow",
                          "scikit", "model training", "dataset"]),
    ("business",      ["business", "startup", "revenue", "marketing", "saas", "monetize"]),
    ("research",      ["research", "find information", "what is", "how does", "explain"]),
    ("skill_learning", ["learn", "skill", "tutorial", "how to", "guide"]),
]


def _infer_domain(task: str, tools_used: list[str]) -> str:
    """Infer the primary skill domain from task text and tools used."""
    task_lower = task.lower()
    # Check task text against domain keywords
    for domain, keywords in _DOMAIN_KEYWORDS:
        if any(kw in task_lower for kw in keywords):
            return domain
    # Fall back to tool-based inference
    if "learn_skill" in tools_used:
        return "skill_learning"
    if "web_search" in tools_used or "web_scrape" in tools_used:
        return "research"
    if "run_python" in tools_used:
        return "python"
    return "general"
