"""ReAct agent loop - Think → Act → Observe → Repeat."""
import json
import re
from datetime import datetime


SYSTEM_PROMPT = """You are NexusAgent, a powerful self-learning AI agent running locally.
You have access to memory, tools, and a growing library of learned skills.

Current date: {date}
Available tools: {tools}
Learned skills: {skills}

Recent relevant memories:
{memories}

## How to respond:

Think step-by-step. You have two response modes:

1. **Use a tool** - respond with ONLY this JSON (nothing else):
{{"tool": "tool_name", "args": {{"arg1": "value1"}}}}

2. **Final answer** - when you have enough information, respond with:
{{"final_answer": "Your complete answer here"}}

Rules:
- Always think about what information you need before acting
- Use web_search and web_scrape to research things you don't know
- Use learn_skill when asked to learn a new technology or technique
- Use recall_skill before trying to learn something (might already know it)
- Use remember to store important findings for future use
- Never make up information - research it
- Maximum {max_iter} tool calls before giving a final answer
"""


class AgentLoop:
    def __init__(self, model, tool_registry, memory_manager, skill_registry, max_iterations: int = 10):
        self._model = model
        self._tools = tool_registry
        self._memory = memory_manager
        self._skills = skill_registry
        self.max_iterations = max_iterations

    def run(self, task: str, verbose: bool = False) -> str:
        skills_summary = self._get_skills_summary()
        memories = self._get_relevant_memories(task)
        system = SYSTEM_PROMPT.format(
            date=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            tools=", ".join(self._tools.names()),
            skills=skills_summary,
            memories=memories,
            max_iter=self.max_iterations,
        )

        messages = [{"role": "system", "content": system}]
        messages.extend(self._memory.short.get_messages(include_system=False))
        messages.append({"role": "user", "content": task})

        tool_schemas = self._tools.get_schemas()
        iterations = 0
        final_answer = ""
        used_tools = []

        while iterations < self.max_iterations:
            iterations += 1
            response = self._model.chat(messages, tools=tool_schemas)

            if verbose:
                print(f"  [Loop {iterations}] Raw: {response[:200]}")

            # Parse response
            action = self._parse_action(response)

            if action is None:
                # Plain text response - treat as final answer
                final_answer = response
                break

            if "final_answer" in action:
                final_answer = action["final_answer"]
                break

            if "tool" in action:
                tool_name = action["tool"]
                args = action.get("args", {})
                used_tools.append(tool_name)
                observation = self._tools.execute(tool_name, args)
                if verbose:
                    print(f"  [Tool: {tool_name}] -> {observation[:150]}")
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Tool '{tool_name}' result:\n{observation}"})
                continue

            if "tool_calls" in action:
                # Handle OpenAI-style tool_calls format
                for tc in action["tool_calls"]:
                    fn = tc.get("function", {})
                    tool_name = fn.get("name", "")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except Exception:
                        args = {}
                    used_tools.append(tool_name)
                    observation = self._tools.execute(tool_name, args)
                    if verbose:
                        print(f"  [Tool: {tool_name}] -> {observation[:150]}")
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"Tool '{tool_name}' result:\n{observation}"})
                continue

            final_answer = response
            break

        if not final_answer:
            final_answer = f"Reached maximum iterations ({self.max_iterations}) without a final answer. Last response: {response[:500]}"

        self._memory.episodic.log(
            task=task[:500],
            action=", ".join(used_tools),
            observation=f"Completed in {iterations} iterations",
            outcome=final_answer[:500],
            success=True,
        )

        return final_answer

    def _parse_action(self, response: str) -> dict | None:
        response = response.strip()

        # Try direct JSON parse
        try:
            obj = json.loads(response)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        # Extract JSON block
        json_patterns = [
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
            r"(\{[^{}]*\"(?:tool|final_answer|tool_calls)\"[^{}]*\})",
        ]
        for pattern in json_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

        # Check for tool_calls JSON (from model responses)
        if '"tool_calls"' in response:
            try:
                start = response.index("{")
                return json.loads(response[start:])
            except Exception:
                pass

        return None

    def _get_skills_summary(self) -> str:
        skills = self._skills.list_all()
        if not skills:
            return "None learned yet"
        return ", ".join(f"{s['name']}({s['domain']})" for s in skills[:10])

    def _get_relevant_memories(self, query: str) -> str:
        results = self._memory.recall(query, k=3)
        if not results:
            return "None"
        return "\n".join(f"- {r['text'][:200]}" for r in results)
