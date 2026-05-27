"""Tool registry with OpenAI-compatible function schemas."""
from . import web_search, web_scrape, code_runner, file_ops


class ToolRegistry:
    def __init__(self, config: dict = None, skill_registry=None,
                 memory_manager=None, skill_learner=None):
        self._config = config or {}
        self._skills = skill_registry
        self._memory = memory_manager
        self._learner = skill_learner
        self._tools = self._build()

    def _build(self) -> dict:
        cfg = self._config
        tools: dict = {}

        # ── Web tools ──────────────────────────────────────────────────────
        if cfg.get("web_search_enabled", True):
            n = cfg.get("max_search_results", 5)
            tools["web_search"] = {
                "func": lambda query: _fmt_search(web_search.search(query, n)),
                "description": "Search the web for current information, documentation, or research.",
                "parameters": {"type": "object",
                                "properties": {"query": {"type": "string", "description": "Search query"}},
                                "required": ["query"]},
            }
            tools["web_scrape"] = {
                "func": lambda url: web_scrape.scrape(url),
                "description": "Fetch and read the text content of a web page.",
                "parameters": {"type": "object",
                                "properties": {"url": {"type": "string", "description": "Full URL to fetch"}},
                                "required": ["url"]},
            }

        # ── Code execution ─────────────────────────────────────────────────
        if cfg.get("code_execution_enabled", True):
            timeout = cfg.get("code_execution_timeout", 30)
            tools["run_python"] = {
                "func": lambda code: _fmt_run(code_runner.run_python(code, timeout)),
                "description": "Execute Python code and return stdout/stderr. Use for calculations, data processing, testing snippets.",
                "parameters": {"type": "object",
                                "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                                "required": ["code"]},
            }
            tools["run_shell"] = {
                "func": lambda command: _fmt_run(code_runner.run_shell(command, timeout)),
                "description": "Execute a shell command.",
                "parameters": {"type": "object",
                                "properties": {"command": {"type": "string", "description": "Shell command"}},
                                "required": ["command"]},
            }

        # ── File tools ─────────────────────────────────────────────────────
        tools["read_file"] = {
            "func": lambda path: file_ops.read_file(path),
            "description": "Read the contents of a file.",
            "parameters": {"type": "object",
                            "properties": {"path": {"type": "string", "description": "File path"}},
                            "required": ["path"]},
        }
        tools["write_file"] = {
            "func": lambda path, content, append=False: file_ops.write_file(path, content, append),
            "description": "Write or append content to a file.",
            "parameters": {"type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                                "append": {"type": "boolean", "description": "Append instead of overwrite"},
                            },
                            "required": ["path", "content"]},
        }
        tools["list_files"] = {
            "func": lambda directory=".", pattern="*": file_ops.list_files(directory, pattern),
            "description": "List files in a directory.",
            "parameters": {"type": "object",
                            "properties": {
                                "directory": {"type": "string"},
                                "pattern": {"type": "string"},
                            },
                            "required": []},
        }

        # ── Memory tools ───────────────────────────────────────────────────
        if self._memory:
            tools["remember"] = {
                "func": lambda content, importance=0.7: self._do_remember(content, importance),
                "description": "Store important information in long-term memory for future recall.",
                "parameters": {"type": "object",
                                "properties": {
                                    "content": {"type": "string", "description": "Text to remember"},
                                    "importance": {"type": "number", "description": "Importance 0.0-1.0"},
                                },
                                "required": ["content"]},
            }
            tools["recall_memory"] = {
                "func": lambda query, k=5: self._do_recall(query, k),
                "description": "Search long-term memory for relevant past information.",
                "parameters": {"type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "k": {"type": "integer", "description": "Number of results"},
                                },
                                "required": ["query"]},
            }

        # ── Skill tools ────────────────────────────────────────────────────
        if self._skills:
            tools["learn_skill"] = {
                "func": lambda topic, domain="auto": self._do_learn(topic, domain),
                "description": "Research and learn a new skill or technique (Blender, Unity, Python pattern, etc). Stores procedure and code template.",
                "parameters": {"type": "object",
                                "properties": {
                                    "topic": {"type": "string", "description": "Skill topic to research and learn"},
                                    "domain": {"type": "string", "description": "Domain hint: blender/unity/godot/python/etc"},
                                },
                                "required": ["topic"]},
            }
            tools["recall_skill"] = {
                "func": lambda query, domain=None: self._do_recall_skill(query, domain),
                "description": "Retrieve a previously learned skill from the library.",
                "parameters": {"type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "domain": {"type": "string"},
                                },
                                "required": ["query"]},
            }
            tools["list_skills"] = {
                "func": lambda domain=None: self._do_list_skills(domain),
                "description": "List all skills in the agent's skill library.",
                "parameters": {"type": "object",
                                "properties": {"domain": {"type": "string", "description": "Filter by domain"}},
                                "required": []},
            }

        return tools

    # ── execution ─────────────────────────────────────────────────────────

    def execute(self, name: str, args: dict) -> str:
        tool = self._tools.get(name)
        if not tool:
            avail = list(self._tools.keys())
            return f"Unknown tool '{name}'. Available: {avail}"
        try:
            result = tool["func"](**args)
            return str(result) if result is not None else "Done."
        except TypeError as e:
            return f"Wrong arguments for '{name}': {e}"
        except Exception as e:
            return f"Tool '{name}' error: {e}"

    def get_schemas(self) -> list[dict]:
        return [
            {"type": "function", "function": {"name": n, "description": t["description"], "parameters": t["parameters"]}}
            for n, t in self._tools.items()
        ]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    # ── tool implementations ──────────────────────────────────────────────

    def _do_remember(self, content: str, importance: float = 0.7) -> str:
        self._memory.store_fact(content, importance=importance)
        return f"Stored in memory: \"{content[:80]}...\""

    def _do_recall(self, query: str, k: int = 5) -> str:
        entries = self._memory.recall(query, k)
        if not entries:
            return "No relevant memories found."
        lines = [f"[{e.memory_type}|{e.importance:.1f}] {e.text[:280]}" for e in entries]
        return "\n---\n".join(lines)

    def _do_learn(self, topic: str, domain: str = "auto") -> str:
        if not self._learner:
            return "Skill learner not available."
        skill = self._learner.learn(topic, domain)
        conf = skill.get("confidence", 0.5)
        issues = skill.get("validation_issues", [])
        issue_note = f" | Issues: {'; '.join(issues[:2])}" if issues else ""
        return (
            f"Learned: {skill['name']} [{skill['domain']}] confidence={conf:.0%}{issue_note}\n\n"
            f"Procedure:\n{skill['procedure'][:600]}\n\n"
            f"Code template:\n{skill.get('code_template', 'N/A')[:400]}"
        )

    def _do_recall_skill(self, query: str, domain: str = None) -> str:
        results = self._skills.search(query, domain)
        if not results:
            return f"No skill found for '{query}'. Use learn_skill to learn it."
        s = results[0]
        conf = s.get("confidence", 0.5)
        conf_note = " ⚠ low confidence" if conf < 0.5 else ""
        return (
            f"Skill: {s['name']}  [{s['domain']}]  confidence={conf:.0%}{conf_note}\n\n"
            f"Procedure:\n{s['procedure']}\n\n"
            f"Code Template:\n{s.get('code_template', 'N/A')}"
        )

    def _do_list_skills(self, domain: str = None) -> str:
        skills = self._skills.list_all(domain=domain)
        if not skills:
            return "No skills in library yet."
        lines = [f"• [{s.get('confidence',0.5):.0%}] [{s['domain']}] {s['name']} — {s['description'][:60]}"
                 for s in skills]
        return f"Skill library ({len(skills)}):\n" + "\n".join(lines)


# ── formatting helpers ────────────────────────────────────────────────────────

def _fmt_search(results: list[dict]) -> str:
    if not results:
        return "No search results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}")
    return "\n\n".join(lines)


def _fmt_run(result: dict) -> str:
    status = "SUCCESS" if result["success"] else "FAILED"
    parts = [f"[{status}]"]
    if result.get("stdout"):
        parts.append(f"stdout:\n{result['stdout']}")
    if result.get("stderr"):
        parts.append(f"stderr:\n{result['stderr']}")
    return "\n".join(parts)
