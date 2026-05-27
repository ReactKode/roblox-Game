"""Tool registry with OpenAI-compatible function schemas."""
from . import web_search, web_scrape, code_runner, file_ops


class ToolRegistry:
    def __init__(self, config: dict = None, skill_registry=None, memory_manager=None):
        self._config = config or {}
        self._skill_registry = skill_registry
        self._memory = memory_manager
        self._tools = self._build()

    def _build(self) -> dict:
        cfg = self._config
        tools = {}

        if cfg.get("web_search_enabled", True):
            max_results = cfg.get("max_search_results", 5)
            tools["web_search"] = {
                "func": lambda query, max_results=max_results: self._format_search(
                    web_search.search(query, max_results)
                ),
                "description": "Search the web for information.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                    "required": ["query"],
                },
            }
            tools["web_scrape"] = {
                "func": lambda url: web_scrape.scrape(url),
                "description": "Fetch and read the content of a web page.",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string", "description": "URL to scrape"}},
                    "required": ["url"],
                },
            }

        if cfg.get("code_execution_enabled", True):
            timeout = cfg.get("code_execution_timeout", 30)
            tools["run_python"] = {
                "func": lambda code: self._format_run(code_runner.run_python(code, timeout)),
                "description": "Execute Python code and return stdout/stderr.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "Python code to run"}},
                    "required": ["code"],
                },
            }
            tools["run_shell"] = {
                "func": lambda command: self._format_run(code_runner.run_shell(command, timeout)),
                "description": "Execute a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string", "description": "Shell command"}},
                    "required": ["command"],
                },
            }

        tools["read_file"] = {
            "func": lambda path: file_ops.read_file(path),
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path"}},
                "required": ["path"],
            },
        }
        tools["write_file"] = {
            "func": lambda path, content, append=False: file_ops.write_file(path, content, append),
            "description": "Write content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"},
                    "append": {"type": "boolean", "description": "Append to existing file"},
                },
                "required": ["path", "content"],
            },
        }
        tools["list_files"] = {
            "func": lambda directory=".", pattern="*": file_ops.list_files(directory, pattern),
            "description": "List files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory path"},
                    "pattern": {"type": "string", "description": "Glob pattern"},
                },
                "required": [],
            },
        }

        if self._memory:
            tools["remember"] = {
                "func": lambda content, metadata=None: self._remember(content, metadata),
                "description": "Store information in long-term memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Information to remember"},
                        "metadata": {"type": "object", "description": "Optional metadata tags"},
                    },
                    "required": ["content"],
                },
            }
            tools["recall_memory"] = {
                "func": lambda query, k=5: self._recall(query, k),
                "description": "Search long-term memory for relevant information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for"},
                        "k": {"type": "integer", "description": "Number of results"},
                    },
                    "required": ["query"],
                },
            }

        if self._skill_registry:
            tools["learn_skill"] = {
                "func": lambda topic, domain="general": self._learn_skill(topic, domain),
                "description": "Research and learn a new skill or technique. Use for learning Blender, Unity, coding patterns, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Skill or topic to learn"},
                        "domain": {"type": "string", "description": "Domain category (blender, unity, godot, python, etc.)"},
                    },
                    "required": ["topic"],
                },
            }
            tools["recall_skill"] = {
                "func": lambda query, domain=None: self._recall_skill(query, domain),
                "description": "Retrieve a previously learned skill from the skill library.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Skill name or description"},
                        "domain": {"type": "string", "description": "Optional domain filter"},
                    },
                    "required": ["query"],
                },
            }
            tools["list_skills"] = {
                "func": lambda domain=None: self._list_skills(domain),
                "description": "List all learned skills.",
                "parameters": {
                    "type": "object",
                    "properties": {"domain": {"type": "string", "description": "Filter by domain"}},
                    "required": [],
                },
            }

        return tools

    def execute(self, tool_name: str, args: dict) -> str:
        tool = self._tools.get(tool_name)
        if not tool:
            return f"Unknown tool: {tool_name}. Available: {list(self._tools.keys())}"
        try:
            return str(tool["func"](**args))
        except TypeError as e:
            return f"Tool argument error for '{tool_name}': {e}"
        except Exception as e:
            return f"Tool '{tool_name}' error: {e}"

    def get_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for name, tool in self._tools.items()
        ]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def inject_skill_registry(self, registry):
        self._skill_registry = registry
        self._tools = self._build()

    def inject_memory(self, memory):
        self._memory = memory
        self._tools = self._build()

    # --- helpers ---

    def _format_search(self, results: list[dict]) -> str:
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}")
        return "\n\n".join(lines)

    def _format_run(self, result: dict) -> str:
        status = "SUCCESS" if result["success"] else "FAILED"
        out = result.get("stdout", "")
        err = result.get("stderr", "")
        parts = [f"[{status}]"]
        if out:
            parts.append(f"stdout:\n{out}")
        if err:
            parts.append(f"stderr:\n{err}")
        return "\n".join(parts)

    def _remember(self, content: str, metadata=None) -> str:
        self._memory.long.add(content, metadata=metadata or {})
        return f"Stored in memory: {content[:100]}..."

    def _recall(self, query: str, k: int = 5) -> str:
        results = self._memory.recall(query, k)
        if not results:
            return "No relevant memories found."
        return "\n---\n".join(f"[score={r['score']:.2f}] {r['text'][:300]}" for r in results)

    def _learn_skill(self, topic: str, domain: str = "general") -> str:
        if not self._skill_registry:
            return "Skill registry not available."
        from ..skills.learner import SkillLearner
        learner = SkillLearner(self._skill_registry, self)
        skill = learner.learn(topic, domain)
        return f"Learned skill '{skill['name']}' in domain '{skill['domain']}'. Procedures stored."

    def _recall_skill(self, query: str, domain=None) -> str:
        results = self._skill_registry.search(query, domain)
        if not results:
            return f"No skill found for '{query}'."
        skill = results[0]
        return f"Skill: {skill['name']}\nDomain: {skill['domain']}\n\nProcedure:\n{skill['procedure']}\n\nCode Template:\n{skill.get('code_template', 'N/A')}"

    def _list_skills(self, domain=None) -> str:
        skills = self._skill_registry.list_all(domain=domain)
        if not skills:
            return "No skills learned yet."
        lines = [f"• [{s['domain']}] {s['name']} — {s['description'][:80]}" for s in skills]
        return f"Learned skills ({len(skills)}):\n" + "\n".join(lines)
