"""NexusAgent - main agent class and CLI interface."""
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from ..memory.manager import MemoryManager
from ..models.router import ModelRouter
from ..skills.registry import SkillRegistry
from ..skills.learner import SkillLearner
from ..skills.executor import SkillExecutor
from ..tools.registry import ToolRegistry
from .heartbeat import Heartbeat
from .loop import AgentLoop
from .planner import TaskPlanner

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table
    from rich import print as rprint
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None


def _print(msg: str, style: str = ""):
    if HAS_RICH and console:
        console.print(msg, style=style)
    else:
        print(msg)


def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


class NexusAgent:
    def __init__(self, config_path: str = "config.yaml"):
        load_dotenv()
        self._config = _load_config(config_path)

        data_dir = Path(os.getenv("NEXUS_DATA_DIR", self._config.get("agent", {}).get("data_dir", "./data")))
        data_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = data_dir

        model_cfg = self._config.get("models", {})
        self._model = ModelRouter(model_cfg)

        mem_cfg = self._config.get("memory", {})
        self._memory = MemoryManager(
            data_dir,
            short_term_limit=mem_cfg.get("short_term_limit", 20),
            long_term_backend=mem_cfg.get("long_term_backend", "chromadb"),
        )
        self._memory.set_embed_fn(self._model.embed)

        skill_cfg = self._config.get("skills", {})
        self._skills = SkillRegistry(Path(skill_cfg.get("skill_dir", str(data_dir / "skills"))))
        self._learner = SkillLearner(self._skills, model_client=self._model)
        self._executor = SkillExecutor(self._skills, model_client=self._model)

        tools_cfg = self._config.get("tools", {})
        self._tools = ToolRegistry(tools_cfg, skill_registry=self._skills, memory_manager=self._memory)

        self._loop = AgentLoop(
            model=self._model,
            tool_registry=self._tools,
            memory_manager=self._memory,
            skill_registry=self._skills,
            max_iterations=12,
        )
        self._planner = TaskPlanner(self._model)

        hb_cfg = self._config.get("heartbeat", {})
        self._heartbeat = Heartbeat(
            data_dir=data_dir,
            interval=hb_cfg.get("interval", 300),
            on_beat=self._on_beat,
        )
        if hb_cfg.get("enabled", True):
            self._heartbeat.start()

        self._started_at = datetime.utcnow()

    def chat(self, message: str, verbose: bool = False) -> str:
        self._memory.short.add("user", message)
        response = self._loop.run(message, verbose=verbose)
        self._memory.short.add("assistant", response)
        return response

    def learn(self, topic: str, domain: str = "auto") -> str:
        _print(f"[bold cyan]Researching '{topic}'...[/bold cyan]" if HAS_RICH else f"Researching '{topic}'...")
        skill = self._learner.learn(topic, domain)
        return (
            f"Learned skill: {skill['name']}\n"
            f"Domain: {skill['domain']}\n"
            f"Procedure preview: {skill['procedure'][:300]}..."
        )

    def status(self) -> dict:
        mem_stats = self._memory.stats()
        skill_count = len(self._skills.list_all())
        return {
            "agent": self._config.get("agent", {}).get("name", "NexusAgent"),
            "version": self._config.get("agent", {}).get("version", "1.0.0"),
            "uptime_seconds": (datetime.utcnow() - self._started_at).total_seconds(),
            "model_backend": self._model.active_backend,
            "heartbeat": {
                "alive": self._heartbeat.is_alive(),
                "beats": self._heartbeat.beat_count,
                "uptime_seconds": self._heartbeat.uptime_seconds,
            },
            "memory": mem_stats,
            "skills_learned": skill_count,
            "available_tools": self._tools.names(),
        }

    def run_cli(self):
        self._print_banner()
        _print("Type [bold]/help[/bold] for commands, or just chat. [bold]/quit[/bold] to exit.\n" if HAS_RICH
               else "Type /help for commands, or just chat. /quit to exit.\n")

        while True:
            try:
                raw = input("you> ").strip()
            except (KeyboardInterrupt, EOFError):
                _print("\nExiting NexusAgent. Goodbye!")
                break

            if not raw:
                continue

            if raw.startswith("/"):
                self._handle_command(raw)
            else:
                _print("\nnexus> " if not HAS_RICH else "[bold green]nexus>[/bold green] ", style="")
                response = self.chat(raw)
                if HAS_RICH and console:
                    console.print(Markdown(response))
                else:
                    print(response)
                print()

    def _handle_command(self, cmd: str):
        parts = cmd.strip().split(maxsplit=1)
        verb = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if verb in ("/quit", "/exit", "/q"):
            _print("Goodbye!")
            sys.exit(0)

        elif verb == "/help":
            self._print_help()

        elif verb == "/status":
            self._print_status()

        elif verb == "/skills":
            skills = self._skills.list_all()
            if not skills:
                _print("No skills learned yet. Use /learn <topic> to learn one.")
            else:
                if HAS_RICH:
                    t = Table(title=f"Learned Skills ({len(skills)})")
                    t.add_column("Name"); t.add_column("Domain"); t.add_column("Used"); t.add_column("Description")
                    for s in skills:
                        t.add_row(s["name"], s["domain"], str(s["use_count"]), s["description"][:50])
                    console.print(t)
                else:
                    for s in skills:
                        print(f"  [{s['domain']}] {s['name']} (used {s['use_count']}x)")

        elif verb == "/learn":
            if not arg:
                _print("Usage: /learn <topic>  e.g. /learn Blender 3D modeling")
            else:
                result = self.learn(arg)
                _print(result)

        elif verb == "/skill":
            if not arg:
                _print("Usage: /skill <name>")
            else:
                result = self._tools._recall_skill(arg)
                if HAS_RICH:
                    console.print(Panel(result, title=f"Skill: {arg}"))
                else:
                    print(result)

        elif verb == "/memory":
            stats = self._memory.stats()
            _print(f"Short-term: {stats['short_term_messages']} messages")
            _print(f"Long-term: {stats['long_term_entries']} entries")
            ep = stats["episodes"]
            _print(f"Episodes: {ep['total_episodes']} ({ep['successful']} successful)")

        elif verb == "/episodes":
            episodes = self._memory.episodic.recall(limit=10)
            if not episodes:
                _print("No episodes recorded yet.")
            else:
                for ep in episodes:
                    _print(f"[{ep['timestamp'][:19]}] {ep['task'][:60]} → {ep['outcome'][:60]}")

        elif verb == "/clear":
            self._memory.short.clear()
            _print("Short-term memory cleared.")

        elif verb == "/verbose":
            self._verbose = not getattr(self, "_verbose", False)
            _print(f"Verbose mode: {'ON' if self._verbose else 'OFF'}")

        elif verb == "/model":
            _print(f"Active model backend: {self._model.active_backend}")

        else:
            _print(f"Unknown command: {verb}. Type /help for commands.")

    def _print_banner(self):
        if HAS_RICH:
            console.print(Panel.fit(
                "[bold cyan]NexusAgent[/bold cyan] [dim]v1.0.0[/dim]\n"
                "[dim]Local Self-Learning AI Agent[/dim]\n"
                f"[dim]Model: {self._model.active_backend} | Skills: {len(self._skills.list_all())} | Heartbeat: {'active' if self._heartbeat.is_alive() else 'off'}[/dim]",
                border_style="cyan",
            ))
        else:
            print("=" * 50)
            print("  NexusAgent v1.0.0 - Local Self-Learning AI")
            print(f"  Model: {self._model.active_backend}")
            print("=" * 50)

    def _print_help(self):
        help_text = """
**Commands:**
- `/learn <topic>` — Research and learn a new skill (e.g. `/learn Blender scripting`)
- `/skills` — List all learned skills
- `/skill <name>` — Show details of a skill
- `/status` — Show agent status and stats
- `/memory` — Show memory statistics
- `/episodes` — Show recent action history
- `/clear` — Clear short-term memory
- `/model` — Show active model backend
- `/verbose` — Toggle verbose tool call logging
- `/quit` — Exit

**Just chat to use the agent.** It will search the web, run code, learn skills, and remember things automatically.

**Example prompts:**
- "Learn how to create 3D objects in Blender using Python"
- "Research how to build a Unity game from scratch"
- "Write a Python web scraper for news articles"
- "What are the best practices for starting a software company?"
"""
        if HAS_RICH:
            console.print(Markdown(help_text))
        else:
            print(help_text)

    def _print_status(self):
        s = self.status()
        uptime = int(s["uptime_seconds"])
        h, m, sec = uptime // 3600, (uptime % 3600) // 60, uptime % 60
        if HAS_RICH:
            t = Table(title="NexusAgent Status")
            t.add_column("Property"); t.add_column("Value")
            t.add_row("Agent", s["agent"])
            t.add_row("Model Backend", s["model_backend"])
            t.add_row("Uptime", f"{h:02d}:{m:02d}:{sec:02d}")
            t.add_row("Heartbeat", f"{'alive' if s['heartbeat']['alive'] else 'stopped'} ({s['heartbeat']['beats']} beats)")
            t.add_row("Skills Learned", str(s["skills_learned"]))
            t.add_row("Long-term Memories", str(s["memory"]["long_term_entries"]))
            t.add_row("Episodes Logged", str(s["memory"]["episodes"]["total_episodes"]))
            t.add_row("Available Tools", ", ".join(s["available_tools"]))
            console.print(t)
        else:
            print(f"Agent: {s['agent']} | Model: {s['model_backend']} | Uptime: {h:02d}:{m:02d}:{sec:02d}")
            print(f"Skills: {s['skills_learned']} | Memories: {s['memory']['long_term_entries']} | Tools: {len(s['available_tools'])}")

    def _on_beat(self, beat_count: int):
        try:
            self._memory.consolidate()
        except Exception:
            pass

    def shutdown(self):
        self._heartbeat.stop()
        self._memory.episodic.close()
