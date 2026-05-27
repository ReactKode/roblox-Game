"""NexusAgent — main orchestrator, CLI, and public API."""
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
from ..skills.validator import SkillValidator
from ..skills.executor import SkillExecutor
from ..tools.registry import ToolRegistry
from .heartbeat import Heartbeat
from .loop import AgentLoop
from .planner import TaskPlanner
from .reflector import Reflector
from .self_model import SelfModel

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.text import Text
    HAS_RICH = True
    _console = Console()
except ImportError:
    HAS_RICH = False
    _console = None


def _p(msg, style=""):
    if HAS_RICH and _console:
        _console.print(msg, style=style)
    else:
        print(msg)


def _load_cfg(path: str) -> dict:
    p = Path(path)
    return yaml.safe_load(p.read_text()) if p.exists() else {}


class NexusAgent:
    """
    Public API:
      agent.chat(message) -> str
      agent.learn(topic)  -> str
      agent.status()      -> dict
      agent.run_cli()     -> interactive loop
    """

    def __init__(self, config_path: str = "config.yaml"):
        load_dotenv()
        cfg = _load_cfg(config_path)

        data_dir = Path(os.getenv("NEXUS_DATA_DIR", cfg.get("agent", {}).get("data_dir", "./data")))
        data_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = data_dir
        self._started = datetime.utcnow()
        self._verbose = False

        # ── Model ──────────────────────────────────────────────────────────
        self._model = ModelRouter(cfg.get("models", {}))

        # ── Memory ────────────────────────────────────────────────────────
        mem_cfg = cfg.get("memory", {})
        self._memory = MemoryManager(
            data_dir,
            short_term_limit=mem_cfg.get("short_term_limit", 20),
            backend=mem_cfg.get("long_term_backend", "auto"),
        )
        self._memory.set_embed_fn(self._model.embed)

        # ── Self-model ────────────────────────────────────────────────────
        self._self_model = SelfModel(data_dir)

        # ── Skills ────────────────────────────────────────────────────────
        skill_dir = Path(cfg.get("skills", {}).get("skill_dir", str(data_dir / "skills")))
        self._skills = SkillRegistry(skill_dir)
        self._validator = SkillValidator(self._model)
        self._learner = SkillLearner(self._skills, self._validator, self._model, self._memory)
        self._executor = SkillExecutor(self._skills, model_client=self._model)

        # ── Tools ─────────────────────────────────────────────────────────
        tools_cfg = cfg.get("tools", {})
        self._tools = ToolRegistry(
            tools_cfg,
            skill_registry=self._skills,
            memory_manager=self._memory,
            skill_learner=self._learner,
        )

        # ── Reasoning ─────────────────────────────────────────────────────
        self._reflector = Reflector(self._model, self._memory.episodic)
        self._planner = TaskPlanner(self._model)
        self._loop = AgentLoop(
            model=self._model,
            tool_registry=self._tools,
            memory_manager=self._memory,
            skill_registry=self._skills,
            reflector=self._reflector,
            self_model=self._self_model,
            max_iterations=cfg.get("agent", {}).get("max_iterations", 12),
        )

        # ── Heartbeat ─────────────────────────────────────────────────────
        hb_cfg = cfg.get("heartbeat", {})
        self._heartbeat = Heartbeat(
            data_dir=data_dir,
            interval=hb_cfg.get("interval", 300),
            memory_manager=self._memory,
            skill_registry=self._skills,
            skill_learner=self._learner,
        )
        if hb_cfg.get("enabled", True):
            self._heartbeat.start()

    # ── Public API ────────────────────────────────────────────────────────

    def chat(self, message: str) -> str:
        self._memory.short.add("user", message)

        # Route complex tasks through the planner
        if self._planner.is_complex(message):
            response = self._run_planned(message)
        else:
            response = self._loop.run(message, verbose=self._verbose)

        self._memory.short.add("assistant", response)
        return response

    def learn(self, topic: str) -> str:
        skill = self._learner.learn(topic)
        conf = skill.get("confidence", 0.5)
        status = "validated ✓" if skill.get("validated") else "unvalidated"
        issues = skill.get("validation_issues", [])
        issue_str = ("\n  Issues: " + "; ".join(issues)) if issues else ""
        self._self_model.record_skill(skill["name"], skill["domain"], conf)
        return (
            f"Learned: {skill['name']}\n"
            f"Domain:  {skill['domain']}\n"
            f"Confidence: {conf:.0%} ({status}){issue_str}\n\n"
            f"Procedure preview:\n{skill['procedure'][:400]}..."
        )

    def status(self) -> dict:
        mem = self._memory.stats()
        skills = self._skills.list_all()
        uptime = (datetime.utcnow() - self._started).total_seconds()
        profile = self._self_model.profile
        return {
            "agent": "NexusAgent",
            "version": "2.0.0",
            "uptime_seconds": round(uptime, 1),
            "model_backend": self._model.active_backend,
            "heartbeat": {
                "alive": self._heartbeat.is_alive(),
                "beats": self._heartbeat.beat_count,
                "uptime_s": round(self._heartbeat.uptime_seconds, 1),
            },
            "memory": mem,
            "skills": {
                "count": len(skills),
                "confidence": {s["name"]: s.get("confidence", 0.5) for s in skills},
            },
            "self_model": {
                "tasks_completed": profile["stats"]["tasks_completed"],
                "tasks_failed": profile["stats"]["tasks_failed"],
                "reflections": profile["stats"]["reflections_stored"],
                "strengths": profile["strengths"],
                "weaknesses": profile["weaknesses"],
            },
            "tools": self._tools.names(),
        }

    def run_cli(self):
        self._print_banner()
        _p("Type [bold]/help[/bold] for commands, or just chat naturally.\n" if HAS_RICH
           else "Type /help for commands, or just chat. /quit to exit.\n")

        while True:
            try:
                user_input = input("\n[you] " if not HAS_RICH else "")
                if HAS_RICH:
                    _console.print("[bold cyan]you>[/bold cyan] ", end="")
                    user_input = input()
                user_input = user_input.strip()
            except (KeyboardInterrupt, EOFError):
                _p("\nGoodbye.")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                self._handle_command(user_input)
            else:
                self._run_with_spinner(user_input)

    # ── private ───────────────────────────────────────────────────────────

    def _run_planned(self, task: str) -> str:
        subtasks = self._planner.decompose(task, self._tools.names())
        if len(subtasks) <= 1:
            return self._loop.run(task, verbose=self._verbose)

        results = []
        for st in subtasks:
            if self._verbose:
                _p(f"  [plan] Step {st.id}: {st.description}")
            result = self._loop.run(st.description, verbose=self._verbose)
            st.completed = True
            st.result = result
            results.append(f"**Step {st.id} — {st.description}**\n{result}")

        # Synthesize all step results into a coherent final answer
        synthesis_prompt = (
            f"Original goal: {task}\n\nResults from each step:\n\n" +
            "\n\n---\n\n".join(results) +
            "\n\nProvide a clear, unified summary of what was accomplished."
        )
        return self._loop.run(synthesis_prompt, verbose=self._verbose)

    def _run_with_spinner(self, message: str):
        if HAS_RICH and _console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Thinking...[/cyan]"),
                console=_console,
                transient=True,
            ) as prog:
                prog.add_task("", total=None)
                response = self.chat(message)
            _console.print("\n[bold green]nexus>[/bold green]")
            _console.print(Markdown(response))
        else:
            print("nexus> thinking...")
            response = self.chat(message)
            print(f"\nnexus> {response}")

    def _handle_command(self, raw: str):
        parts = raw.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        commands = {
            "/quit": self._cmd_quit, "/exit": self._cmd_quit, "/q": self._cmd_quit,
            "/help": self._cmd_help,
            "/status": self._cmd_status,
            "/skills": self._cmd_skills,
            "/learn": self._cmd_learn,
            "/skill": self._cmd_skill,
            "/memory": self._cmd_memory,
            "/episodes": self._cmd_episodes,
            "/profile": self._cmd_profile,
            "/clear": self._cmd_clear,
            "/verbose": self._cmd_verbose,
            "/model": self._cmd_model,
        }
        fn = commands.get(cmd)
        if fn:
            fn(arg)
        else:
            _p(f"Unknown command: {cmd}. Type /help.")

    # ── CLI command handlers ───────────────────────────────────────────────

    def _cmd_quit(self, _):
        _p("Saving state and exiting...")
        self.shutdown()
        sys.exit(0)

    def _cmd_help(self, _):
        text = """
**NexusAgent Commands**

| Command | Description |
|---|---|
| `/learn <topic>` | Research & learn a skill (Blender, Unity, Godot, Python...) |
| `/skills` | List all learned skills with confidence scores |
| `/skill <name>` | Show full skill details and code template |
| `/status` | Full agent status |
| `/profile` | Self-model: strengths, weaknesses, lessons |
| `/memory` | Memory statistics |
| `/episodes` | Recent action history |
| `/clear` | Clear short-term memory |
| `/verbose` | Toggle verbose tool-call logging |
| `/model` | Show active model backend |
| `/quit` | Exit |

**Examples:**
- `How do I create a 3D mesh in Blender with Python?`
- `/learn Godot 4 GDScript`
- `Write a FastAPI server with authentication`
- `Research the best ways to monetize a SaaS product`
- `Build me a Python web scraper for job listings`
"""
        if HAS_RICH:
            _console.print(Markdown(text))
        else:
            print(text)

    def _cmd_status(self, _):
        s = self.status()
        uptime = int(s["uptime_seconds"])
        h, m, sec = uptime // 3600, (uptime % 3600) // 60, uptime % 60
        mem = s["memory"]
        sm = s["self_model"]
        if HAS_RICH:
            t = Table(title="NexusAgent Status", show_header=True, header_style="bold cyan")
            t.add_column("Property", style="dim"); t.add_column("Value")
            t.add_row("Model Backend", s["model_backend"])
            t.add_row("Uptime", f"{h:02d}:{m:02d}:{sec:02d}")
            t.add_row("Heartbeat", f"{'● alive' if s['heartbeat']['alive'] else '○ stopped'} ({s['heartbeat']['beats']} beats)")
            t.add_row("Memory Backend", mem["semantic_backend"])
            t.add_row("Semantic Memories", str(mem["semantic_entries"]))
            t.add_row("Short-term Messages", str(mem["short_term_messages"]))
            t.add_row("Episodes Logged", str(mem["episodes"]["total_tasks"]))
            t.add_row("Reflections Stored", str(sm["reflections"]))
            t.add_row("Skills Learned", str(s["skills"]["count"]))
            t.add_row("Tasks Completed", f"{sm['tasks_completed']} ✓  {sm['tasks_failed']} ✗")
            t.add_row("Tools Available", str(len(s["tools"])))
            _console.print(t)
        else:
            print(f"Model: {s['model_backend']} | Uptime: {h:02d}:{m:02d}:{sec:02d}")
            print(f"Skills: {s['skills']['count']} | Memories: {mem['semantic_entries']} | Tasks: {sm['tasks_completed']}✓ {sm['tasks_failed']}✗")

    def _cmd_skills(self, domain_filter: str):
        skills = self._skills.list_all(domain=domain_filter or None)
        if not skills:
            _p("No skills learned yet. Use [bold]/learn <topic>[/bold] to start." if HAS_RICH
               else "No skills yet. Use /learn <topic>")
            return
        if HAS_RICH:
            t = Table(title=f"Learned Skills ({len(skills)})", header_style="bold cyan")
            t.add_column("Confidence"); t.add_column("Name"); t.add_column("Domain"); t.add_column("Used"); t.add_column("Description")
            for s in skills:
                conf = s.get("confidence", 0.5)
                c_str = f"{conf:.0%}"
                c_style = "green" if conf >= 0.7 else "yellow" if conf >= 0.5 else "red"
                t.add_row(Text(c_str, style=c_style), s["name"], s["domain"],
                          str(s.get("use_count", 0)), s["description"][:55])
            _console.print(t)
        else:
            for s in skills:
                print(f"  [{s.get('confidence', 0.5):.0%}] [{s['domain']}] {s['name']} (x{s.get('use_count',0)})")

    def _cmd_learn(self, topic: str):
        if not topic:
            _p("Usage: /learn <topic>  e.g. /learn Blender Python scripting")
            return
        if HAS_RICH:
            with Progress(SpinnerColumn(), TextColumn(f"[cyan]Researching '{topic}'...[/cyan]"),
                          console=_console, transient=True) as prog:
                prog.add_task("", total=None)
                result = self.learn(topic)
        else:
            print(f"Researching '{topic}'...")
            result = self.learn(topic)
        _p(result)

    def _cmd_skill(self, name: str):
        if not name:
            _p("Usage: /skill <name>")
            return
        results = self._skills.search(name)
        if not results:
            _p(f"No skill found for '{name}'.")
            return
        s = results[0]
        conf = s.get("confidence", 0.5)
        issues = s.get("validation_issues", [])
        content = (
            f"**{s['name']}**  v{s.get('version',1)}  [{s['domain']}]\n"
            f"Confidence: {conf:.0%} | Used: {s.get('use_count',0)}x\n"
        )
        if issues:
            content += f"Validation issues: {'; '.join(issues)}\n"
        content += f"\n## Procedure\n{s['procedure']}\n\n## Code Template\n```\n{s.get('code_template','N/A')}\n```"
        if HAS_RICH:
            _console.print(Markdown(content))
        else:
            print(content)

    def _cmd_memory(self, _):
        m = self._memory.stats()
        ep = m["episodes"]
        _p(f"Semantic memory: {m['semantic_entries']} entries ({m['semantic_backend']})")
        _p(f"Short-term:      {m['short_term_messages']} messages")
        _p(f"Episodes:        {ep['total_tasks']} tasks, {ep['reflections']} reflections")
        if ep.get("top_skills"):
            _p("Top skills used: " + ", ".join(f"{x['skill']}({x['count']})" for x in ep["top_skills"][:5]))

    def _cmd_episodes(self, _):
        eps = self._memory.episodic.recall(limit=12, include_reflections=False)
        if not eps:
            _p("No episodes yet.")
            return
        if HAS_RICH:
            t = Table(title="Recent Episodes", header_style="bold cyan")
            t.add_column("Time", style="dim"); t.add_column("Task"); t.add_column("Tools"); t.add_column("✓")
            for e in eps:
                ts = e["timestamp"][:16].replace("T", " ")
                t.add_row(ts, e["task"][:50], e["action"][:30], "✓" if e["success"] else "✗")
            _console.print(t)
        else:
            for e in eps:
                print(f"[{e['timestamp'][:16]}] {'✓' if e['success'] else '✗'} {e['task'][:60]}")

    def _cmd_profile(self, _):
        p = self._self_model.profile
        s = p["stats"]
        if HAS_RICH:
            t = Table(title="Agent Self-Model", header_style="bold cyan")
            t.add_column("Property"); t.add_column("Value")
            t.add_row("Tasks Completed", str(s["tasks_completed"]))
            t.add_row("Tasks Failed", str(s["tasks_failed"]))
            t.add_row("Reflections", str(s["reflections_stored"]))
            t.add_row("Skills Learned", str(s["skills_learned"]))
            t.add_row("Strengths", ", ".join(p["strengths"]) or "none yet")
            t.add_row("Weaknesses", ", ".join(p["weaknesses"]) or "none yet")
            _console.print(t)
            if p["lessons_learned"]:
                _console.print("\n[bold]Lessons Learned:[/bold]")
                for l in p["lessons_learned"][-8:]:
                    _console.print(f"  • {l}")
        else:
            print(f"Tasks: {s['tasks_completed']}✓ {s['tasks_failed']}✗ | Reflections: {s['reflections_stored']}")
            for l in p["lessons_learned"][-5:]:
                print(f"  • {l}")

    def _cmd_clear(self, _):
        self._memory.short.clear()
        _p("Short-term memory cleared.")

    def _cmd_verbose(self, _):
        self._verbose = not self._verbose
        _p(f"Verbose: {'ON — tool calls will be shown' if self._verbose else 'OFF'}")

    def _cmd_model(self, _):
        _p(f"Active model backend: [bold]{self._model.active_backend}[/bold]" if HAS_RICH
           else f"Active model: {self._model.active_backend}")

    def _print_banner(self):
        skill_count = len(self._skills.list_all())
        mem_count = self._memory.semantic.count()
        if HAS_RICH:
            _console.print(Panel(
                "[bold cyan]NexusAgent[/bold cyan]  [dim]v2.0.0[/dim]\n"
                "[dim]Local Self-Learning AI Agent[/dim]\n\n"
                f"[dim]  Model:    [white]{self._model.active_backend}[/white][/dim]\n"
                f"[dim]  Skills:   [white]{skill_count} learned[/white][/dim]\n"
                f"[dim]  Memories: [white]{mem_count} stored[/white][/dim]\n"
                f"[dim]  Heartbeat: [white]{'active' if self._heartbeat.is_alive() else 'off'}[/white][/dim]",
                border_style="cyan",
                expand=False,
            ))
        else:
            print("=" * 52)
            print("  NexusAgent v2.0.0 — Local Self-Learning AI")
            print(f"  Model: {self._model.active_backend} | Skills: {skill_count} | Memories: {mem_count}")
            print("=" * 52)

    def shutdown(self):
        self._heartbeat.stop()
        self._memory.episodic.close()
