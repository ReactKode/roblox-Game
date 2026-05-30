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
from .subagent import SubAgent
from agent.hive.orchestrator import OrchestratorAgent
from agent.hive.roles import ROLES
from agent.hive.asset_pipeline import AssetPipelineAgent, ENGINE_SETUP
from agent.hive.world_builder import WorldBuilderAgent
from agent.scheduler import TaskScheduler
from agent.sandbox.backends import SandboxManager
from agent.gateway.manager import GatewayManager

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

        # ── Hive ──────────────────────────────────────────────────────────
        self._hive = OrchestratorAgent(
            model_client=self._model,
            tool_registry=self._tools,
            skill_registry=self._skills,
            memory_manager=self._memory,
            data_dir=data_dir,
        )
        self._hive.set_print(self._hive_log)

        # ── Asset pipeline ────────────────────────────────────────────────
        self._assets = AssetPipelineAgent(self._model, data_dir)
        self._assets.set_print(self._hive_log)

        # ── World builder ─────────────────────────────────────────────────
        self._world_builder = WorldBuilderAgent(self._model, data_dir)
        self._world_builder.set_print(self._hive_log)

        # ── Subagent delegation ───────────────────────────────────────────
        self._subagent = SubAgent(
            model_client=self._model,
            tool_registry=self._tools,
            memory_manager=self._memory,
            skill_registry=self._skills,
            reflector=self._reflector,
            self_model=self._self_model,
            max_iterations=cfg.get("agent", {}).get("max_iterations", 12),
        )

        # ── Sandbox ───────────────────────────────────────────────────────
        self._sandbox = SandboxManager(cfg.get("sandbox", {}))

        # ── Scheduler ────────────────────────────────────────────────────
        sched_cfg = cfg.get("scheduler", {})
        self._scheduler = TaskScheduler(agent_fn=self.chat)
        if not sched_cfg.get("enabled", True) and self._scheduler.available:
            self._scheduler.shutdown()

        # ── Gateways ─────────────────────────────────────────────────────
        self._gateway_manager = GatewayManager(agent_fn=self.chat)
        if cfg.get("gateway"):
            self._register_gateways(cfg["gateway"])

        # ── Extra tools (scheduler / subagent / sandbox) ──────────────────
        self._register_extra_tools()

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

    def run_project(self, goal: str) -> "Project":
        """Run the full Hive pipeline for a project goal."""
        return self._hive.run_project(goal, parallel=False)

    def create_assets(self, request: str, engines: list[str] = None) -> list:
        """Run the asset pipeline: Blender → engine importers."""
        pkgs = self._assets.run(request, engines or ["unreal", "unity", "godot"])
        if self._memory:
            names = ", ".join(p.asset_name for p in pkgs)
            self._memory.store_fact(
                f"Created 3D assets: {names} for {request}. Exported for {engines}.",
                importance=0.8,
                metadata={"asset_pipeline": True},
            )
        return pkgs

    def build_world(self, description: str, style: str = "low poly stylized",
                    size: float = 100.0, engines: list[str] = None):
        """Build a complete 3D world: plan → create each asset → assemble world scene."""
        world = self._world_builder.build(
            description=description,
            style=style,
            size=size,
            engines=engines or ["unreal", "unity", "godot"],
        )
        if self._memory:
            self._memory.store_fact(
                f"Built 3D world '{world.name}': {len(world.assets)} asset types, "
                f"{sum(a.count for a in world.assets)} total objects. Description: {description}",
                importance=0.9,
                metadata={"world_builder": True, "world_name": world.name},
            )
        return world

    def _hive_log(self, msg: str):
        if HAS_RICH and _console:
            # Style different log lines
            if "[Orchestrator]" in msg:
                _console.print(f"[bold magenta]{msg}[/bold magenta]")
            elif "✓" in msg:
                _console.print(f"[green]{msg}[/green]")
            elif "✗" in msg or "Error" in msg:
                _console.print(f"[red]{msg}[/red]")
            elif "↺" in msg:
                _console.print(f"[yellow]{msg}[/yellow]")
            else:
                _console.print(f"[cyan]{msg}[/cyan]")
        else:
            print(msg)

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
            "scheduler": {
                "available": self._scheduler.available,
                "tasks": len(self._scheduler.list_tasks()),
            },
            "sandbox": {
                "backend": self._sandbox.backend_name,
            },
            "gateways": {
                "active": self._gateway_manager.active_gateways(),
            },
        }

    def run_cli(self):
        self._print_banner()
        _p("Type [bold]/help[/bold] for commands, or just start talking.\n" if HAS_RICH
           else "Type /help for commands. /quit to exit.\n")

        while True:
            try:
                user_input = input("\n[you] " if not HAS_RICH else "")
                if HAS_RICH:
                    _console.print("[bold white]you>[/bold white] ", end="")
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
            _console.print("\n[bold cyan]hermes>[/bold cyan]")
            _console.print(Markdown(response))
        else:
            print("hermes> thinking...")
            response = self.chat(message)
            print(f"\nhermes> {response}")

    def _handle_command(self, raw: str):
        parts = raw.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        commands = {
            "/quit": self._cmd_quit, "/exit": self._cmd_quit, "/q": self._cmd_quit,
            "/help": self._cmd_help,
            "/status": self._cmd_status,
            "/project": self._cmd_project,
            "/projects": self._cmd_projects,
            "/asset": self._cmd_asset,
            "/assets": self._cmd_assets,
            "/world": self._cmd_world,
            "/worlds": self._cmd_worlds,
            "/skills": self._cmd_skills,
            "/learn": self._cmd_learn,
            "/skill": self._cmd_skill,
            "/memory": self._cmd_memory,
            "/episodes": self._cmd_episodes,
            "/profile": self._cmd_profile,
            "/clear": self._cmd_clear,
            "/verbose": self._cmd_verbose,
            "/model": self._cmd_model,
            "/schedule": self._cmd_schedule,
            "/schedules": self._cmd_schedules,
            "/gateway": self._cmd_gateway,
            "/gateways": self._cmd_gateways,
            "/sandbox": self._cmd_sandbox,
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
**Hermes — Commands**

**🧠 Solo Agent**
| Command | Description |
|---|---|
| `/learn <topic>` | Research & learn a skill (Blender, Unity, Godot, Python...) |
| `/skills` | List all learned skills with confidence scores |
| `/skill <name>` | Show full skill details and code template |
| `/profile` | Self-model: strengths, weaknesses, lessons learned |
| `/memory` | Memory statistics |
| `/episodes` | Recent action history |

**🐝 Hive (Multi-Agent Team)**
| Command | Description |
|---|---|
| `/project <goal>` | Spawn a full specialist team to build your project |
| `/projects` | List all completed projects |

**🎨 Asset Pipeline (Blender → Engines)**
| Command | Description |
|---|---|
| `/asset <description>` | Create 3D asset in Blender, export to all 3 engines |
| `/asset <description> --engines unreal,godot` | Target specific engines |
| `/assets` | List all created assets |

**🌍 World Builder (Complete Game Worlds)**
| Command | Description |
|---|---|
| `/world <description>` | Plan + build a full 3D world scene (terrain, buildings, trees, props) |
| `/world <description> --style realistic --size 200` | Custom style and size in metres |
| `/worlds` | List all built worlds |

**⏰ Scheduler (Automated Recurring Tasks)**
| Command | Description |
|---|---|
| `/schedule <when> -- <task>` | Schedule a recurring task in natural language |
| `/schedules` | List all active scheduled tasks |

**🔌 Gateways (Multi-Platform Messaging)**
| Command | Description |
|---|---|
| `/gateway <platform>` | Start a gateway: telegram, discord, slack, email, cli |
| `/gateways` | List active gateways |

**📦 Sandbox (Isolated Code Execution)**
| Command | Description |
|---|---|
| `/sandbox <code>` | Run Python code in the sandbox |

**⚙️ System**
| Command | Description |
|---|---|
| `/status` | Full agent status |
| `/clear` | Clear short-term memory |
| `/verbose` | Toggle verbose tool-call logging |
| `/model` | Show active model backend |
| `/quit` | Exit |

**Solo examples:**
- `How do I rig a character in Blender with Python?`
- `/learn Godot 4 GDScript game mechanics`
- `/schedule every 30 minutes -- search for new Godot 4 tutorials`

**Hive examples:**
- `/project build an AAA souls-like action RPG`
- `/project build a mobile fitness tracking app`

**Gateway examples:**
- `/gateway telegram`   (requires token in config.yaml)
- `/gateway cli`        (launches a gateway-mode terminal session)
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

    def _cmd_project(self, goal: str):
        if not goal:
            _p("Usage: /project <goal>\n"
               "Examples:\n"
               "  /project build an AAA open-world action RPG\n"
               "  /project build a mobile fitness tracking app\n"
               "  /project create a SaaS project management tool\n"
               "  /project build a developer CLI tool for database migrations")
            return
        if HAS_RICH:
            _console.rule("[bold magenta]NexusAgent Hive[/bold magenta]")
            _console.print(f"[bold]Goal:[/bold] {goal}\n")
        else:
            print(f"\n{'='*60}\nNexusAgent Hive — {goal}\n{'='*60}")
        try:
            project = self.run_project(goal)
            output_dir = self._data_dir / "projects" / project.name.lower().replace(" ", "_")[:40]
            if HAS_RICH:
                _console.rule("[bold green]Project Complete[/bold green]")
                t = Table(title=f"🎯 {project.name}", header_style="bold cyan")
                t.add_column("Role"); t.add_column("Deliverable"); t.add_column("Words"); t.add_column("Status")
                for role_id, deliv in project.deliverables.items():
                    if role_id.startswith("_"):
                        continue
                    role_cfg = ROLES.get(role_id)
                    emoji = role_cfg.emoji if role_cfg else "•"
                    t.add_row(f"{emoji} {role_cfg.title if role_cfg else role_id}",
                              deliv.title[:45], str(deliv.word_count),
                              "[green]✓[/green]" if deliv.approved else "[yellow]~[/yellow]")
                _console.print(t)
                _console.print(f"\n[dim]Saved to: {output_dir}[/dim]")
            else:
                print(f"\nProject '{project.name}' complete!")
                for role_id, deliv in project.deliverables.items():
                    if not role_id.startswith("_"):
                        print(f"  ✓ {deliv.title} ({deliv.word_count} words)")
                print(f"Saved to: {output_dir}")
        except Exception as e:
            _p(f"[red]Project failed: {e}[/red]" if HAS_RICH else f"Project failed: {e}")

    def _cmd_projects(self, _):
        projects_dir = self._data_dir / "projects"
        if not projects_dir.exists():
            _p("No projects built yet. Use /project <goal> to start one.")
            return
        projects = [p for p in projects_dir.iterdir() if p.is_dir()]
        if not projects:
            _p("No projects found.")
            return
        import json
        if HAS_RICH:
            t = Table(title=f"Projects ({len(projects)})", header_style="bold cyan")
            t.add_column("Name"); t.add_column("Type"); t.add_column("Files")
            for p in sorted(projects):
                manifest = p / "manifest.json"
                if manifest.exists():
                    try:
                        data = json.loads(manifest.read_text())
                        files = len(list(p.glob("*.md")))
                        t.add_row(data.get("name", p.name), data.get("type", "?"), str(files) + " docs")
                    except Exception:
                        t.add_row(p.name, "?", "?")
            _console.print(t)
        else:
            for p in sorted(projects):
                print(f"  {p.name}/")

    def _cmd_asset(self, arg: str):
        """Usage: /asset <description> [--engines unreal,unity,godot]"""
        if not arg:
            _p(
                "Usage: /asset <description> [--engines unreal,unity,godot]\n\n"
                "Examples:\n"
                "  /asset a medieval sword for a dark fantasy RPG\n"
                "  /asset a low-poly treasure chest --engines godot,unity\n"
                "  /asset a sci-fi spaceship hull --engines unreal\n"
                "  /asset a wooden barrel prop --engines unity,godot\n\n"
                "Engines: unreal | unity | godot  (default: all three)"
            )
            return

        # Parse --engines flag
        engines = ["unreal", "unity", "godot"]
        request = arg
        if "--engines" in arg:
            parts = arg.split("--engines")
            request = parts[0].strip()
            engines_raw = [e.strip() for e in parts[1].split(",")]
            engines = engines_raw

        if HAS_RICH:
            _console.rule("[bold cyan]Asset Pipeline[/bold cyan]")
        try:
            packages = self.create_assets(request, engines)
            if HAS_RICH:
                _console.rule("[bold green]Pipeline Complete[/bold green]")
                t = Table(title=f"Assets Created ({len(packages)})", header_style="bold cyan")
                t.add_column("Asset"); t.add_column("Script"); t.add_column("FBX"); t.add_column("GLB"); t.add_column("Engines")
                for pkg in packages:
                    has_fbx = "✓" if pkg.fbx_path() and pkg.fbx_path().exists() else "script only"
                    has_glb = "✓" if pkg.glb_path() and pkg.glb_path().exists() else "script only"
                    eng_dirs = [e for e in engines if (pkg.asset_dir / e).exists()]
                    t.add_row(pkg.asset_name, "✓", has_fbx, has_glb, " ".join(eng_dirs))
                _console.print(t)
        except Exception as e:
            _p(f"[red]Asset pipeline error: {e}[/red]" if HAS_RICH else f"Error: {e}")

    def _cmd_assets(self, _):
        assets_dir = self._data_dir / "assets"
        if not assets_dir.exists():
            _p("No assets created yet. Use /asset <description> to create one.")
            return
        asset_list = [p for p in assets_dir.iterdir() if p.is_dir()]
        if not asset_list:
            _p("No assets found.")
            return
        if HAS_RICH:
            t = Table(title=f"Asset Library ({len(asset_list)})", header_style="bold cyan")
            t.add_column("Name"); t.add_column("Engines"); t.add_column("Exported")
            for a in sorted(asset_list):
                engines = [e for e in ["unreal", "unity", "godot"] if (a / e).exists()]
                has_fbx = "fbx" if (a / "exports" / f"{a.name}.fbx").exists() else ""
                has_glb = "glb" if (a / "exports" / f"{a.name}.glb").exists() else ""
                t.add_row(a.name, " ".join(engines) or "—",
                          ", ".join(filter(None, [has_fbx, has_glb])) or "script only")
            _console.print(t)
        else:
            for a in sorted(asset_list):
                print(f"  {a.name}/")

    def _cmd_world(self, arg: str):
        """Usage: /world <description> [--style <style>] [--size <metres>] [--engines unreal,unity,godot]"""
        if not arg:
            _p(
                "Usage: /world <description> [--style <style>] [--size <metres>] [--engines e1,e2]\n\n"
                "Examples:\n"
                "  /world a medieval village with a castle and marketplace\n"
                "  /world a dark forest with ancient ruins --style dark fantasy\n"
                "  /world a sci-fi space station interior --size 50\n"
                "  /world a tropical island with beaches and jungle --engines godot,unity\n\n"
                "Defaults: style='low poly stylized', size=100m, engines=all three"
            )
            return

        # Parse optional flags
        style = "low poly stylized"
        size = 100.0
        engines = ["unreal", "unity", "godot"]
        description = arg

        if "--style" in arg:
            parts = arg.split("--style", 1)
            description = parts[0].strip()
            remainder = parts[1].strip()
            if "--size" in remainder:
                s, remainder = remainder.split("--size", 1)
                style = s.strip()
                remainder = remainder.strip()
            elif "--engines" in remainder:
                s, remainder = remainder.split("--engines", 1)
                style = s.strip()
                remainder = remainder.strip()
                engines = [e.strip() for e in remainder.split(",")]
                remainder = ""
            else:
                style = remainder.strip()
                remainder = ""

        if "--size" in description:
            parts = description.split("--size", 1)
            description = parts[0].strip()
            rest = parts[1].strip()
            try:
                size = float(rest.split()[0])
            except (ValueError, IndexError):
                pass

        if "--engines" in description:
            parts = description.split("--engines", 1)
            description = parts[0].strip()
            engines = [e.strip() for e in parts[1].split(",")]

        if HAS_RICH:
            _console.rule("[bold cyan]World Builder[/bold cyan]")
            _console.print(f"[bold]World:[/bold] {description}")
            _console.print(f"[dim]Style: {style} | Size: {size}m × {size}m | Engines: {', '.join(engines)}[/dim]\n")
        else:
            print(f"\n{'='*60}\nWorld Builder — {description}\n{'='*60}")

        try:
            world = self.build_world(description, style=style, size=size, engines=engines)
            if HAS_RICH:
                _console.rule("[bold green]World Complete[/bold green]")
                t = Table(title=f"🌍 {world.name}", header_style="bold cyan")
                t.add_column("Asset"); t.add_column("Type"); t.add_column("Count"); t.add_column("Files")
                for pa in world.assets:
                    files = len(pa.package.files) if pa.package else 0
                    t.add_row(pa.name, pa.asset_type, str(pa.count),
                              f"{files} exported" if files else "script only")
                _console.print(t)
                _console.print(f"\n[dim]Saved to: {world.world_dir}[/dim]")
                if world.assembly_script:
                    _console.print(
                        f"[dim]Assembly: blender --background --python {world.assembly_script}[/dim]"
                    )
            else:
                print(f"\nWorld '{world.name}' complete!")
                for pa in world.assets:
                    print(f"  • {pa.name} × {pa.count} [{pa.asset_type}]")
        except Exception as e:
            _p(f"[red]World builder error: {e}[/red]" if HAS_RICH else f"Error: {e}")

    def _cmd_worlds(self, _):
        worlds_dir = self._data_dir / "worlds"
        if not worlds_dir.exists():
            _p("No worlds built yet. Use /world <description> to build one.")
            return
        world_list = [w for w in worlds_dir.iterdir() if w.is_dir()]
        if not world_list:
            _p("No worlds found.")
            return
        import json as _json
        if HAS_RICH:
            t = Table(title=f"Built Worlds ({len(world_list)})", header_style="bold cyan")
            t.add_column("Name"); t.add_column("Style"); t.add_column("Assets"); t.add_column("Exported")
            for w in sorted(world_list):
                manifest = w / "manifest.json"
                if manifest.exists():
                    try:
                        data = _json.loads(manifest.read_text())
                        n_assets = len(data.get("assets", []))
                        executed = "✓ FBX+GLB" if data.get("blender_executed") else "scripts only"
                        t.add_row(data.get("name", w.name), data.get("style", "?"),
                                  str(n_assets), executed)
                    except Exception:
                        t.add_row(w.name, "?", "?", "?")
            _console.print(t)
        else:
            for w in sorted(world_list):
                print(f"  {w.name}/")

    def _cmd_clear(self, _):
        self._memory.short.clear()
        _p("Short-term memory cleared.")

    def _cmd_verbose(self, _):
        self._verbose = not self._verbose
        _p(f"Verbose: {'ON — tool calls will be shown' if self._verbose else 'OFF'}")

    def _cmd_model(self, _):
        _p(f"Active model backend: [bold]{self._model.active_backend}[/bold]" if HAS_RICH
           else f"Active model: {self._model.active_backend}")

    def _cmd_schedule(self, arg: str):
        """Usage: /schedule <when> -- <task>"""
        if not arg or "--" not in arg:
            _p(
                "Usage: /schedule <when> -- <task>\n\n"
                "Examples:\n"
                "  /schedule every 30 minutes -- check the latest Blender news and summarize\n"
                "  /schedule daily at 09:00 -- send me a morning briefing\n"
                "  /schedule every Monday at 08:00 -- list this week's dev priorities\n\n"
                "Schedule expressions:\n"
                "  'every N minutes/hours/days', 'daily at HH:MM', 'every Monday at HH:MM'\n"
                "  'hourly', 'daily', 'weekly'"
            )
            return
        parts = arg.split("--", 1)
        schedule = parts[0].strip()
        task = parts[1].strip()
        if not schedule or not task:
            _p("Usage: /schedule <when> -- <task>")
            return
        result = self._do_schedule(schedule, task)
        _p(f"[green]{result}[/green]" if HAS_RICH else result)

    def _cmd_schedules(self, _):
        tasks = self._scheduler.list_tasks()
        if not tasks:
            _p("No scheduled tasks. Use /schedule <when> -- <task> to add one.")
            return
        if HAS_RICH:
            from rich.table import Table
            t = Table(title=f"Scheduled Tasks ({len(tasks)})", header_style="bold cyan")
            t.add_column("ID"); t.add_column("Schedule"); t.add_column("Runs"); t.add_column("Task")
            for task in tasks:
                t.add_row(task["id"], task["schedule"], str(task["run_count"]), task["task"][:55])
            _console.print(t)
        else:
            for task in tasks:
                print(f"  [{task['id']}] {task['schedule']} (×{task['run_count']}) — {task['task'][:60]}")

    def _cmd_gateway(self, platform: str):
        """Usage: /gateway <platform>  — start a gateway (telegram/discord/slack/email/cli)"""
        if not platform:
            _p(
                "Usage: /gateway <platform>\n\n"
                "Platforms: telegram | discord | slack | email | cli\n\n"
                "Configure tokens in config.yaml under 'gateway:' then run /gateway <name>.\n"
                "Active gateways: " + (", ".join(self._gateway_manager.active_gateways()) or "none")
            )
            return
        result = self.start_gateway(platform)
        _p(f"[cyan]{result}[/cyan]" if HAS_RICH else result)

    def _cmd_gateways(self, _):
        active = self._gateway_manager.active_gateways()
        if not active:
            _p("No active gateways. Use /gateway <platform> to start one.")
        else:
            _p(f"Active gateways: {', '.join(active)}")

    def _cmd_sandbox(self, arg: str):
        """Usage: /sandbox <python code>"""
        if not arg:
            _p(
                f"Usage: /sandbox <code>\n\n"
                f"Executes Python code in the sandbox (backend: {self._sandbox.backend_name}).\n\n"
                "Example: /sandbox print('hello from sandbox')"
            )
            return
        result = self._do_run_sandbox(arg, "python")
        _p(result)

    def _print_banner(self):
        skill_count = len(self._skills.list_all())
        mem_count = self._memory.semantic.count()
        if HAS_RICH:
            _console.print(Panel(
                "[bold cyan]Hermes[/bold cyan]  [dim]v2.0.0 — Local AI Agent[/dim]\n\n"
                f"[dim]  Model:     [white]{self._model.active_backend}[/white][/dim]\n"
                f"[dim]  Skills:    [white]{skill_count} learned[/white][/dim]\n"
                f"[dim]  Memories:  [white]{mem_count} stored[/white][/dim]\n"
                f"[dim]  Heartbeat: [white]{'active' if self._heartbeat.is_alive() else 'off'}[/white][/dim]",
                border_style="cyan",
                expand=False,
            ))
        else:
            print("=" * 52)
            print("  Hermes v2.0.0 — Local AI Agent")
            print(f"  Model: {self._model.active_backend} | Skills: {skill_count} | Memories: {mem_count}")
            print("=" * 52)

    def start_gateway(self, platform: str) -> str:
        """Start a named gateway (e.g. 'telegram', 'discord', 'slack', 'email', 'cli')."""
        from agent.gateway.cli_gateway import CLIGateway
        from agent.gateway.telegram import TelegramGateway
        from agent.gateway.discord_gw import DiscordGateway
        from agent.gateway.slack_gw import SlackGateway
        from agent.gateway.email_gw import EmailGateway

        gw_map = {
            "cli": lambda cfg: CLIGateway(),
            "telegram": lambda cfg: TelegramGateway(
                token=cfg.get("token", ""),
                allowed_users=cfg.get("allowed_users", []),
            ),
            "discord": lambda cfg: DiscordGateway(
                token=cfg.get("token", ""),
                allowed_guilds=cfg.get("allowed_guilds", []),
            ),
            "slack": lambda cfg: SlackGateway(
                bot_token=cfg.get("bot_token", ""),
                app_token=cfg.get("app_token", ""),
            ),
            "email": lambda cfg: EmailGateway(
                imap_host=cfg.get("imap_host", ""),
                imap_port=int(cfg.get("imap_port", 993)),
                smtp_host=cfg.get("smtp_host", ""),
                smtp_port=int(cfg.get("smtp_port", 465)),
                username=cfg.get("username", ""),
                password=cfg.get("password", ""),
                poll_interval=int(cfg.get("poll_interval", 60)),
                allowed_senders=cfg.get("allowed_senders"),
            ),
        }
        factory = gw_map.get(platform.lower())
        if not factory:
            return f"Unknown gateway '{platform}'. Choose: {', '.join(gw_map)}"
        gw = factory({})
        self._gateway_manager.register(gw)
        self._gateway_manager.start_all()
        return f"Gateway '{platform}' started."

    def shutdown(self):
        self._heartbeat.stop()
        self._scheduler.shutdown()
        self._subagent.shutdown()
        self._gateway_manager.stop_all()
        self._memory.episodic.close()

    # ── Gateway wiring ────────────────────────────────────────────────────

    def _register_gateways(self, gateway_cfg: dict):
        """Read config and start any enabled gateways as daemon threads."""
        from agent.gateway.telegram import TelegramGateway
        from agent.gateway.discord_gw import DiscordGateway
        from agent.gateway.slack_gw import SlackGateway
        from agent.gateway.email_gw import EmailGateway

        enabled = []

        tg = gateway_cfg.get("telegram", {})
        if tg.get("enabled") and tg.get("token"):
            from agent.gateway.telegram import TelegramGateway
            gw = TelegramGateway(token=tg["token"], allowed_users=tg.get("allowed_users", []))
            self._gateway_manager.register(gw)
            enabled.append("telegram")

        dc = gateway_cfg.get("discord", {})
        if dc.get("enabled") and dc.get("token"):
            gw = DiscordGateway(token=dc["token"], allowed_guilds=dc.get("allowed_guilds", []))
            self._gateway_manager.register(gw)
            enabled.append("discord")

        sl = gateway_cfg.get("slack", {})
        if sl.get("enabled") and sl.get("bot_token"):
            gw = SlackGateway(bot_token=sl["bot_token"], app_token=sl.get("app_token", ""))
            self._gateway_manager.register(gw)
            enabled.append("slack")

        em = gateway_cfg.get("email", {})
        if em.get("enabled") and em.get("imap_host"):
            gw = EmailGateway(
                imap_host=em["imap_host"],
                imap_port=int(em.get("imap_port", 993)),
                smtp_host=em.get("smtp_host", ""),
                smtp_port=int(em.get("smtp_port", 465)),
                username=em.get("username", ""),
                password=em.get("password", ""),
                poll_interval=int(em.get("poll_interval", 60)),
                allowed_senders=em.get("allowed_senders"),
            )
            self._gateway_manager.register(gw)
            enabled.append("email")

        if enabled:
            self._gateway_manager.start_all()
            _p(f"Gateways started: {', '.join(enabled)}")

    # ── Extra tools ───────────────────────────────────────────────────────

    def _register_extra_tools(self):
        """Register scheduler, subagent, and sandbox tools into the tool registry."""
        self._tools.add_tool(
            "schedule_task",
            lambda schedule, task: self._do_schedule(schedule, task),
            "Schedule a recurring task using natural language (e.g. 'every 5 minutes', 'daily at 09:00', 'every Monday at 08:00').",
            {"type": "object", "properties": {
                "schedule": {"type": "string", "description": "Natural language schedule expression"},
                "task": {"type": "string", "description": "Task or question to run on that schedule"},
            }, "required": ["schedule", "task"]},
        )
        self._tools.add_tool(
            "cancel_schedule",
            lambda task_id: self._do_cancel_schedule(task_id),
            "Cancel a previously scheduled task by its ID.",
            {"type": "object", "properties": {
                "task_id": {"type": "string", "description": "ID returned when the task was scheduled"},
            }, "required": ["task_id"]},
        )
        self._tools.add_tool(
            "list_schedules",
            lambda: self._do_list_schedules(),
            "List all currently scheduled tasks with their IDs and run counts.",
            {"type": "object", "properties": {}, "required": []},
        )
        self._tools.add_tool(
            "run_subagent",
            lambda task: self._subagent.run(task),
            "Delegate a complex task to an isolated subagent with its own conversation context.",
            {"type": "object", "properties": {
                "task": {"type": "string", "description": "Full description of the task for the subagent"},
            }, "required": ["task"]},
        )
        self._tools.add_tool(
            "run_in_sandbox",
            lambda code, language="python": self._do_run_sandbox(code, language),
            "Execute code in an isolated sandbox. Returns stdout/stderr.",
            {"type": "object", "properties": {
                "code": {"type": "string", "description": "Source code to execute"},
                "language": {"type": "string", "description": "Language: python/javascript/bash (default: python)"},
            }, "required": ["code"]},
        )

    def _do_schedule(self, schedule: str, task: str) -> str:
        task_id = self._scheduler.schedule(schedule, task)
        if len(task_id) == 8:
            return f"Scheduled (ID: {task_id}): '{task[:60]}' — {schedule}"
        return task_id

    def _do_cancel_schedule(self, task_id: str) -> str:
        return f"Cancelled schedule {task_id}." if self._scheduler.cancel(task_id) \
            else f"No schedule found with ID '{task_id}'."

    def _do_list_schedules(self) -> str:
        tasks = self._scheduler.list_tasks()
        if not tasks:
            return "No scheduled tasks."
        lines = [
            f"[{t['id']}] {t['schedule']} | runs={t['run_count']} | {t['task'][:60]}"
            for t in tasks
        ]
        return f"Scheduled tasks ({len(tasks)}):\n" + "\n".join(lines)

    def _do_run_sandbox(self, code: str, language: str = "python") -> str:
        result = self._sandbox.execute(code, language)
        status = "SUCCESS" if result["success"] else "FAILED"
        parts = [f"[{status}] (backend: {result['backend']})"]
        if result.get("stdout"):
            parts.append(f"stdout:\n{result['stdout']}")
        if result.get("stderr"):
            parts.append(f"stderr:\n{result['stderr']}")
        return "\n".join(parts)
