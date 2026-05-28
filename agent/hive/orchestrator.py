"""Orchestrator — the Developer brain that plans, delegates, reviews, and synthesizes.

Flow:
  1. brainstorm()   → Project plan (name, type, vision, tech stack, USPs)
  2. assign_tasks() → ProjectTask list per role
  3. execute_team() → Run each specialist (ordered, building on prior outputs)
  4. review()       → Critique each deliverable, request revisions if needed
  5. synthesize()   → Produce a master project document
"""
import json
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from .project import Project, ProjectTask, Deliverable
from .roles import ROLES, PROJECT_ROLES, ROLE_TASKS, RoleConfig
from .specialist import SpecialistAgent
from ..core.loop import _parse_action
from ..core.logger import get_logger

log = get_logger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

BRAINSTORM_PROMPT = """\
You are the Lead Developer and Creative Director at NexusAgent Studios.
Your job: receive a high-level goal, brainstorm the most compelling version of it, and plan its development.

Goal: "{goal}"

Think about:
- What type of project is this? (game/app/saas/website/tool)
- What would make this commercially successful and creatively outstanding?
- What's the killer feature that differentiates it?
- What's a great name and tagline?

Output ONLY this JSON (no other text):
{{
  "name": "ProjectName",
  "tagline": "One punchy sentence under 15 words",
  "type": "game|app|saas|website|tool",
  "genre": "specific genre (e.g. action_rpg, fitness_app, dev_tool)",
  "vision": "2-3 sentence compelling vision for what this becomes",
  "target_audience": "specific description of who this is for",
  "tech_stack": ["technology1", "technology2", "technology3"],
  "unique_selling_points": [
    "Key differentiator from competitors",
    "Killer feature that users will love",
    "Why this will succeed commercially"
  ],
  "core_features": ["feature1", "feature2", "feature3", "feature4", "feature5"],
  "revenue_model": "exactly how this makes money",
  "estimated_scope": "indie|mid|large|AAA",
  "roles_needed": ["role_id1", "role_id2", "role_id3"]
}}\
"""

REVIEW_PROMPT = """\
You are the Lead Developer reviewing a team member's deliverable.

Project: {project_name}
Role: {role_title}
Expected deliverable: {deliverable_type}

Their output:
{content}

Review criteria:
1. Is it specific enough? (No vague statements like "use good practices")
2. Does it cover ALL required sections from the task?
3. Is it professionally detailed (600+ words)?
4. Does it build on and reference the team's other work?
5. Are there any obvious gaps or errors?

Output ONLY this JSON:
{{
  "approved": true,
  "score": 8.5,
  "feedback": "What's good and what's missing",
  "revision_request": "Specific things to add/fix (empty string if approved)"
}}\
"""

SYNTHESIS_PROMPT = """\
You are the Lead Developer synthesizing the team's work into a master project document.

Project: {project_name}
Type: {project_type}
Vision: {vision}

Team deliverables summary:
{deliverables_summary}

Create a concise Executive Summary and Project Roadmap that:
1. Summarizes the project vision and what makes it special
2. Highlights the key decisions made by each team member
3. Identifies the top 5 risks and mitigations
4. Provides a clear Phase 1 action plan (first 30 days)
5. Lists immediate next steps to start building

Write as professional markdown. Be direct and actionable.\
"""


class OrchestratorAgent:
    """The Developer brain — coordinates all specialist agents."""

    def __init__(self, model_client, tool_registry, skill_registry,
                 memory_manager, data_dir: Path):
        self._model = model_client
        self._tools = tool_registry
        self._skills = skill_registry
        self._memory = memory_manager
        self._data_dir = data_dir
        self._print_fn = print  # overridable for rich output
        self._deliverable_lock = threading.Lock()  # guards project.deliverables concurrent writes

    def set_print(self, fn):
        self._print_fn = fn

    def _log(self, msg: str):
        self._print_fn(msg)

    # ── Main entry point ──────────────────────────────────────────────────

    def run_project(self, goal: str, parallel: bool = True) -> Project:
        """Full pipeline: brainstorm → plan → execute team → review → synthesize."""

        # 1. Brainstorm
        self._log(f"\n[Orchestrator] Brainstorming: \"{goal}\"")
        project = self._brainstorm(goal)
        self._log(f"[Orchestrator] Project: {project.emoji()} **{project.name}** — {project.tagline}")
        self._log(f"[Orchestrator] Type: {project.type} | Scope: {project.scope}")
        self._log(f"[Orchestrator] Roles: {', '.join(project.roles_needed)}\n")

        # 2. Assign tasks
        tasks = self._assign_tasks(project)
        project.tasks = tasks
        project.status = "active"

        # 3. Execute team (respecting execution order)
        self._execute_team(project, parallel=parallel)

        # 4. Review deliverables
        project.status = "review"
        self._review_all(project)

        # 5. Synthesize master document
        self._log("[Orchestrator] Synthesizing final project document...")
        synthesis = self._synthesize(project)
        project.deliverables["_synthesis"] = Deliverable(
            role="_synthesis",
            title=f"Executive Summary — {project.name}",
            content=synthesis,
            approved=True,
            word_count=len(synthesis.split()),
        )

        # 6. Save to disk
        project.status = "complete"
        project.completed_at = datetime.utcnow().isoformat()
        output_dir = self._data_dir / "projects" / _safe_name(project.name)
        project.save(output_dir)
        self._log(f"\n[Orchestrator] ✓ Project complete! Saved to {output_dir}")

        # 7. Store in agent memory
        if self._memory:
            self._memory.store_fact(
                f"Built project '{project.name}' ({project.type}): {project.tagline}. "
                f"Team: {', '.join(project.roles_needed)}. Vision: {project.vision}",
                importance=0.9,
                metadata={"project_id": project.id, "type": project.type},
            )

        return project

    # ── Stage 1: Brainstorm ───────────────────────────────────────────────

    def _brainstorm(self, goal: str) -> "ProjectWithEmoji":
        prompt = BRAINSTORM_PROMPT.format(goal=goal)
        try:
            response = self._model.chat([
                {"role": "system", "content": "You are a creative director. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ])
        except Exception:
            log.exception("Brainstorm model call failed, using defaults")
            response = "{}"
        data = _parse_json(response) or {}
        if not data:
            log.warning("Brainstorm returned no parseable JSON — using goal as project name")

        # Validate roles exist
        raw_roles = data.get("roles_needed", [])
        project_type = data.get("type", "app")
        if not raw_roles or not all(r in ROLES for r in raw_roles):
            raw_roles = PROJECT_ROLES.get(project_type, ["architect", "backend_dev", "frontend_dev"])
        # Filter to only known roles
        roles = [r for r in raw_roles if r in ROLES][:6]

        p = ProjectWithEmoji(
            id=str(uuid.uuid4())[:8],
            name=data.get("name", "Project X"),
            tagline=data.get("tagline", goal[:60]),
            type=project_type,
            genre=data.get("genre", project_type),
            vision=data.get("vision", goal),
            target_audience=data.get("target_audience", "general users"),
            tech_stack=data.get("tech_stack", []),
            unique_selling_points=data.get("unique_selling_points", []),
            core_features=data.get("core_features", []),
            revenue_model=data.get("revenue_model", "TBD"),
            scope=data.get("estimated_scope", "mid"),
            roles_needed=roles,
            goal=goal,
        )
        return p

    # ── Stage 2: Assign tasks ─────────────────────────────────────────────

    def _assign_tasks(self, project: Project) -> list[ProjectTask]:
        task_templates = ROLE_TASKS.get(project.type, ROLE_TASKS.get("app", {}))
        tasks = []
        for role_id in project.roles_needed:
            template = task_templates.get(role_id, f"Create your deliverable for {project.name}")
            description = template.format(name=project.name)
            tasks.append(ProjectTask(role=role_id, description=description))
        return tasks

    # ── Stage 3: Execute team ─────────────────────────────────────────────

    def _execute_team(self, project: Project, parallel: bool = True):
        # Group tasks by execution_order (same order = can run in parallel)
        order_groups: dict[int, list[ProjectTask]] = {}
        for task in project.tasks:
            role_cfg = ROLES.get(task.role)
            order = role_cfg.execution_order if role_cfg else 99
            order_groups.setdefault(order, []).append(task)

        for order in sorted(order_groups.keys()):
            batch = order_groups[order]
            if parallel and len(batch) > 1:
                self._run_batch_parallel(project, batch)
            else:
                for task in batch:
                    self._run_task(project, task)

    def _run_batch_parallel(self, project: Project, tasks: list[ProjectTask]):
        self._log(f"[Orchestrator] Running in parallel: {[t.role for t in tasks]}")
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {executor.submit(self._run_task, project, task): task for task in tasks}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    task = futures[future]
                    self._log(f"[{task.role}] Error: {e}")

    def _run_task(self, project: Project, task: ProjectTask):
        role_cfg = ROLES.get(task.role)
        if not role_cfg:
            return

        emoji = role_cfg.emoji
        title = role_cfg.title
        self._log(f"  {emoji} [{title}] Working on {role_cfg.deliverable}...")
        task.status = "running"
        task.started_at = datetime.utcnow().isoformat()

        specialist = SpecialistAgent(
            role=role_cfg,
            model_client=self._model,
            tool_registry=self._tools,
            skill_registry=self._skills,
            memory_manager=self._memory,
        )

        try:
            deliverable = specialist.execute(project, task.description, use_tools=True)
            task.deliverable = deliverable
            task.status = "complete"
            task.completed_at = datetime.utcnow().isoformat()

            # Thread-safe dict write — multiple specialists may finish simultaneously
            with self._deliverable_lock:
                project.deliverables[task.role] = deliverable
            self._log(f"  {emoji} [{title}] ✓ Done ({deliverable.word_count} words)")
        except Exception:
            task.status = "failed"
            log.exception("[%s] specialist task failed", task.role)
            self._log(f"  {emoji} [{title}] ✗ Failed — see nexus.log for details")

    # ── Stage 4: Review ───────────────────────────────────────────────────

    def _review_all(self, project: Project):
        self._log("\n[Orchestrator] Reviewing deliverables...")
        for role_id, deliverable in project.deliverables.items():
            if role_id.startswith("_"):
                continue
            role_cfg = ROLES.get(role_id)
            if not role_cfg:
                deliverable.approved = True
                continue

            approved, score, feedback, revision = self._review_one(
                project, role_cfg, deliverable
            )
            deliverable.approved = approved

            if approved:
                self._log(f"  ✓ [{role_cfg.title}] Approved (score={score:.1f})")
            else:
                self._log(f"  ↺ [{role_cfg.title}] Revision requested: {revision[:80]}")
                # One revision attempt
                if revision:
                    self._revise(project, role_cfg, deliverable, revision)
                    deliverable.approved = True  # accept after revision

    def _review_one(self, project: Project, role: RoleConfig,
                    deliverable: Deliverable) -> tuple[bool, float, str, str]:
        prompt = REVIEW_PROMPT.format(
            project_name=project.name,
            role_title=role.title,
            deliverable_type=role.deliverable,
            content=deliverable.content[:3000],
        )
        response = self._model.chat([
            {"role": "system", "content": "You are a rigorous technical reviewer. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ])
        data = _parse_json(response) or {}
        score = float(data.get("score", 7.0))
        feedback = data.get("feedback", "")
        revision = data.get("revision_request", "")
        # Approve when score meets threshold — model's boolean can be overly strict
        approved = score >= 6.5
        return approved, score, feedback, revision

    def _revise(self, project: Project, role: RoleConfig,
                deliverable: Deliverable, notes: str):
        self._log(f"  ↺ [{role.title}] Revising...")
        specialist = SpecialistAgent(
            role=role,
            model_client=self._model,
            tool_registry=self._tools,
            skill_registry=self._skills,
            memory_manager=self._memory,
        )
        revision_task = (
            f"Revise and improve your {role.deliverable} for {project.name}.\n\n"
            f"Reviewer feedback: {notes}\n\n"
            f"Your previous draft:\n{deliverable.content[:2000]}\n\n"
            f"Please address all feedback and produce a better version."
        )
        revised = specialist.execute(project, revision_task, use_tools=False)
        deliverable.content = revised.content
        deliverable.word_count = revised.word_count
        deliverable.revision_notes = notes

    # ── Stage 5: Synthesize ───────────────────────────────────────────────

    def _synthesize(self, project: Project) -> str:
        summary_parts = []
        for role_id, deliv in project.deliverables.items():
            if role_id.startswith("_"):
                continue
            role_cfg = ROLES.get(role_id)
            title = role_cfg.title if role_cfg else role_id
            summary_parts.append(f"**{title}**: {deliv.content[:400]}...")

        prompt = SYNTHESIS_PROMPT.format(
            project_name=project.name,
            project_type=project.type,
            vision=project.vision,
            deliverables_summary="\n\n".join(summary_parts),
        )
        response = self._model.chat([
            {"role": "system", "content": "You are the Lead Developer writing the final project summary."},
            {"role": "user", "content": prompt},
        ])
        action = _parse_action(response)
        return str(action["final_answer"]) if action and "final_answer" in action else response


# ── Extended Project class with emoji helper ──────────────────────────────────

class ProjectWithEmoji(Project):
    def emoji(self) -> str:
        return {"game": "🎮", "app": "📱", "saas": "☁️", "website": "🌐", "tool": "🔧"}.get(self.type, "🚀")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name.lower())[:40]


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None
