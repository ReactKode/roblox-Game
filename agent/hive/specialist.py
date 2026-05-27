"""Specialist agent — a focused worker with a role-specific identity and system prompt."""
from datetime import datetime

from .roles import RoleConfig
from .project import Project, Deliverable
from ..core.loop import _parse_action

SPECIALIST_SYSTEM = """\
You are the **{title}** on the NexusAgent development team.

{role_description}

━━━ PROJECT ━━━
{project_summary}

━━━ WHAT YOUR TEAM HAS BUILT SO FAR ━━━
{team_context}

━━━ YOUR TASK ━━━
{task}

━━━ YOUR EXPERTISE ━━━
{expertise}

━━━ RELEVANT SKILLS FROM LIBRARY ━━━
{skills}

━━━ INSTRUCTIONS ━━━
{instructions}

Produce your complete **{deliverable}** now.
- Write in professional markdown
- Be specific and detailed — this goes directly into production planning
- Minimum 600 words
- Build on and complement your teammates' work above
- Think like the best {title} in the world working on the best version of this project\
"""

REACT_SYSTEM = """\
You are the **{title}** on the NexusAgent development team doing research for your deliverable.

Project: {project_name}
Your task: {task}

Use tools to research and gather information, then output your complete deliverable.

Available tools: {tools}

RESPONSE FORMAT — output ONLY valid JSON:
  Use a tool:    {{"tool": "name", "args": {{"key": "value"}}}}
  When done:     {{"final_answer": "your complete markdown deliverable"}}
\
"""


class SpecialistAgent:
    """One specialist worker. Receives a task, produces a deliverable."""

    def __init__(self, role: RoleConfig, model_client, tool_registry,
                 skill_registry, memory_manager):
        self._role = role
        self._model = model_client
        self._tools = tool_registry
        self._skills = skill_registry
        self._memory = memory_manager

    @property
    def role_id(self) -> str:
        return self._role.id

    @property
    def title(self) -> str:
        return self._role.title

    def execute(self, project: Project, task_description: str,
                use_tools: bool = True) -> Deliverable:
        """Run this specialist on a task and return a Deliverable."""
        started = datetime.utcnow().isoformat()

        # Optionally do a research pass first (web search / skill recall)
        research_context = ""
        if use_tools:
            research_context = self._research_pass(project, task_description)

        # Main generation pass — deep, focused deliverable
        content = self._generate(project, task_description, research_context)

        # Store result in shared semantic memory so other agents can recall it
        if self._memory and content:
            self._memory.store_research(
                f"[{self._role.title}] {project.name}: {content[:600]}",
                source=f"hive/{project.id}/{self._role.id}",
            )

        word_count = len(content.split())
        return Deliverable(
            role=self._role.id,
            title=f"{self._role.deliverable} — {project.name}",
            content=content,
            word_count=word_count,
            approved=False,
            timestamp=started,
        )

    # ── private ───────────────────────────────────────────────────────────

    def _research_pass(self, project: Project, task: str) -> str:
        """Quick tool-use pass to gather domain-specific research."""
        relevant_skills = self._get_skills_context()
        if not relevant_skills or "None" in relevant_skills:
            return ""

        system = REACT_SYSTEM.format(
            title=self._role.title,
            project_name=project.name,
            task=task[:300],
            tools=", ".join(self._tools.names()),
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user",
             "content": (
                 f"Before writing your deliverable for {project.name}, do a quick research pass.\n"
                 f"Check recall_skill for any relevant skills ({', '.join(self._role.skill_domains)}), "
                 f"and if needed do 1-2 web searches for current best practices.\n"
                 f"Once you have enough context, output {{\"final_answer\": \"RESEARCH_COMPLETE: <key findings>\"}}."
             )},
        ]
        tools = self._tools.get_schemas()
        iterations = 0
        research_findings = ""

        while iterations < 4:
            iterations += 1
            response = self._model.chat(messages, tools)
            action = _parse_action(response)
            if action is None:
                break
            if "final_answer" in action:
                ans = str(action["final_answer"])
                if "RESEARCH_COMPLETE" in ans:
                    research_findings = ans.replace("RESEARCH_COMPLETE:", "").strip()
                break
            if "tool" in action:
                obs = self._tools.execute(action["tool"], action.get("args", {}))
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Tool result:\n{obs[:1200]}"})

        return research_findings

    def _generate(self, project: Project, task: str, research: str) -> str:
        """Main generation — produces the full deliverable."""
        skills_ctx = self._get_skills_context()
        team_ctx = project.team_context(exclude_role=self._role.id)
        extra = f"\n\n### Research Findings\n{research}" if research else ""

        system = SPECIALIST_SYSTEM.format(
            title=self._role.title,
            role_description=self._role.description,
            project_summary=project.summary(),
            team_context=team_ctx + extra,
            task=task,
            expertise=", ".join(self._role.expertise),
            skills=skills_ctx,
            instructions=self._role.instructions,
            deliverable=self._role.deliverable,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user",
             "content": (
                 f"Write your complete {self._role.deliverable} for **{project.name}** now. "
                 f"Cover everything in the task. Be thorough, specific, and professional."
             )},
        ]
        response = self._model.chat(messages)

        # If model returned JSON (some models do this), extract the text
        action = _parse_action(response)
        if action and "final_answer" in action:
            return str(action["final_answer"])

        return response

    def _get_skills_context(self) -> str:
        skills = []
        for domain in self._role.skill_domains:
            found = self._skills.search(domain, domain)
            skills.extend(found[:2])
        if not skills:
            return "None in library yet — consider using learn_skill"
        return "\n".join(
            f"  [{s.get('confidence',0.5):.0%}] {s['name']} — {s['description'][:60]}"
            for s in skills[:4]
        )
