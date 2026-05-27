"""Skill executor - runs a skill by name."""


class SkillExecutor:
    def __init__(self, registry, tool_registry=None, model_client=None):
        self._registry = registry
        self._tools = tool_registry
        self._model = model_client

    def set_model(self, model_client):
        self._model = model_client

    def execute(self, skill_name: str, task: str = "") -> str:
        skill = self._registry.get(skill_name)
        if not skill:
            # Try fuzzy search
            results = self._registry.search(skill_name)
            if results:
                skill = results[0]
            else:
                return f"Skill '{skill_name}' not found. Use learn_skill to learn it first."

        self._registry.update_usage(skill["name"])

        if not self._model:
            return f"Skill loaded: {skill['name']}\n\nProcedure:\n{skill['procedure']}\n\nCode Template:\n{skill.get('code_template', 'N/A')}"

        prompt = f"""Apply the following skill to complete the task.

Skill: {skill['name']}
Domain: {skill['domain']}

Procedure:
{skill['procedure']}

Code Template:
{skill.get('code_template', 'N/A')}

Task to complete: {task or 'Demonstrate how to use this skill'}

Provide practical guidance and code to complete the task using this skill."""

        return self._model.chat([
            {"role": "system", "content": "You are an expert applying learned skills to solve specific tasks."},
            {"role": "user", "content": prompt},
        ])
