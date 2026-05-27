"""Autonomous skill learner - researches and stores new skills."""
from datetime import datetime
from ..tools import web_search, web_scrape


LEARN_PROMPT = """You are an expert AI trainer. Analyze the following documentation and research about "{topic}" and create a structured skill guide.

Research content:
{content}

Create a comprehensive skill guide with:
1. PROCEDURE: Step-by-step instructions for using this skill (numbered list, clear and actionable)
2. CODE_TEMPLATE: A practical Python/script code template demonstrating this skill (if applicable)
3. KEY_CONCEPTS: The 5 most important concepts to understand
4. TIPS: 3-5 pro tips and best practices
5. TOOLS_REQUIRED: List of tools, software, or libraries needed

Format your response EXACTLY as:
---PROCEDURE---
<numbered steps>
---CODE_TEMPLATE---
<code or "N/A">
---KEY_CONCEPTS---
<comma-separated list>
---TIPS---
<bullet points>
---TOOLS_REQUIRED---
<comma-separated list>
"""


class SkillLearner:
    def __init__(self, registry, tool_registry=None, model_client=None):
        self._registry = registry
        self._tools = tool_registry
        self._model = model_client

    def set_model(self, model_client):
        self._model = model_client

    def learn(self, topic: str, domain: str = "auto") -> dict:
        if domain == "auto" or not domain:
            domain = self._registry.detect_domain(topic)

        existing = self._registry.search(topic, domain)
        if existing and existing[0]["name"].lower() == topic.lower():
            self._registry.update_usage(existing[0]["name"])
            return existing[0]

        content = self._research(topic)
        skill = self._synthesize(topic, domain, content)
        return self._registry.register(skill)

    def _research(self, topic: str) -> str:
        """Search and scrape documentation for topic."""
        queries = [
            f"{topic} tutorial documentation how to",
            f"{topic} Python API example code",
            f"{topic} getting started guide",
        ]
        collected = []
        for query in queries[:2]:
            results = web_search.search(query, max_results=3)
            for r in results[:2]:
                snippet = r.get("snippet", "")
                if snippet:
                    collected.append(f"Source: {r['title']}\n{snippet}")
                url = r.get("url", "")
                if url and len(collected) < 6:
                    page_content = web_scrape.scrape(url, max_chars=3000)
                    if page_content and not page_content.startswith("Error"):
                        collected.append(f"From {r['title']}:\n{page_content}")
                    if len(collected) >= 4:
                        break

        return "\n\n===\n\n".join(collected[:4]) if collected else f"No documentation found for '{topic}'."

    def _synthesize(self, topic: str, domain: str, research_content: str) -> dict:
        """Use LLM to synthesize skill from research, or use template if no model."""
        procedure = ""
        code_template = ""
        key_concepts = []
        tips = ""
        tools_required = []

        if self._model:
            prompt = LEARN_PROMPT.format(topic=topic, content=research_content[:6000])
            response = self._model.chat([
                {"role": "system", "content": "You are an expert technical trainer creating concise skill guides."},
                {"role": "user", "content": prompt},
            ])
            procedure, code_template, key_concepts, tips, tools_required = self._parse_response(response)

        if not procedure:
            procedure = self._template_procedure(topic, domain, research_content)
            code_template = self._template_code(topic, domain)

        return {
            "name": topic,
            "description": f"Skills and techniques for {topic}",
            "domain": domain,
            "version": "1.0.0",
            "created_at": datetime.utcnow().isoformat(),
            "last_used": datetime.utcnow().isoformat(),
            "use_count": 1,
            "procedure": procedure,
            "code_template": code_template,
            "tools_required": tools_required if isinstance(tools_required, list) else [tools_required],
            "examples": [],
            "tags": key_concepts[:5] if isinstance(key_concepts, list) else [],
            "metadata": {"source": "autonomous_research", "research_chars": len(research_content)},
        }

    def _parse_response(self, response: str) -> tuple:
        sections = {
            "procedure": "",
            "code_template": "",
            "key_concepts": [],
            "tips": "",
            "tools_required": [],
        }
        current = None
        lines = []
        for line in response.splitlines():
            if "---PROCEDURE---" in line:
                current = "procedure"; lines = []
            elif "---CODE_TEMPLATE---" in line:
                sections["procedure"] = "\n".join(lines).strip(); current = "code_template"; lines = []
            elif "---KEY_CONCEPTS---" in line:
                sections["code_template"] = "\n".join(lines).strip(); current = "key_concepts"; lines = []
            elif "---TIPS---" in line:
                sections["key_concepts"] = [c.strip() for c in "\n".join(lines).split(",") if c.strip()]
                current = "tips"; lines = []
            elif "---TOOLS_REQUIRED---" in line:
                sections["tips"] = "\n".join(lines).strip(); current = "tools_required"; lines = []
            elif current:
                lines.append(line)
        if current == "tools_required":
            sections["tools_required"] = [t.strip() for t in "\n".join(lines).split(",") if t.strip()]

        return (
            sections["procedure"],
            sections["code_template"],
            sections["key_concepts"],
            sections["tips"],
            sections["tools_required"],
        )

    def _template_procedure(self, topic: str, domain: str, research: str) -> str:
        lines = [f"# How to use: {topic}", ""]
        lines.append("## Getting Started")
        lines.append(f"1. Install and set up {topic}")
        lines.append(f"2. Learn the basic interface and concepts of {topic}")
        lines.append(f"3. Follow official documentation at the project's website")
        lines.append("")
        lines.append("## Key Steps")
        research_lines = [l for l in research.splitlines() if len(l.strip()) > 30][:10]
        for i, l in enumerate(research_lines, 4):
            lines.append(f"{i}. {l.strip()[:200]}")
        lines.append("")
        lines.append("## Best Practices")
        lines.append(f"- Read the official {topic} documentation")
        lines.append(f"- Start with small projects to build familiarity")
        lines.append(f"- Join the {domain} community for support")
        return "\n".join(lines)

    def _template_code(self, topic: str, domain: str) -> str:
        templates = {
            "blender": '"""Blender Python (bpy) template"""\nimport bpy\n\n# Clear scene\nbpy.ops.object.select_all(action="SELECT")\nbpy.ops.object.delete()\n\n# Add a mesh\nbpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))\nobj = bpy.context.active_object\nobj.name = "MyMesh"\nprint(f"Created: {obj.name}")',
            "unity": '// Unity C# MonoBehaviour template\nusing UnityEngine;\n\npublic class MyScript : MonoBehaviour\n{\n    void Start() {\n        Debug.Log("NexusAgent Unity skill loaded");\n    }\n    void Update() {\n        // Called every frame\n    }\n}',
            "godot": '# Godot GDScript template\nextends Node\n\nfunc _ready():\n    print("NexusAgent Godot skill loaded")\n\nfunc _process(delta):\n    pass  # Called every frame',
            "python": '"""Python skill template"""\ndef main():\n    print("NexusAgent Python skill")\n\nif __name__ == "__main__":\n    main()',
        }
        return templates.get(domain, f'# {topic} - code template\n# TODO: implement specific code for {topic}')
