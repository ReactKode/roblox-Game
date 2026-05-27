"""Autonomous skill learner — multi-pass research, synthesis, and validation."""
from datetime import datetime
from ..tools import web_search, web_scrape


SYNTHESIZE_PROMPT = """You are an expert technical trainer. Analyze this research about "{topic}" and create a high-quality, actionable skill guide.

Research (from {source_count} sources):
{content}

Create a comprehensive skill guide. Output ONLY this JSON:
{{
  "description": "one sentence describing what this skill enables",
  "procedure": "NUMBERED STEP-BY-STEP PROCEDURE:\\n1. Install...\\n2. Open...\\n3. ...",
  "code_template": "# working code template here or N/A",
  "key_concepts": ["concept1", "concept2", "concept3"],
  "tips": ["tip1", "tip2", "tip3"],
  "tools_required": ["tool1", "tool2"],
  "tags": ["tag1", "tag2", "tag3"]
}}

Rules:
- Procedure must have at least 6 numbered steps
- Code template must be real, runnable code (not pseudocode)
- Tips must be specific and actionable, not generic
- If no code applies, set code_template to "N/A"
"""

RESEARCH_QUERIES = {
    "blender":       ["{topic} Python bpy API tutorial", "{topic} Blender scripting documentation"],
    "unity":         ["{topic} Unity C# tutorial", "{topic} Unity3D documentation example"],
    "godot":         ["{topic} Godot GDScript tutorial", "{topic} Godot 4 documentation"],
    "unreal_engine": ["{topic} Unreal Engine 5 tutorial", "{topic} UE5 blueprint documentation"],
    "python":        ["{topic} Python tutorial", "{topic} Python documentation examples"],
    "machine_learning": ["{topic} machine learning tutorial code", "{topic} PyTorch TensorFlow example"],
    "business":      ["{topic} business strategy guide", "{topic} startup best practices"],
}


class SkillLearner:
    def __init__(self, registry, validator=None, model_client=None, memory_manager=None):
        self._registry = registry
        self._validator = validator
        self._model = model_client
        self._memory = memory_manager

    def set_model(self, model):
        self._model = model
        if self._validator:
            self._validator.set_model(model)

    def learn(self, topic: str, domain: str = "auto", force: bool = False) -> dict:
        if domain == "auto" or not domain:
            domain = self._registry.detect_domain(topic)

        # Return existing skill if confident enough (don't waste resources)
        if not force:
            existing = self._registry.search(topic, domain)
            if existing:
                best = existing[0]
                if best["name"].lower() == topic.lower() and best.get("confidence", 0) >= 0.7:
                    self._registry.update_usage(best["name"])
                    return best

        # Multi-pass research
        research_content, source_count = self._research(topic, domain)

        # Synthesize skill from research
        skill_data = self._synthesize(topic, domain, research_content, source_count)

        # Register first (gives it a proper path)
        skill = self._registry.register(skill_data)

        # Validate and update confidence
        if self._validator:
            confidence, issues = self._validator.validate(skill)
            self._registry.update_confidence(skill["name"], confidence, issues)
            skill["confidence"] = confidence
            skill["validation_issues"] = issues
        else:
            skill["confidence"] = 0.55  # unvalidated default

        # Store research findings in semantic memory
        if self._memory and research_content:
            self._memory.store_research(
                f"Research on {topic}: {research_content[:600]}",
                source=f"skill_learning/{domain}",
            )

        return skill

    def _research(self, topic: str, domain: str) -> tuple[str, int]:
        """Search and scrape from multiple sources. Returns (content, source_count)."""
        queries = RESEARCH_QUERIES.get(domain, [
            f"{topic} tutorial documentation how to",
            f"{topic} complete guide examples",
        ])
        queries = [q.format(topic=topic) for q in queries[:2]]
        # Add a general query
        queries.append(f"{topic} best practices tips")

        collected = []
        urls_scraped = set()

        for query in queries[:3]:
            results = web_search.search(query, max_results=4)
            for r in results[:3]:
                snippet = r.get("snippet", "").strip()
                url = r.get("url", "")
                title = r.get("title", "")

                if snippet and len(snippet) > 30:
                    collected.append(f"[{title}]\n{snippet}")

                if url and url not in urls_scraped and len(collected) < 8:
                    page = web_scrape.scrape(url, max_chars=2500)
                    if page and not page.startswith("Error") and len(page) > 100:
                        collected.append(f"[Full page: {title}]\n{page}")
                        urls_scraped.add(url)

            if len(collected) >= 6:
                break

        content = "\n\n---\n\n".join(collected[:6]) if collected else f"No research found for '{topic}'."
        return content, len(urls_scraped)

    def _synthesize(self, topic: str, domain: str, content: str, source_count: int) -> dict:
        """Use LLM to create structured skill from research content."""
        if self._model:
            return self._llm_synthesize(topic, domain, content, source_count)
        return self._template_skill(topic, domain, content)

    def _llm_synthesize(self, topic: str, domain: str, content: str, source_count: int) -> dict:
        import json, re
        prompt = SYNTHESIZE_PROMPT.format(
            topic=topic,
            source_count=max(source_count, 1),
            content=content[:6000],
        )
        response = self._model.chat([
            {"role": "system", "content": "You are an expert technical trainer creating skill guides. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ])

        # Parse JSON from response
        data = None
        try:
            data = json.loads(response.strip())
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if not data:
            return self._template_skill(topic, domain, content)

        return {
            "name": topic,
            "description": data.get("description", f"Skills and techniques for {topic}"),
            "domain": domain,
            "procedure": data.get("procedure", self._template_procedure(topic, content)),
            "code_template": data.get("code_template", self._template_code(domain)),
            "tools_required": data.get("tools_required", []),
            "tags": data.get("key_concepts", [])[:6] + data.get("tags", [])[:3],
            "metadata": {
                "source": "llm_synthesized",
                "sources_researched": source_count,
                "synthesized_at": datetime.utcnow().isoformat(),
            },
        }

    def _template_skill(self, topic: str, domain: str, content: str) -> dict:
        """Fallback when no LLM available."""
        return {
            "name": topic,
            "description": f"Skills and techniques for {topic}",
            "domain": domain,
            "procedure": self._template_procedure(topic, content),
            "code_template": self._template_code(domain),
            "tools_required": [],
            "tags": [domain, topic.split()[0].lower()],
            "metadata": {"source": "template_fallback"},
        }

    def _template_procedure(self, topic: str, research: str) -> str:
        lines = [f"# How to work with: {topic}", ""]
        lines.append("## Setup")
        lines.append(f"1. Install required dependencies for {topic}")
        lines.append(f"2. Read the official documentation")
        lines.append("")
        lines.append("## Core Steps (from research)")
        research_lines = [l.strip() for l in research.splitlines() if len(l.strip()) > 40][:8]
        for i, l in enumerate(research_lines, 3):
            lines.append(f"{i}. {l[:200]}")
        lines.append("")
        lines.append("## Best Practices")
        lines.append(f"- Start with small examples to build understanding")
        lines.append(f"- Refer to official {topic} documentation for edge cases")
        lines.append(f"- Join community forums for advanced help")
        return "\n".join(lines)

    def _template_code(self, domain: str) -> str:
        templates = {
            "blender": (
                '"""Blender Python (bpy) — run inside Blender Script Editor"""\n'
                "import bpy\n\n"
                "# Clear default objects\n"
                "bpy.ops.object.select_all(action='SELECT')\n"
                "bpy.ops.object.delete()\n\n"
                "# Add a mesh object\n"
                "bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))\n"
                "obj = bpy.context.active_object\n"
                "obj.name = 'NexusObject'\n\n"
                "# Apply a material\n"
                "mat = bpy.data.materials.new(name='NexusMaterial')\n"
                "mat.use_nodes = True\n"
                "obj.data.materials.append(mat)\n"
                "print(f'Created {obj.name} with material {mat.name}')"
            ),
            "unity": (
                "// Unity C# MonoBehaviour\n"
                "using UnityEngine;\n\n"
                "public class NexusController : MonoBehaviour\n"
                "{\n"
                "    [SerializeField] private float speed = 5f;\n"
                "    private Rigidbody _rb;\n\n"
                "    void Start() {\n"
                "        _rb = GetComponent<Rigidbody>();\n"
                "        Debug.Log(\"NexusAgent skill initialized\");\n"
                "    }\n\n"
                "    void Update() {\n"
                "        float h = Input.GetAxis(\"Horizontal\");\n"
                "        float v = Input.GetAxis(\"Vertical\");\n"
                "        _rb.MovePosition(transform.position + new Vector3(h, 0, v) * speed * Time.deltaTime);\n"
                "    }\n"
                "}"
            ),
            "godot": (
                "# Godot 4 GDScript\n"
                "extends CharacterBody3D\n\n"
                "const SPEED = 5.0\n"
                "const JUMP_VELOCITY = 4.5\n\n"
                "func _ready():\n"
                '    print("NexusAgent Godot skill loaded")\n\n'
                "func _physics_process(delta: float) -> void:\n"
                "    if not is_on_floor():\n"
                "        velocity += get_gravity() * delta\n"
                "    var direction = Input.get_vector(\"ui_left\", \"ui_right\", \"ui_up\", \"ui_down\")\n"
                "    velocity.x = direction.x * SPEED\n"
                "    velocity.z = direction.y * SPEED\n"
                "    move_and_slide()"
            ),
            "python": (
                '"""Python skill template"""\n'
                "import os\n"
                "import json\n"
                "from pathlib import Path\n\n\n"
                "def process(data: dict) -> dict:\n"
                '    """Process input data and return results."""\n'
                "    results = {}\n"
                "    for key, value in data.items():\n"
                "        results[key] = str(value).upper()\n"
                "    return results\n\n\n"
                'if __name__ == "__main__":\n'
                '    sample = {"input": "hello from NexusAgent"}\n'
                "    print(json.dumps(process(sample), indent=2))"
            ),
            "machine_learning": (
                '"""ML skill template using scikit-learn"""\n'
                "from sklearn.ensemble import RandomForestClassifier\n"
                "from sklearn.model_selection import train_test_split\n"
                "from sklearn.metrics import accuracy_score\n"
                "import numpy as np\n\n"
                "# Generate sample data\n"
                "X = np.random.rand(200, 4)\n"
                "y = (X[:, 0] + X[:, 1] > 1).astype(int)\n\n"
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n"
                "model = RandomForestClassifier(n_estimators=100, random_state=42)\n"
                "model.fit(X_train, y_train)\n"
                "print(f'Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2%}')"
            ),
        }
        return templates.get(domain, f"# {domain} code template\n# TODO: implement\nprint('NexusAgent skill')")
