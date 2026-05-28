"""Skill validator — tests learned skills before storing them as trusted knowledge."""
import json
import re

from ..core.logger import get_logger

log = get_logger(__name__)

# Patterns that indicate the code template can actually be run safely
_RUNNABLE_DOMAINS = {"python", "javascript", "web", "research", "machine_learning", "business", "general"}
# Domains that require external software — can't run, do LLM review instead
_REVIEW_ONLY_DOMAINS = {"blender", "unity", "unreal_engine", "godot", "game_dev"}

_UNSAFE_PATTERNS = [
    r"\brm\s+-rf\b", r"\bos\.remove\b", r"\bshutil\.rmtree\b",
    r"\bsubprocess\b.*shell\s*=\s*True",
    r"\b__import__\b", r"\beval\b\s*\(",
    r"\bos\.system\b", r"\bpickle\.loads\b",
]

REVIEW_PROMPT = """Review this skill procedure for quality and accuracy.

Skill: {name}
Domain: {domain}
Procedure:
{procedure}

Code Template:
{code}

Rate this skill on:
1. Is the procedure actionable and specific? (not vague)
2. Are the steps in the right order?
3. Does the code template look syntactically correct?
4. Are there any obvious errors or hallucinations?

Output ONLY this JSON:
{{
  "is_valid": true,
  "confidence": 0.85,
  "issues": ["issue1 if any"],
  "suggestion": "one improvement suggestion"
}}"""


class SkillValidator:
    def __init__(self, model_client=None):
        self._model = model_client

    def set_model(self, m):
        self._model = m

    def validate(self, skill: dict) -> tuple[float, list[str]]:
        """Returns (confidence: float, issues: list[str])."""
        domain = skill.get("domain", "general")
        code = skill.get("code_template", "")
        procedure = skill.get("procedure", "")

        issues: list[str] = []

        # Basic quality checks (these add to issues but don't block code execution)
        if len(procedure.strip()) < 50:
            issues.append("Procedure is too short — likely incomplete")
        if procedure.count("\n") < 2:
            issues.append("Procedure has fewer than 3 lines — may be too sparse")
        if not code or code.strip() == "N/A":
            issues.append("No code template provided")

        # Safety check on code — unsafe patterns block execution
        is_unsafe = False
        for pattern in _UNSAFE_PATTERNS:
            if re.search(pattern, code):
                issues.append(f"Potentially unsafe code pattern detected: {pattern}")
                is_unsafe = True

        # Try running the code if domain allows it (run regardless of quality issues,
        # just not if code is unsafe or missing)
        run_confidence = None
        if domain in _RUNNABLE_DOMAINS and code and "N/A" not in code and not is_unsafe:
            run_confidence = self._run_code(code, issues)

        # LLM review for non-runnable domains or to supplement run result
        llm_confidence = None
        if self._model and (domain in _REVIEW_ONLY_DOMAINS or run_confidence is None):
            llm_confidence = self._llm_review(skill, issues)

        # Combine scores
        if run_confidence is not None and llm_confidence is not None:
            confidence = run_confidence * 0.6 + llm_confidence * 0.4
        elif run_confidence is not None:
            confidence = run_confidence
        elif llm_confidence is not None:
            confidence = llm_confidence
        else:
            # No validation path available — conservative score
            confidence = 0.5 if not issues else 0.3

        # Penalise for each confirmed issue
        confidence = max(0.1, confidence - len(issues) * 0.05)

        log.debug("Validated skill '%s' (domain=%s): confidence=%.2f, issues=%d",
                  skill.get("name", "?"), domain, confidence, len(issues))
        return round(confidence, 2), issues

    def _run_code(self, code: str, issues: list) -> float:
        try:
            from ..tools.code_runner import run_python
        except ImportError:
            log.warning("code_runner not available — skipping execution validation")
            return None

        test_code = f"""
import sys
try:
    exec(compile({repr(code)}, '<skill>', 'exec'), {{}})
except ImportError as e:
    print(f"IMPORT_WARNING: {{e}}")
except SyntaxError as e:
    print(f"SYNTAX_ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"RUNTIME_WARNING: {{e}}")
"""
        try:
            result = run_python(test_code, timeout=10)
        except Exception:
            log.exception("Code runner raised an exception")
            return 0.4

        stderr = result.get("stderr", "")
        stdout = result.get("stdout", "")

        if "SYNTAX_ERROR" in stderr:
            issues.append(f"Code has syntax error: {stderr[:200]}")
            return 0.2
        if "IMPORT_WARNING" in stdout:
            issues.append("Code requires additional libraries to run")
            return 0.6
        if result.get("success"):
            return 0.9
        issues.append(f"Code execution failed: {stderr[:150]}")
        return 0.35

    def _llm_review(self, skill: dict, issues: list) -> float | None:
        if not self._model:
            return None
        prompt = REVIEW_PROMPT.format(
            name=skill.get("name", ""),
            domain=skill.get("domain", ""),
            procedure=skill.get("procedure", "")[:800],
            code=skill.get("code_template", "N/A")[:500],
        )
        try:
            response = self._model.chat([
                {"role": "system", "content": "You are a technical reviewer. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ])
            data = _parse_review_json(response)
            if not data:
                log.warning("Validator LLM review returned no parseable JSON")
                return None
            for issue in data.get("issues", []):
                if issue and issue not in issues:
                    issues.append(issue)
            conf = float(data.get("confidence", 0.6))
            return max(0.0, min(1.0, conf))
        except Exception:
            log.exception("LLM review failed for skill '%s'", skill.get("name", "?"))
            return None


def _parse_review_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in [r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    for m in re.finditer(r"\{[^{}]{10,}\}", text, re.DOTALL):
        try:
            obj = json.loads(m.group())
            if any(k in obj for k in ("confidence", "is_valid", "issues")):
                return obj
        except json.JSONDecodeError:
            continue
    return None
