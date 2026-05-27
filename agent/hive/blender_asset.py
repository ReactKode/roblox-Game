"""Blender headless asset creator — generates bpy scripts and executes them."""
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class AssetFile:
    name: str
    format: str          # fbx | glb | obj
    path: Path | None    # None = Blender not installed, script only


@dataclass
class AssetPackage:
    asset_name: str
    description: str
    bpy_script: str          # Full Python script source
    script_path: Path        # Where script.py was saved
    files: list[AssetFile]   # Exported files (empty if no Blender)
    blender_available: bool
    asset_dir: Path
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def fbx_path(self) -> Path | None:
        for f in self.files:
            if f.format == "fbx":
                return f.path
        return None

    def glb_path(self) -> Path | None:
        for f in self.files:
            if f.format == "glb":
                return f.path
        return None


# ── Blender detection ─────────────────────────────────────────────────────────

_BLENDER_SEARCH = [
    "blender",
    "/usr/bin/blender",
    "/usr/local/bin/blender",
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "C:/Program Files/Blender Foundation/Blender 4.1/blender.exe",
    "C:/Program Files/Blender Foundation/Blender 4.0/blender.exe",
    "C:/Program Files/Blender Foundation/Blender 3.6/blender.exe",
]


def find_blender() -> str | None:
    for candidate in _BLENDER_SEARCH:
        found = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if found:
            return found
    return None


# ── Prompt templates ──────────────────────────────────────────────────────────

BPY_SYSTEM = """\
You are an expert Blender Python developer specializing in game-ready asset creation.
You write clean, headless-compatible bpy scripts that create production-quality 3D assets.

Critical headless rules (NEVER violate these):
- NO bpy.ops.view3d.* calls (requires viewport)
- Always set active object: bpy.context.view_layer.objects.active = obj
- Use bpy.ops.object.select_all(action='DESELECT') before manual selection
- SmartUV project works headlessly: bpy.ops.uv.smart_project()
- Output only the Python script, starting with: import bpy\
"""

BPY_PROMPT = """\
Create a complete Blender Python script (headless) for this asset:

Asset: {description}
Object name: {asset_name}
FBX export path: {fbx_path}
GLTF export path: {glb_path}

Script must:
1. `import bpy, os` at top
2. Clear default scene (select all → delete)
3. Create realistic {asset_type} geometry using primitives + modifiers (Subdivision, Solidify, Bevel)
4. Add proper Principled BSDF material with game-appropriate values (Metallic/Roughness)
5. UV Unwrap with Smart UV Project
6. Export FBX to: {fbx_path}
7. Export GLTF/GLB to: {glb_path}
8. Use print() to log each major step

Game-ready specs:
- Poly count: 500-3000 for small props, 2000-8000 for characters/weapons
- Scale: real-world (1 unit = 1 meter), weapon ~1m long
- Origin: at object base or logical center
- No loose geometry, clean topology

Output ONLY the Python code starting with `import bpy`:\
"""


# ── Blender Asset Agent ───────────────────────────────────────────────────────

class BlenderAssetAgent:
    def __init__(self, model_client, data_dir: Path):
        self._model = model_client
        self._data_dir = data_dir
        self._blender = find_blender()

    @property
    def blender_available(self) -> bool:
        return self._blender is not None

    def create(self, description: str, asset_name: str = None,
                asset_type: str = "prop") -> AssetPackage:
        """Generate bpy script, optionally execute headlessly, return AssetPackage."""
        if not asset_name:
            asset_name = _slug(description)

        asset_dir = self._data_dir / "assets" / asset_name
        asset_dir.mkdir(parents=True, exist_ok=True)

        blender_dir = asset_dir / "blender"
        blender_dir.mkdir(exist_ok=True)
        exports_dir = asset_dir / "exports"
        exports_dir.mkdir(exist_ok=True)

        fbx_path = exports_dir / f"{asset_name}.fbx"
        glb_path = exports_dir / f"{asset_name}.glb"

        # Generate script
        script = self._generate_script(description, asset_name, asset_type,
                                        fbx_path, glb_path)
        script_path = blender_dir / f"create_{asset_name}.py"
        script_path.write_text(script)

        # Execute headlessly if Blender found
        files: list[AssetFile] = []
        if self._blender:
            success = self._execute(script_path)
            if success:
                if fbx_path.exists():
                    files.append(AssetFile(name=asset_name, format="fbx", path=fbx_path))
                if glb_path.exists():
                    files.append(AssetFile(name=asset_name, format="glb", path=glb_path))

        return AssetPackage(
            asset_name=asset_name,
            description=description,
            bpy_script=script,
            script_path=script_path,
            files=files,
            blender_available=self._blender is not None,
            asset_dir=asset_dir,
        )

    def _generate_script(self, description: str, asset_name: str,
                          asset_type: str, fbx_path: Path, glb_path: Path) -> str:
        prompt = BPY_PROMPT.format(
            description=description,
            asset_name=asset_name,
            asset_type=asset_type,
            fbx_path=str(fbx_path).replace("\\", "/"),
            glb_path=str(glb_path).replace("\\", "/"),
        )
        response = self._model.chat([
            {"role": "system", "content": BPY_SYSTEM},
            {"role": "user", "content": prompt},
        ])
        script = _extract_code(response)
        # Inject path variables at top in case LLM hardcoded wrong paths
        header = (
            f"# NexusAgent — Generated Blender script for: {description}\n"
            f"# Run: blender --background --python {fbx_path.name.replace('.fbx', '.py')}\n\n"
        )
        return header + script

    def _execute(self, script_path: Path) -> bool:
        """Run Blender headlessly. Returns True if exit code 0."""
        try:
            result = subprocess.run(
                [self._blender, "--background", "--python", str(script_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            # Write Blender stdout/stderr log
            log_path = script_path.parent / "blender_output.log"
            log_path.write_text(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False


# ── helpers ───────────────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    return re.sub(r"[^\w]", "_", text.lower().strip())[:40].strip("_")


def _extract_code(response: str) -> str:
    # Pull code from ```python block if present
    m = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(import bpy.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Find start of actual code
    idx = response.find("import bpy")
    return response[idx:].strip() if idx != -1 else response.strip()
