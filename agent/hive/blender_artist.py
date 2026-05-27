"""
BlenderArtistAgent — the full creative pipeline:

  Developer brief → Research references → Analyze style → Plan geometry
  → Generate expert bpy script → Execute headlessly → Return AssetPackage

Key insight: quality comes from baking real Blender technique knowledge
into every prompt. The LLM doesn't need to invent approaches — we give
it the right Blender pattern for each asset type and it fills in the
creative details (proportions, colors, variation).
"""
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..tools import web_search, web_scrape
from .blender_asset import AssetPackage, AssetFile, _slug, _extract_code, find_blender


# ═══════════════════════════════════════════════════════════════════════════
#  Technique library — the RIGHT Blender approach per asset type
#  These snippets are injected into the LLM prompt so it generates
#  technically correct, headless-compatible bpy code.
# ═══════════════════════════════════════════════════════════════════════════

TECHNIQUES: dict[str, str] = {

    "castle_tower": """\
Blender approach for a castle tower:
- Base: bpy.ops.mesh.primitive_cylinder_add(radius=3, depth=10, vertices=12)
- Battlements: create one merlon cube (0.6w, 0.5d, 1.0h), position at top edge
  Add Array modifier: count=12, relative_offset X=1.1
  Wrap around tower top using a Curve modifier (circle path, radius=3)
- Arrow slits: boolean DIFFERENCE with a thin elongated cube (0.15w, 1.0d, 0.5h)
  Rotate/position around cylinder at mid height
- Stone material: Principled BSDF, base_color=(0.35,0.32,0.29), metallic=0.0, roughness=0.92
- Floor of tower: cap both ends (Fill Holes in edit mode)
- Scale: 1 unit = 1 metre, tower ~8-12m tall, radius 2.5-4m""",

    "castle_wall": """\
Blender approach for a castle wall section:
- Base wall: cube scaled to (15.0, 1.5, 5.0) — length, thickness, height
- Battlements along top:
  Create merlon cube (1.0, 1.5, 1.0), position flush with wall top
  Array modifier: count=7, relative_offset_displace X=1.5
- Gateway arch (optional): two cube pillars + top lintel
  OR Boolean DIFFERENCE with a rounded arch shape from a bezier curve circle
- Walk platform inside: thin plane at 4.5m height, width of wall
- Stone material: same parameters as tower for consistency
- Scale: section 15m wide, 5m tall, 1.5m thick""",

    "castle_full": """\
Blender approach for a full castle structure (assemble from parts):
1. Main keep: large cube (20x20x15), scaled with thick walls implied by windows
2. Four corner towers: cylinders (radius=3, depth=12) at each corner using Array
   or manually placed. Use duplivert or manual loop.
3. Outer curtain wall: four wall sections using Array modifier along each side
4. Gatehouse: two towers close together with portcullis gap
5. Battlements on all upper edges (Array modifier on each wall/tower top)
6. Drawbridge slot: Boolean at gate base
Material strategy: one shared stone material for all parts for consistency
Assembly: use bpy.ops.object.join() to merge non-boolean pieces""",

    "house": """\
Blender approach for a house/building:
- Foundation slab: cube scaled (8, 6, 0.3), z_location=-0.15
- Walls: cube scaled (8, 6, 3.5), location z=1.75
  Boolean DIFFERENCE windows: cube (0.9, 1.5, 1.2) per window position
  Boolean DIFFERENCE door: cube (1.0, 0.5, 2.1) at front center
- Roof (gabled): start with a cube, go to edit mode (bpy.ops.object.mode_set mode='EDIT')
  Select top 4 verts, scale X to 0 (ridge), extrude up 1.5m — this creates gable
  Overhang: Solidify modifier or manual extrude outward 0.3m
- Chimney: cylinder (radius=0.25, depth=1.5) on roof slope
- Door frame: cube around door hole, slightly protruding
- Window frames: thin cube border around each window
Materials:
  walls = Principled BSDF (0.75, 0.68, 0.58) roughness=0.9  — plaster/stone
  roof  = Principled BSDF (0.25, 0.18, 0.12) roughness=0.95 — dark thatch/tile
  wood  = Principled BSDF (0.40, 0.25, 0.12) roughness=0.88 — timber frames""",

    "tree": """\
Blender approach for a game-ready stylized tree:
Option A — Low poly stylized:
- Trunk: bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=4, vertices=8)
  Add Taper: proportional edit, narrow top. Or use CTRL+R loop cuts + scale.
- Canopy layers (3): ico sphere at different heights, scales, slight offsets
  bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=2.0)
  Position: bottom canopy at h=3.5, middle at h=5, top at h=6.5
  Scale randomly within (0.8-1.3, 0.8-1.3, 0.7-1.1) for natural look
- Apply Decimate (ratio=0.4) to all canopy spheres for faceted low-poly look
Materials:
  bark    = Principled BSDF (0.25, 0.15, 0.08) roughness=0.95
  foliage = Principled BSDF (0.12, 0.38, 0.08) roughness=0.85
Option B — Add Subdivision + Displace to trunk for organic shape
  texture = bpy.data.textures.new('bark', type='CLOUDS')
  displace = trunk.modifiers.new('Displace','DISPLACE'); displace.texture=texture; displace.strength=0.15""",

    "terrain": """\
Blender approach for game terrain:
- Base plane: bpy.ops.mesh.primitive_plane_add(size=200)
  Add Subdivision modifier: render_levels=6, levels=4
- Height variation using Displace modifier:
  tex = bpy.data.textures.new('TerrainNoise', type='CLOUDS')
  tex.noise_scale = 2.5; tex.noise_depth = 4
  disp = terrain.modifiers.new('Terrain','DISPLACE')
  disp.texture = tex; disp.strength = 8.0; disp.mid_level = 0.0
- Smooth normals: bpy.ops.object.shade_smooth()
- For paths: add a second plane on top, boolean union or vertex color painting
- Material: Principled BSDF (0.28, 0.22, 0.15) roughness=0.98 — dirt/earth
  Add MixShader with grass color (0.15, 0.35, 0.10) driven by object Z position
- Scale: 200x200m for a mid-size game level""",

    "rock": """\
Blender approach for rocks/boulders:
- Base: bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.5)
- Decimate modifier: ratio=0.35 for chunky low-poly look
- Displace modifier for surface variation:
  tex = bpy.data.textures.new('RockNoise', type='CLOUDS')
  tex.noise_scale = 0.8
  disp = rock.modifiers.new('Disp','DISPLACE'); disp.strength=0.3
- Squash/stretch: scale(1.3, 0.8, 0.9) for non-uniform boulder shape
- Material: Principled BSDF (0.42, 0.40, 0.38) roughness=0.95
  Subsurface=0 — pure stone
- Variations: duplicate base rock, apply random scale/rotation for variety""",

    "fence": """\
Blender approach for wooden fence:
- Single post: cylinder (radius=0.06, depth=1.2), location z=0.6
- Two horizontal rails: cube (0.05, 3.0, 0.08) at heights z=0.4 and z=0.85
- Array modifier on post: count=4, offset X=1.0 (1 metre spacing)
- Join post + rails into one mesh
- Repeat for full fence run using another Array
- Wood material: Principled BSDF (0.38, 0.22, 0.10) roughness=0.92
- Weathered look: slight blue tint in roughness, scale roughness texture""",

    "prop_barrel": """\
Blender approach for a wooden barrel prop:
- Body: cylinder (radius=0.35, depth=0.7, vertices=16)
  Scale mid section wider: add 2 loop cuts at mid, scale out to 0.38
- Top/bottom caps: fill with flat face
- Metal bands (3): torus (major_radius=0.36, minor_radius=0.025) at top, mid, bottom
  Or use cylinder (same radius, depth=0.03) as ring
- Wood material: Principled BSDF (0.35, 0.20, 0.10) roughness=0.90
- Metal material: Principled BSDF (0.2, 0.2, 0.2) metallic=0.9, roughness=0.4""",

    "prop_crate": """\
Blender approach for a wooden crate:
- Box: cube scaled (1.0, 1.0, 1.0)
- Edge boards: 4 thin cubes (0.08 thick) along each vertical edge
- Top/bottom planks: 3 planks across top face (use Boolean or model directly)
- Solidify on box if needed for wall thickness (0.1)
- Iron corner brackets: small cube at each corner, metallic material
- Wood material: (0.42, 0.28, 0.14) roughness=0.88
- Metal brackets: metallic=0.85, roughness=0.45""",

    "ground_plane": """\
Blender approach for a flat ground/floor:
- Plane: bpy.ops.mesh.primitive_plane_add(size=100)
- Subdivide: bpy.ops.mesh.subdivide(number_cuts=10) for texture variation
- Slight displacement for non-flat feel
- Ground material: mix of dirt (0.28,0.22,0.15) and grass (0.15,0.35,0.10)
  Use MixShader with object coordinate Z to blend
- For road/path: second plane slightly above, different material (packed earth)""",

    "water": """\
Blender approach for water surface:
- Plane: bpy.ops.mesh.primitive_plane_add(size=50, location=(0,0,-0.2))
- Subdivision modifier (level=4) for wave simulation capability
- Material (translucent water):
  mat = bpy.data.materials.new('Water'); mat.use_nodes = True
  bsdf = nodes['Principled BSDF']
  bsdf.inputs['Base Color'].default_value = (0.02, 0.15, 0.25, 1.0)
  bsdf.inputs['Alpha'].default_value = 0.7
  bsdf.inputs['Roughness'].default_value = 0.05
  bsdf.inputs['IOR'].default_value = 1.333
  mat.blend_method = 'BLEND'""",
}

# Maps keyword patterns → technique keys
TECHNIQUE_KEYWORDS: list[tuple[list[str], str]] = [
    (["castle tower", "tower", "turret", "spire"],       "castle_tower"),
    (["castle wall", "curtain wall", "battlement", "rampart"], "castle_wall"),
    (["castle", "fortress", "keep", "citadel"],          "castle_full"),
    (["house", "cottage", "building", "hut", "cabin", "tavern", "inn", "shop"], "house"),
    (["tree", "oak", "pine", "birch", "forest", "foliage"], "tree"),
    (["terrain", "landscape", "ground", "hillside", "mountain", "hill"], "terrain"),
    (["rock", "boulder", "stone", "cliff"],              "rock"),
    (["fence", "palisade", "stockade", "railing"],       "fence"),
    (["barrel", "cask", "keg"],                          "prop_barrel"),
    (["crate", "box", "chest", "container"],             "prop_crate"),
    (["ground", "floor", "surface", "path", "road"],     "ground_plane"),
    (["water", "lake", "river", "pond", "sea", "ocean"], "water"),
]


def get_technique(description: str) -> str:
    d = description.lower()
    for keywords, key in TECHNIQUE_KEYWORDS:
        if any(kw in d for kw in keywords):
            return TECHNIQUES.get(key, "")
    return ""


# ═══════════════════════════════════════════════════════════════════════════
#  Reference research
# ═══════════════════════════════════════════════════════════════════════════

RESEARCH_QUERIES: list[str] = [
    "{description} 3D model reference proportions dimensions",
    "{description} game asset blender tutorial technique",
    "{description} low poly game ready stylized concept art",
]


def research_asset(description: str, asset_type: str, style: str = "low poly stylized") -> str:
    """Search web for reference material and return consolidated findings."""
    queries = [q.format(description=f"{style} {description}") for q in RESEARCH_QUERIES[:2]]
    collected: list[str] = []

    for query in queries:
        results = web_search.search(query, max_results=4)
        for r in results[:3]:
            snippet = r.get("snippet", "").strip()
            if snippet and len(snippet) > 30:
                collected.append(f"[{r['title']}]: {snippet}")
            url = r.get("url", "")
            if url and len(collected) < 8:
                page = web_scrape.scrape(url, max_chars=2000)
                if page and not page.startswith("Error") and len(page) > 100:
                    collected.append(f"[{r['title']} — full]: {page[:1500]}")
        if len(collected) >= 5:
            break

    if not collected:
        return f"No reference found for '{description}'. Use standard game art conventions."
    return "\n\n".join(collected[:5])


# ═══════════════════════════════════════════════════════════════════════════
#  Vision analysis (if multimodal model available)
# ═══════════════════════════════════════════════════════════════════════════

VISION_PROMPT = """\
You are analyzing reference images for a 3D artist creating a game asset.

Asset to create: {description}

From what you can observe in these reference images, extract:
1. Key proportions (height-to-width ratio, relative sizes of parts)
2. Main geometric shapes (cylinders, cubes, arches, etc.)
3. Color palette (describe 3-5 key colors with approximate hex codes)
4. Surface texture character (rough stone, smooth wood, worn metal, etc.)
5. Style (realistic vs stylized, low-poly vs detailed, dark vs bright)
6. Unique identifying features that should be in the 3D model

Be specific and practical for a Blender artist."""


def analyze_with_vision(model_client, image_urls: list[str], description: str) -> str:
    """Use vision model to analyze reference images. Returns analysis text."""
    if not image_urls or not hasattr(model_client, "chat_with_image"):
        return ""
    try:
        analysis = model_client.chat_with_image(
            VISION_PROMPT.format(description=description),
            image_urls[:3],
        )
        return analysis
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════════════
#  Script generation prompt
# ═══════════════════════════════════════════════════════════════════════════

ARTIST_SYSTEM = """\
You are a senior Blender Python developer and 3D game artist.
You create production-quality headless bpy scripts for game assets.

HEADLESS RULES (never break these):
- No bpy.ops.view3d.* (needs viewport)
- Set active: bpy.context.view_layer.objects.active = obj
- Select: bpy.context.selected_objects OR obj.select_set(True)
- Smart UV: bpy.ops.object.mode_set(mode='EDIT') then bpy.ops.uv.smart_project()
  Then back: bpy.ops.object.mode_set(mode='OBJECT')
- Edit mode ops need active object selected first
- Modifiers: apply ONLY if needed for export; unapplied is fine for FBX

Output ONLY the Python script starting with: import bpy\
"""

ARTIST_PROMPT = """\
Create a complete headless Blender Python script for this game asset.

━━━ ASSET BRIEF ━━━
Description:  {description}
Asset type:   {asset_type}
Style:        {style}
Object name:  {asset_name}
FBX output:   {fbx_path}
GLB output:   {glb_path}

━━━ BLENDER TECHNIQUE FOR THIS ASSET TYPE ━━━
{technique}

━━━ REFERENCE RESEARCH ━━━
{research}

━━━ VISUAL ANALYSIS ━━━
{vision_analysis}

━━━ SCRIPT REQUIREMENTS ━━━
1. import bpy, os, math at top
2. Clear scene: select all → delete
3. Build asset using techniques above as your foundation
4. Add 2-3 material slots with proper Principled BSDF values
5. UV unwrap: bpy.ops.object.mode_set(mode='EDIT') + bpy.ops.uv.smart_project()
6. Export FBX:  bpy.ops.export_scene.fbx(filepath=r"{fbx_path}", use_selection=False)
7. Export GLTF: bpy.ops.export_scene.gltf(filepath=r"{glb_path}", export_format='GLB')
8. print() each major step for logging

Game specs: 1 unit = 1 metre | poly budget: {poly_budget} | origin at base
Style guidance: {style} — match color palette and proportional feel

Output only the Python script:\
"""


# ═══════════════════════════════════════════════════════════════════════════
#  BlenderArtistAgent
# ═══════════════════════════════════════════════════════════════════════════

POLY_BUDGETS: dict[str, int] = {
    "hero": 8000, "character": 6000, "building": 4000,
    "castle": 5000, "prop": 1500, "tree": 2000,
    "terrain": 10000, "environment": 8000, "default": 3000,
}


@dataclass
class AssetBrief:
    name: str
    description: str
    asset_type: str = "prop"     # prop | building | character | terrain | environment
    style: str = "low poly stylized dark fantasy"
    poly_budget: int = 3000
    position: tuple = (0.0, 0.0, 0.0)
    scale: float = 1.0


class BlenderArtistAgent:
    """
    Full creative pipeline:
      Brief → Research → Vision analysis → Technique selection → Script → Execute
    """

    def __init__(self, model_client, data_dir: Path):
        self._model = model_client
        self._data_dir = data_dir
        self._blender = find_blender()
        self._print_fn = print

    @property
    def blender_available(self) -> bool:
        return self._blender is not None

    def set_print(self, fn):
        self._print_fn = fn

    def _log(self, msg: str):
        self._print_fn(msg)

    # ── Public API ────────────────────────────────────────────────────────

    def create(self, brief: AssetBrief) -> AssetPackage:
        """Full pipeline for a single asset."""
        asset_dir = self._data_dir / "assets" / brief.name
        asset_dir.mkdir(parents=True, exist_ok=True)
        (asset_dir / "blender").mkdir(exist_ok=True)
        (asset_dir / "exports").mkdir(exist_ok=True)

        fbx_path = asset_dir / "exports" / f"{brief.name}.fbx"
        glb_path = asset_dir / "exports" / f"{brief.name}.glb"

        self._log(f"    🔍 Researching references for '{brief.description[:50]}'...")
        research = research_asset(brief.description, brief.asset_type, brief.style)

        technique = get_technique(brief.description)
        if not technique:
            technique = f"Use appropriate primitive meshes and modifiers for a {brief.asset_type}."

        vision = ""  # populated if vision model available

        self._log(f"    ✏️  Generating bpy script...")
        poly_budget = POLY_BUDGETS.get(brief.asset_type, POLY_BUDGETS["default"])
        script = self._generate_script(brief, technique, research, vision, fbx_path, glb_path, poly_budget)
        script_path = asset_dir / "blender" / f"create_{brief.name}.py"
        script_path.write_text(script)

        files: list[AssetFile] = []
        if self._blender:
            self._log(f"    ⚙️  Running Blender headlessly...")
            ok = self._execute(script_path, asset_dir)
            if ok:
                for fmt, p in [("fbx", fbx_path), ("glb", glb_path)]:
                    if p.exists():
                        files.append(AssetFile(name=brief.name, format=fmt, path=p))
                self._log(f"    ✓  Exported: {[f.format for f in files]}")
            else:
                self._log(f"    ⚠️  Blender execution failed — check blender/blender_output.log")
        else:
            self._log(f"    📄 Script saved (install Blender to execute)")

        return AssetPackage(
            asset_name=brief.name,
            description=brief.description,
            bpy_script=script,
            script_path=script_path,
            files=files,
            blender_available=self._blender is not None,
            asset_dir=asset_dir,
        )

    # ── Script generation ─────────────────────────────────────────────────

    def _generate_script(self, brief: AssetBrief, technique: str, research: str,
                          vision: str, fbx_path: Path, glb_path: Path, poly_budget: int) -> str:
        prompt = ARTIST_PROMPT.format(
            description=brief.description,
            asset_type=brief.asset_type,
            style=brief.style,
            asset_name=brief.name,
            fbx_path=str(fbx_path).replace("\\", "/"),
            glb_path=str(glb_path).replace("\\", "/"),
            technique=technique or "Use standard Blender primitives and modifiers.",
            research=research[:2000] if research else "No specific reference found.",
            vision_analysis=vision or "No visual reference available — use standard conventions.",
            poly_budget=poly_budget,
        )
        response = self._model.chat([
            {"role": "system", "content": ARTIST_SYSTEM},
            {"role": "user", "content": prompt},
        ])
        header = (
            f"# NexusAgent Blender Artist — {brief.description}\n"
            f"# Style: {brief.style}\n"
            f"# Run: blender --background --python {brief.name}.py\n\n"
        )
        return header + _extract_code(response)

    # ── Execution ─────────────────────────────────────────────────────────

    def _execute(self, script_path: Path, asset_dir: Path) -> bool:
        log_path = asset_dir / "blender" / "blender_output.log"
        try:
            result = subprocess.run(
                [self._blender, "--background", "--python", str(script_path)],
                capture_output=True, text=True, timeout=180,
            )
            log_path.write_text(f"EXIT CODE: {result.returncode}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            log_path.write_text("ERROR: Blender timed out after 180s")
            return False
        except Exception as e:
            log_path.write_text(f"ERROR: {e}")
            return False
