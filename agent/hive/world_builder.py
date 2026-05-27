"""WorldBuilder — plans and assembles complete game worlds from a single description.

Flow:
  1. Plan    — LLM creates a world layout: asset list, positions, scales, style
  2. Create  — BlenderArtistAgent builds each individual asset
  3. Assemble— Master Blender script loads all assets and positions them in scene
  4. Export  — Full world exported as FBX + GLB + per-engine import guides
  5. Report  — Summary of what was built
"""
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .blender_artist import BlenderArtistAgent, AssetBrief, find_blender
from .engine_importers import setup_unreal, setup_unity, setup_godot
from .blender_asset import AssetPackage

ENGINE_SETUP = {"unreal": setup_unreal, "unity": setup_unity, "godot": setup_godot}

# ── World plan prompt ──────────────────────────────────────────────────────────

WORLD_PLAN_PROMPT = """\
You are a 3D World Architect planning a complete game scene.

World Description: "{description}"
Art Style: {style}
World Size: {size}m × {size}m
Poly Budget: {poly_budget} total triangles

Design a world layout with 4-10 unique assets. Consider visual composition,
focal points, natural flow, and gameplay usability.

For each asset specify:
- name: snake_case identifier (unique, max 30 chars)
- description: detailed Blender-artist-level description (geometry, style, materials)
- asset_type: prop | building | terrain | environment | tree | rock | fence | water
- count: how many instances (1-8)
- positions: array of [x, y, z] positions in metres (one per instance)
- scale: uniform scale factor (0.5-3.0)
- rotation_z_degrees: array of Z rotations per instance (for variety)
- poly_budget: tris per instance (total/count, max 5000 each)

Think about:
- A terrain base layer (always first)
- Landmark/hero objects at center or slight offset
- Supporting props scattered naturally
- Depth: foreground, midground, background elements
- Natural groupings (e.g. house cluster, tree grove)

World size reference: {size}m = {size} units. Place assets within ±{half_size}m of origin.

Output ONLY valid JSON — an array of asset objects:
[
  {{
    "name": "terrain_base",
    "description": "rolling grassy terrain with slight hills, dirt path running east-west",
    "asset_type": "terrain",
    "count": 1,
    "positions": [[0, 0, 0]],
    "scale": 1.0,
    "rotation_z_degrees": [0],
    "poly_budget": 8000
  }},
  ...
]
"""

WORLD_ASSEMBLY_SCRIPT = '''\
"""
NexusAgent World Assembly Script
World: {world_name}
Assets: {asset_count}

Loads all individual asset GLBs and positions them to compose the complete world.
Run: blender --background --python world_assembly.py
"""
import bpy
import os
import math

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

WORLD_DIR = r"{world_dir}"
ASSETS = {assets_json}

def load_and_place(asset_name, glb_path, instances):
    """Load a GLB asset and create instances at specified positions."""
    if not os.path.exists(glb_path):
        print(f"  [SKIP] {{asset_name}}: GLB not found at {{glb_path}}")
        return

    for i, inst in enumerate(instances):
        bpy.ops.import_scene.gltf(filepath=glb_path)
        imported = bpy.context.selected_objects[:]
        if not imported:
            continue

        root = imported[0]
        if len(imported) > 1:
            bpy.context.view_layer.objects.active = root
            bpy.ops.object.join()
            root = bpy.context.active_object

        root.name = f"{{asset_name}}_{{i:02d}}"
        root.location = inst["position"]
        root.scale = (inst["scale"],) * 3
        root.rotation_euler = (0, 0, math.radians(inst.get("rotation_z", 0)))
        print(f"  Placed {{root.name}} at {{root.location}}")

print("\\n[WorldBuilder] Assembling world: {world_name}")
print(f"[WorldBuilder] {{len(ASSETS)}} asset types to place\\n")

for asset in ASSETS:
    name = asset["name"]
    glb_path = os.path.join(WORLD_DIR, name, "exports", f"{{name}}.glb")
    instances = [
        {{"position": pos, "scale": asset["scale"], "rotation_z": rot}}
        for pos, rot in zip(asset["positions"], asset["rotation_z_degrees"])
    ]
    print(f"Placing {{asset.get('count',1)}}x {{name}}...")
    load_and_place(name, glb_path, instances)

# Set world lighting
world = bpy.data.worlds["World"] if "World" in bpy.data.worlds else bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.05, 0.07, 0.12, 1.0)  # night sky
bg.inputs["Strength"].default_value = 0.3

# Add sun lamp
bpy.ops.object.light_add(type='SUN', location=(10, -10, 20))
sun = bpy.context.active_object
sun.name = "WorldSun"
sun.data.energy = 3.0
sun.rotation_euler = (math.radians(45), 0, math.radians(-45))

# Export full world
fbx_out = os.path.join(WORLD_DIR, "{world_name}.fbx")
glb_out = os.path.join(WORLD_DIR, "{world_name}.glb")

print(f"\\n[WorldBuilder] Exporting complete world...")
bpy.ops.export_scene.fbx(filepath=fbx_out, use_selection=False, global_scale=1.0)
bpy.ops.export_scene.gltf(filepath=glb_out, export_format='GLB', use_selection=False)
print(f"[WorldBuilder] Exported: {{fbx_out}}")
print(f"[WorldBuilder] Exported: {{glb_out}}")
print("[WorldBuilder] World assembly complete!")
'''


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class PlacedAsset:
    """A single asset type with placement data for all its instances."""
    name: str
    description: str
    asset_type: str
    count: int
    positions: list[list[float]]
    scale: float
    rotation_z_degrees: list[float]
    poly_budget: int
    package: AssetPackage = None  # filled after creation


@dataclass
class World:
    """A complete built world."""
    name: str
    description: str
    style: str
    size: float
    world_dir: Path
    assets: list[PlacedAsset] = field(default_factory=list)
    assembly_script: Path = None
    fbx_path: Path = None
    glb_path: Path = None
    blender_executed: bool = False


# ── WorldBuilderAgent ─────────────────────────────────────────────────────────

class WorldBuilderAgent:
    """
    Orchestrates world creation:
      description → plan → create each asset → assemble world scene → export
    """

    def __init__(self, model_client, data_dir: Path):
        self._model = model_client
        self._data_dir = data_dir
        self._artist = BlenderArtistAgent(model_client, data_dir / "assets")
        self._blender = find_blender()
        self._print_fn = print

    def set_print(self, fn):
        self._print_fn = fn
        self._artist.set_print(fn)

    def _log(self, msg: str):
        self._print_fn(msg)

    # ── Main entry ─────────────────────────────────────────────────────────

    def build(
        self,
        description: str,
        style: str = "low poly stylized",
        size: float = 100.0,
        engines: list[str] = None,
    ) -> World:
        """
        Full pipeline: plan → create assets → assemble world → distribute to engines.
        """
        engines = engines or ["unreal", "unity", "godot"]
        world_name = _slug(description)
        world_dir = self._data_dir / "worlds" / world_name
        world_dir.mkdir(parents=True, exist_ok=True)

        world = World(
            name=world_name,
            description=description,
            style=style,
            size=size,
            world_dir=world_dir,
        )

        # 1. Plan
        self._log(f"\n[WorldBuilder] Planning world: \"{description}\"")
        self._log(f"[WorldBuilder] Style: {style} | Size: {size}m × {size}m")
        placed_assets = self._plan(description, style, size)
        self._log(f"[WorldBuilder] {len(placed_assets)} asset types planned:\n")
        for a in placed_assets:
            self._log(f"   • {a.name} × {a.count}  [{a.asset_type}]")

        # 2. Create each asset
        self._log(f"\n[WorldBuilder] Creating assets...\n")
        for pa in placed_assets:
            self._log(f"🖌️  Creating '{pa.name}' × {pa.count}...")
            brief = AssetBrief(
                name=pa.name,
                description=pa.description,
                asset_type=pa.asset_type,
                style=style,
                poly_budget=pa.poly_budget,
            )
            pkg = self._artist.create(brief)
            pa.package = pkg
            status = "✓ executed" if pkg.files else "✓ script generated"
            self._log(f"   {status}")
        world.assets = placed_assets

        # 3. Write assembly script
        self._log(f"\n[WorldBuilder] Writing world assembly script...")
        script_path = self._write_assembly_script(world)
        world.assembly_script = script_path

        # 4. Execute assembly with Blender
        if self._blender:
            self._log(f"[WorldBuilder] Running Blender assembly headlessly...")
            ok = self._execute_assembly(script_path, world_dir)
            world.blender_executed = ok
            if ok:
                world.fbx_path = world_dir / f"{world_name}.fbx"
                world.glb_path = world_dir / f"{world_name}.glb"
                self._log(f"[WorldBuilder] ✓ World assembled!")
            else:
                self._log(f"[WorldBuilder] ⚠️  Assembly failed — check world_assembly.log")
        else:
            self._log(f"[WorldBuilder] Blender not found — scripts generated only")

        # 5. Engine distribution
        if engines:
            self._log(f"\n[WorldBuilder] Setting up engine importers...")
            self._distribute(world, engines)

        # 6. Report
        self._print_summary(world, engines)
        self._save_manifest(world)

        return world

    # ── Stage 1: Plan ─────────────────────────────────────────────────────

    def _plan(self, description: str, style: str, size: float) -> list[PlacedAsset]:
        poly_budget = int(size * 800)  # rough total budget based on world size
        prompt = WORLD_PLAN_PROMPT.format(
            description=description,
            style=style,
            size=int(size),
            half_size=int(size / 2),
            poly_budget=poly_budget,
        )
        response = self._model.chat([
            {"role": "system", "content": "You are a 3D World Architect. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ])
        return self._parse_plan(response)

    def _parse_plan(self, response: str) -> list[PlacedAsset]:
        try:
            m = re.search(r"\[.*\]", response, re.DOTALL)
            if m:
                raw = json.loads(m.group())
                placed = []
                for item in raw[:10]:
                    if "name" not in item or "description" not in item:
                        continue
                    count = int(item.get("count", 1))
                    positions = item.get("positions", [[0, 0, 0]])[:count]
                    rots = item.get("rotation_z_degrees", [0] * count)[:count]
                    # Pad if LLM gave fewer positions than count
                    while len(positions) < count:
                        positions.append([0, 0, 0])
                    while len(rots) < count:
                        rots.append(0)
                    placed.append(PlacedAsset(
                        name=_slug(item["name"])[:30],
                        description=item["description"],
                        asset_type=item.get("asset_type", "prop"),
                        count=count,
                        positions=positions,
                        scale=float(item.get("scale", 1.0)),
                        rotation_z_degrees=rots,
                        poly_budget=int(item.get("poly_budget", 3000)),
                    ))
                if placed:
                    return placed
        except Exception:
            pass
        # Fallback: minimal world
        return [
            PlacedAsset("terrain_base", f"terrain for {response[:80]}", "terrain",
                        1, [[0, 0, 0]], 1.0, [0], 8000),
        ]

    # ── Stage 3: Assembly script ───────────────────────────────────────────

    def _write_assembly_script(self, world: World) -> Path:
        assets_json = json.dumps([
            {
                "name": pa.name,
                "positions": pa.positions,
                "scale": pa.scale,
                "rotation_z_degrees": pa.rotation_z_degrees,
                "count": pa.count,
            }
            for pa in world.assets
        ], indent=4)

        script = WORLD_ASSEMBLY_SCRIPT.format(
            world_name=world.name,
            world_dir=str(self._data_dir / "assets").replace("\\", "/"),
            asset_count=len(world.assets),
            assets_json=assets_json,
        )
        script_path = world.world_dir / "world_assembly.py"
        script_path.write_text(script)
        return script_path

    def _execute_assembly(self, script_path: Path, world_dir: Path) -> bool:
        log_path = world_dir / "world_assembly.log"
        try:
            result = subprocess.run(
                [self._blender, "--background", "--python", str(script_path)],
                capture_output=True, text=True, timeout=300,
            )
            log_path.write_text(
                f"EXIT CODE: {result.returncode}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            log_path.write_text("ERROR: Blender assembly timed out after 300s")
            return False
        except Exception as e:
            log_path.write_text(f"ERROR: {e}")
            return False

    # ── Stage 5: Engine distribution ──────────────────────────────────────

    def _distribute(self, world: World, engines: list[str]):
        engine_names = {"unreal": "Unreal Engine 5", "unity": "Unity", "godot": "Godot 4"}
        engine_emojis = {"unreal": "🔵", "unity": "⬜", "godot": "🟣"}

        for pa in world.assets:
            if pa.package is None:
                continue
            for engine in engines:
                fn = ENGINE_SETUP.get(engine)
                if not fn:
                    continue
                emoji = engine_emojis.get(engine, "•")
                name = engine_names.get(engine, engine)
                try:
                    out_dir = fn(pa.package, model_client=self._model)
                    rel = out_dir.relative_to(self._data_dir)
                    self._log(f"  {emoji} [{name}] ✓ {pa.name} → {rel}")
                except Exception as e:
                    self._log(f"  {emoji} [{name}] ✗ {pa.name}: {e}")

        # Also write a world-level engine setup guide
        for engine in engines:
            guide_dir = world.world_dir / engine
            guide_dir.mkdir(exist_ok=True)
            self._write_engine_world_guide(world, engine, guide_dir)

    def _write_engine_world_guide(self, world: World, engine: str, guide_dir: Path):
        asset_list = "\n".join(
            f"- `{pa.name}` × {pa.count} ({pa.asset_type})"
            for pa in world.assets
        )
        if engine == "unreal":
            content = f"""# {world.name} — Unreal Engine 5 World Import Guide

## Assets in this world ({len(world.assets)} types)
{asset_list}

## Import Steps
1. Import each asset using its individual `import_{{asset}}.py` script
2. In UE5 Content Browser: File → Import → select `{world.name}.fbx`
3. Use World Partition for large worlds (> 1km)
4. Add Landscape for terrain: Landscape → Import from heightmap
5. For trees: use Foliage tool with imported tree mesh
6. Enable Nanite on all static meshes via right-click → Nanite → Enable

## Recommended Settings
- World Partition: enable for worlds > 200m
- Lumen: enabled by default in UE5
- Auto Exposure: Project Settings → Rendering → Auto Exposure
"""
        elif engine == "unity":
            content = f"""# {world.name} — Unity World Import Guide

## Assets in this world ({len(world.assets)} types)
{asset_list}

## Import Steps
1. Copy each asset GLB into `Assets/Worlds/{world.name}/`
2. Unity auto-imports on file drop
3. Create empty GameObject named "{world.name}_Root"
4. Instantiate each asset prefab at the positions listed in `manifest.json`
5. Add Terrain component for landscape (or use imported terrain mesh)

## Positioning Reference
See `manifest.json` in this world directory for exact positions.

## Recommended Settings
- URP or HDRP pipeline
- Enable GPU Instancing on materials with multiple instances
- Use LOD Groups for trees and rocks
"""
        else:  # godot
            content = f"""# {world.name} — Godot 4 World Import Guide

## Assets in this world ({len(world.assets)} types)
{asset_list}

## Import Steps
1. Copy all GLB files to `res://worlds/{world.name}/`
2. Godot auto-imports meshes
3. Create a new 3D scene: add Node3D as root, name it "{world.name}"
4. For each asset, add a StaticBody3D + MeshInstance3D child
5. Use MultiMeshInstance3D for repeated assets (trees, rocks, barrels)

## GDScript World Loader
```gdscript
extends Node3D

func _ready():
    var manifest = JSON.parse_string(
        FileAccess.open("res://worlds/{world.name}/manifest.json", FileAccess.READ).get_as_text()
    )
    for asset in manifest.assets:
        var scene = load("res://worlds/{world.name}/" + asset.name + "/" + asset.name + ".tscn")
        for pos in asset.positions:
            var inst = scene.instantiate()
            inst.position = Vector3(pos[0], pos[1], pos[2])
            add_child(inst)
```
"""
        (guide_dir / "WORLD_IMPORT_GUIDE.md").write_text(content)

    # ── Summary ───────────────────────────────────────────────────────────

    def _print_summary(self, world: World, engines: list[str]):
        self._log("\n" + "─" * 64)
        self._log(f"  🌍 World: {world.name}")
        self._log(f"  Style:   {world.style}")
        self._log(f"  Size:    {world.size}m × {world.size}m")
        self._log(f"  Assets:  {len(world.assets)} types")
        self._log("")
        total_instances = sum(a.count for a in world.assets)
        for pa in world.assets:
            has_files = bool(pa.package and pa.package.files)
            status = f"{'✓' if has_files else '📄'} {len(pa.package.files) if pa.package else 0} files"
            self._log(f"  • {pa.name:<28} × {pa.count}  [{pa.asset_type}]  {status}")
        self._log(f"\n  Total scene objects: {total_instances}")
        self._log(f"  Assembly script:     {world.assembly_script.relative_to(self._data_dir)}")
        if world.fbx_path:
            self._log(f"  World FBX:          {world.fbx_path.relative_to(self._data_dir)}")
        if world.glb_path:
            self._log(f"  World GLB:          {world.glb_path.relative_to(self._data_dir)}")
        if engines:
            self._log(f"  Engine guides:      {', '.join(engines)}")
        self._log("─" * 64)
        if not world.blender_executed:
            self._log(
                "\n  ℹ️  Blender not found — scripts generated but not executed.\n"
                f"     To assemble world:\n"
                f"     blender --background --python {world.assembly_script}"
            )

    def _save_manifest(self, world: World):
        manifest = {
            "name": world.name,
            "description": world.description,
            "style": world.style,
            "size_m": world.size,
            "blender_executed": world.blender_executed,
            "assets": [
                {
                    "name": pa.name,
                    "description": pa.description,
                    "asset_type": pa.asset_type,
                    "count": pa.count,
                    "positions": pa.positions,
                    "scale": pa.scale,
                    "rotation_z_degrees": pa.rotation_z_degrees,
                }
                for pa in world.assets
            ],
        }
        (world.world_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )


def _slug(text: str) -> str:
    s = re.sub(r"[^\w\s]", "", text.lower())
    s = re.sub(r"\s+", "_", s.strip())
    return s.strip("_")[:40]
