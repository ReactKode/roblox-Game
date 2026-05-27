"""Engine-specific importers — generate import scripts, config files, and guides for
Unreal Engine 5, Unity, and Godot from an AssetPackage."""
from pathlib import Path
from .blender_asset import AssetPackage


# ════════════════════════════════════════════════════════════════════════════
#  UNREAL ENGINE 5
# ════════════════════════════════════════════════════════════════════════════

def setup_unreal(pkg: AssetPackage, model_client=None) -> Path:
    """Create everything needed to import this asset into Unreal Engine 5."""
    ue_dir = pkg.asset_dir / "unreal"
    ue_dir.mkdir(exist_ok=True)

    fbx_rel = f"../exports/{pkg.asset_name}.fbx"
    content_path = f"/Game/NexusAssets/{pkg.asset_name.title()}"

    # ── 1. Python import script (run inside UE Editor Python console) ──
    ue_import_py = f'''\
"""
NexusAgent — Unreal Engine 5 Import Script
Asset: {pkg.description}

HOW TO USE:
  1. Copy {pkg.asset_name}.fbx into your UE project Content folder
  2. Open Unreal Engine 5
  3. Go to: Tools → Execute Python Script
  4. Select this file OR paste into Python console
"""
import unreal

ASSET_NAME = "{pkg.asset_name.title()}"
FBX_PATH   = r"{{YOUR_PROJECT_PATH}}\\Content\\NexusAssets\\{pkg.asset_name}.fbx"
DEST_PATH  = "{content_path}"

def import_fbx():
    task = unreal.AssetImportTask()
    task.set_editor_property("filename",         FBX_PATH)
    task.set_editor_property("destination_path", DEST_PATH)
    task.set_editor_property("destination_name", ASSET_NAME)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("automated",        True)
    task.set_editor_property("save",             True)

    opts = unreal.FbxImportUI()
    opts.set_editor_property("import_mesh",      True)
    opts.set_editor_property("import_textures",  True)
    opts.set_editor_property("import_materials", True)
    opts.set_editor_property("import_animations",False)

    fbx_opts = unreal.FbxStaticMeshImportData()
    fbx_opts.set_editor_property("import_translation",
                                  unreal.Vector(0, 0, 0))
    fbx_opts.set_editor_property("import_rotation",
                                  unreal.Rotator(0, 0, 0))
    fbx_opts.set_editor_property("import_uniform_scale", 1.0)
    opts.set_editor_property("static_mesh_import_data", fbx_opts)
    task.set_editor_property("options", opts)

    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    print(f"[NexusAgent] Imported {{ASSET_NAME}} to {{DEST_PATH}}")


def create_material():
    """Create a basic M_{{ASSET_NAME}} material in UE5."""
    mat_path = f"{{DEST_PATH}}/M_{{ASSET_NAME}}"
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    mat_factory = unreal.MaterialFactoryNew()
    material = asset_tools.create_asset(
        f"M_{{ASSET_NAME}}", DEST_PATH, unreal.Material, mat_factory
    )
    if material:
        unreal.EditorAssetLibrary.save_asset(mat_path)
        print(f"[NexusAgent] Created material: M_{{ASSET_NAME}}")


def place_in_level():
    """Spawn the asset in the current level at origin."""
    mesh_path = f"{{DEST_PATH}}/{{ASSET_NAME}}.{{ASSET_NAME}}"
    mesh = unreal.load_asset(mesh_path)
    if mesh:
        loc = unreal.Vector(0, 0, 0)
        rot = unreal.Rotator(0, 0, 0)
        actor = unreal.EditorLevelLibrary.spawn_actor_from_object(mesh, loc, rot)
        print(f"[NexusAgent] Placed {{ASSET_NAME}} in level: {{actor.get_name()}}")
    else:
        print(f"[NexusAgent] Could not find mesh at {{mesh_path}} — import first")


if __name__ == "__main__":
    import_fbx()
    create_material()
    place_in_level()
'''
    (ue_dir / f"import_{pkg.asset_name}.py").write_text(ue_import_py)

    # ── 2. Nanite enable script ────────────────────────────────────────
    nanite_py = f'''\
"""Enable Nanite on {pkg.asset_name.title()} static mesh."""
import unreal

mesh = unreal.load_asset("{content_path}/{pkg.asset_name.title()}")
if mesh:
    mesh.set_editor_property("nanite_settings",
        unreal.MeshNaniteSettings(enabled=True, fallback_percent_triangles=1.0, fallback_relative_error=1.0))
    unreal.EditorAssetLibrary.save_asset("{content_path}/{pkg.asset_name.title()}")
    print("[NexusAgent] Nanite enabled on {pkg.asset_name.title()}")
'''
    (ue_dir / f"enable_nanite_{pkg.asset_name}.py").write_text(nanite_py)

    # ── 3. Lumen material helper ───────────────────────────────────────
    lumen_py = f'''\
"""Create a Lumen-compatible PBR material for {pkg.asset_name.title()} in UE5."""
import unreal

# This script assumes you have run import first
# Add a Metallic/Roughness texture workflow for Lumen
print("[NexusAgent] Lumen materials require manual connection in Material Editor.")
print("Recommended workflow:")
print("  1. Create Texture Sample nodes for Albedo, Normal, Roughness, Metallic")
print("  2. Connect to Principled Material inputs")
print("  3. Enable: Project Settings → Rendering → Global Illumination: Lumen")
'''
    (ue_dir / f"lumen_material_{pkg.asset_name}.py").write_text(lumen_py)

    # ── 4. Guide ──────────────────────────────────────────────────────
    _write_guide(ue_dir / "IMPORT_GUIDE.md", "Unreal Engine 5", pkg, _ue_guide(pkg))

    return ue_dir


def _ue_guide(pkg: AssetPackage) -> str:
    has_file = any(f.format == "fbx" for f in pkg.files)
    file_note = (
        f"`{pkg.asset_name}.fbx` is ready in `exports/`"
        if has_file else
        f"Copy your FBX file as `{pkg.asset_name}.fbx` into your UE Content folder."
    )
    return f"""
## Quick Setup (3 steps)

1. **Copy asset** — {file_note}
   Place it at: `YourProject/Content/NexusAssets/{pkg.asset_name}.fbx`

2. **Import** — In UE5 Editor:
   - `Tools → Execute Python Script → unreal/import_{pkg.asset_name}.py`
   - OR drag-drop the FBX directly into Content Browser

3. **Enable Nanite** (optional, UE5 only):
   - `Tools → Execute Python Script → unreal/enable_nanite_{pkg.asset_name}.py`

## Recommended Settings
| Setting | Value |
|---|---|
| Import as | Static Mesh |
| Generate Lightmap UVs | ✓ |
| Auto Generate Collision | ✓ |
| Nanite | Enabled (for hero assets) |
| LODs | Auto-generated |
"""


# ════════════════════════════════════════════════════════════════════════════
#  UNITY
# ════════════════════════════════════════════════════════════════════════════

def setup_unity(pkg: AssetPackage, model_client=None) -> Path:
    """Create Unity import scripts and folder structure."""
    unity_dir = pkg.asset_dir / "unity"
    (unity_dir / "Assets" / "NexusAssets" / "Models").mkdir(parents=True, exist_ok=True)
    (unity_dir / "Assets" / "NexusAssets" / "Materials").mkdir(parents=True, exist_ok=True)
    (unity_dir / "Assets" / "NexusAssets" / "Prefabs").mkdir(parents=True, exist_ok=True)
    (unity_dir / "Editor").mkdir(exist_ok=True)

    pascal = _pascal(pkg.asset_name)

    # ── 1. Editor import script (C#) ──────────────────────────────────
    editor_cs = f'''\
// NexusAgent — Unity Editor Import Script
// Asset: {pkg.description}
// Place this file in: Assets/Editor/
// Access via: NexusAgent menu in Unity toolbar

using UnityEngine;
using UnityEditor;
using System.IO;

public class {pascal}Importer : EditorWindow
{{
    private const string ASSET_NAME    = "{pkg.asset_name}";
    private const string ASSET_PATH    = "Assets/NexusAssets/Models/{pkg.asset_name}.fbx";
    private const string MATERIAL_PATH = "Assets/NexusAssets/Materials/M_{pkg.asset_name}.mat";
    private const string PREFAB_PATH   = "Assets/NexusAssets/Prefabs/{pascal}.prefab";

    [MenuItem("NexusAgent/Import {pascal}")]
    public static void ImportAsset()
    {{
        if (!File.Exists(Path.Combine(Application.dataPath,
            "NexusAssets/Models/{pkg.asset_name}.fbx")))
        {{
            Debug.LogError("[NexusAgent] FBX not found at " + ASSET_PATH +
                           ". Copy the .fbx there first.");
            return;
        }}

        // Force reimport with proper settings
        AssetDatabase.ImportAsset(ASSET_PATH, ImportAssetOptions.ForceUpdate);

        // Configure import settings
        var importer = AssetImporter.GetAtPath(ASSET_PATH) as ModelImporter;
        if (importer != null)
        {{
            importer.globalScale          = 1.0f;
            importer.meshCompression      = ModelImporterMeshCompression.Off;
            importer.isReadable           = true;
            importer.generateSecondaryUV  = true;  // Lightmap UVs
            importer.optimizeMeshPolygons = true;
            importer.optimizeMeshVertices = true;
            importer.SaveAndReimport();
        }}

        CreateMaterial();
        CreatePrefab();
        AssetDatabase.Refresh();
        Debug.Log("[NexusAgent] {pascal} imported successfully!");
    }}

    static void CreateMaterial()
    {{
        var mat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        mat.name = "M_{pkg.asset_name}";

        // Default PBR values — tweak in Inspector
        mat.SetFloat("_Metallic",   0.0f);
        mat.SetFloat("_Smoothness", 0.5f);

        AssetDatabase.CreateAsset(mat, MATERIAL_PATH);
        Debug.Log("[NexusAgent] Material created: " + MATERIAL_PATH);
    }}

    static void CreatePrefab()
    {{
        var mesh = AssetDatabase.LoadAssetAtPath<GameObject>(ASSET_PATH);
        if (mesh == null) {{ Debug.LogWarning("Mesh not found for prefab."); return; }}

        var instance = (GameObject)PrefabUtility.InstantiatePrefab(mesh);
        instance.name = "{pascal}";

        // Assign material to all renderers
        var mat = AssetDatabase.LoadAssetAtPath<Material>(MATERIAL_PATH);
        foreach (var r in instance.GetComponentsInChildren<Renderer>())
            r.sharedMaterial = mat;

        PrefabUtility.SaveAsPrefabAsset(instance, PREFAB_PATH);
        DestroyImmediate(instance);
        Debug.Log("[NexusAgent] Prefab created: " + PREFAB_PATH);
    }}
}}
'''
    (unity_dir / "Editor" / f"{pascal}Importer.cs").write_text(editor_cs)

    # ── 2. Runtime script (attach to any GameObject) ───────────────────
    runtime_cs = f'''\
// NexusAgent — {pascal} runtime controller
// Attach to the {pascal} prefab GameObject

using UnityEngine;

public class {pascal}Controller : MonoBehaviour
{{
    [Header("NexusAgent Asset: {pkg.description}")]
    [SerializeField] private bool enablePhysics  = false;
    [SerializeField] private bool castShadows    = true;

    void Start()
    {{
        var rb = GetComponent<Rigidbody>();
        if (enablePhysics && rb == null)
            gameObject.AddComponent<Rigidbody>();

        foreach (var r in GetComponentsInChildren<Renderer>())
            r.shadowCastingMode = castShadows
                ? UnityEngine.Rendering.ShadowCastingMode.On
                : UnityEngine.Rendering.ShadowCastingMode.Off;

        Debug.Log($"[NexusAgent] {{gameObject.name}} initialized.");
    }}
}}
'''
    (unity_dir / "Assets" / "NexusAssets" / f"{pascal}Controller.cs").write_text(runtime_cs)

    # ── 3. .meta stub (prevents missing meta warnings) ─────────────────
    (unity_dir / "Assets" / "NexusAssets" / "Models" / f"{pkg.asset_name}.fbx.meta").write_text(
        f"fileFormatVersion: 2\nguid: {_fake_guid(pkg.asset_name)}\n"
        "ModelImporter:\n  serializedVersion: 22200\n"
        "  internalIDToNameTable: []\n  meshes:\n    lODScreenPercentages: []\n"
    )

    # ── 4. Guide ──────────────────────────────────────────────────────
    _write_guide(unity_dir / "IMPORT_GUIDE.md", "Unity", pkg, _unity_guide(pkg, pascal))

    return unity_dir


def _unity_guide(pkg: AssetPackage, pascal: str) -> str:
    return f"""
## Quick Setup

1. **Copy the .fbx file:**
   ```
   cp exports/{pkg.asset_name}.fbx unity/Assets/NexusAssets/Models/
   ```

2. **Copy Editor script** into your Unity project:
   ```
   cp unity/Editor/{pascal}Importer.cs YourUnityProject/Assets/Editor/
   ```

3. **Import via menu** — After Unity recompiles:
   `NexusAgent → Import {pascal}`

4. **Prefab is ready** at `Assets/NexusAssets/Prefabs/{pascal}.prefab` — drag into scene.

## Render Pipeline
Works with both URP and HDRP. Change shader in Material if using HDRP:
`HDRP/Lit` instead of `Universal Render Pipeline/Lit`
"""


# ════════════════════════════════════════════════════════════════════════════
#  GODOT 4
# ════════════════════════════════════════════════════════════════════════════

def setup_godot(pkg: AssetPackage, model_client=None) -> Path:
    """Create Godot 4 import files and scene."""
    godot_dir = pkg.asset_dir / "godot"
    (godot_dir / "assets").mkdir(parents=True, exist_ok=True)
    (godot_dir / "scenes").mkdir(exist_ok=True)
    (godot_dir / "scripts").mkdir(exist_ok=True)

    pascal = _pascal(pkg.asset_name)
    uid_asset = _fake_uid(pkg.asset_name + "_asset")
    uid_scene = _fake_uid(pkg.asset_name + "_scene")

    # ── 1. .import file (Godot auto-import settings) ───────────────────
    import_file = f'''\
[remap]

importer="scene"
importer_version=1
type="PackedScene"
uid="{uid_asset}"
path="res://.godot/imported/{pkg.asset_name}.glb-{uid_asset.replace('uid://', '')}.scn"

[deps]

source_file="res://assets/{pkg.asset_name}.glb"
dest_files=["res://.godot/imported/{pkg.asset_name}.glb-{uid_asset.replace('uid://', '')}.scn"]

[params]

force_clean_import=false
bake_fps=30.0
root_type="Node3D"
root_name="{pascal}"
nodes/apply_root_scale=true
nodes/root_scale=1.0
meshes/ensure_tangents=true
meshes/generate_lods=true
meshes/create_shadow_meshes=true
meshes/light_baking=1
meshes/lightmap_texel_size=0.2
skins/use_named_skins=true
animation/import=true
animation/fps=30
animation/trimming=false
animation/remove_immutable_tracks=true
import_script/path=""
_subresources={{}}
'''
    (godot_dir / "assets" / f"{pkg.asset_name}.glb.import").write_text(import_file)

    # ── 2. Scene file (.tscn) ─────────────────────────────────────────
    tscn = f'''\
[gd_scene load_steps=3 format=3 uid="{uid_scene}"]

[ext_resource type="PackedScene" uid="{uid_asset}" path="res://assets/{pkg.asset_name}.glb" id="1_{pkg.asset_name}"]

[sub_resource type="StandardMaterial3D" id="mat_{pkg.asset_name}"]
roughness = 0.5
metallic = 0.0
metallic_specular = 0.5

[node name="{pascal}" type="Node3D"]

[node name="Mesh" type="MeshInstance3D" parent="."]
mesh = ExtResource("1_{pkg.asset_name}")
surface_material_override/0 = SubResource("mat_{pkg.asset_name}")

[node name="StaticBody3D" type="StaticBody3D" parent="."]

[node name="CollisionShape3D" type="CollisionShape3D" parent="StaticBody3D"]
'''
    (godot_dir / "scenes" / f"{pkg.asset_name}.tscn").write_text(tscn)

    # ── 3. GDScript controller ─────────────────────────────────────────
    gdscript = f'''\
# NexusAgent — {pascal} controller script
# Asset: {pkg.description}
# Attach to the root Node3D in {pkg.asset_name}.tscn

extends Node3D

@export var rotation_speed: float = 0.0   ## Rotate asset (debug/showcase)
@export var cast_shadow: bool = true

func _ready() -> void:
\t_apply_shadow_settings()
\tprint("[NexusAgent] {pascal} ready.")

func _process(delta: float) -> void:
\tif rotation_speed > 0.0:
\t\trotate_y(rotation_speed * delta)

func _apply_shadow_settings() -> void:
\tfor child in find_children("*", "MeshInstance3D", true, false):
\t\tchild.cast_shadow = (
\t\t\tGeometryInstance3D.SHADOW_CASTING_SETTING_ON if cast_shadow
\t\t\telse GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
\t\t)
'''
    (godot_dir / "scripts" / f"{pkg.asset_name}.gd").write_text(gdscript)

    # ── 4. Autoload / project.godot snippet ───────────────────────────
    project_snippet = f'''\
# Add these lines to your project.godot if you want autoloading:
#
# [autoload]
# {pascal}Manager="res://scripts/{pkg.asset_name}.gd"
#
# Or simply drag the .tscn file from the FileSystem dock into your main scene.
'''
    (godot_dir / "PROJECT_SNIPPET.txt").write_text(project_snippet)

    # ── 5. Guide ──────────────────────────────────────────────────────
    _write_guide(godot_dir / "IMPORT_GUIDE.md", "Godot 4", pkg, _godot_guide(pkg, pascal))

    return godot_dir


def _godot_guide(pkg: AssetPackage, pascal: str) -> str:
    return f"""
## Quick Setup (Godot 4)

1. **Copy the .glb file** into your Godot project:
   ```
   cp exports/{pkg.asset_name}.glb YourGodotProject/assets/
   ```
   Copy the import file too:
   ```
   cp godot/assets/{pkg.asset_name}.glb.import YourGodotProject/assets/
   ```

2. **Godot auto-imports** on next editor refresh (it detects the .import file).

3. **Use the scene** — copy `godot/scenes/{pkg.asset_name}.tscn` to your project:
   ```
   cp godot/scenes/{pkg.asset_name}.tscn YourGodotProject/scenes/
   ```
   Then drag it into your main scene from the FileSystem panel.

4. **Attach the script** (optional):
   ```
   cp godot/scripts/{pkg.asset_name}.gd YourGodotProject/scripts/
   ```
   Then attach it to the `{pascal}` node in the Inspector.

## Notes
- Godot 4 natively supports GLB — no plugins needed
- CollisionShape3D is included but unsized — resize to fit mesh in editor
- StandardMaterial3D can be swapped for ORMMaterial3D for PBR texture maps
"""


# ════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ════════════════════════════════════════════════════════════════════════════

def _write_guide(path: Path, engine: str, pkg: AssetPackage, body: str):
    header = (
        f"# {engine} Import Guide — {pkg.asset_name.replace('_', ' ').title()}\n\n"
        f"> **Asset:** {pkg.description}  \n"
        f"> **Blender script:** `blender/create_{pkg.asset_name}.py`  \n"
        f"> **Blender available at build time:** {'Yes ✓' if pkg.blender_available else 'No — script only'}\n"
    )
    path.write_text(header + body)


def _pascal(slug: str) -> str:
    return "".join(w.capitalize() for w in slug.split("_"))


def _fake_guid(seed: str) -> str:
    """Deterministic-looking Unity GUID from seed."""
    import hashlib
    h = hashlib.md5(seed.encode()).hexdigest()
    return h


def _fake_uid(seed: str) -> str:
    """Godot-style UID."""
    import hashlib
    h = hashlib.md5(seed.encode()).hexdigest()[:12]
    return f"uid://{h}"
