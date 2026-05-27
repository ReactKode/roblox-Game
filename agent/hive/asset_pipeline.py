"""Asset Pipeline — orchestrates Blender asset creation and engine distribution.

Flow:
  1. Plan     — LLM decides what assets to create from a description
  2. Create   — BlenderAssetAgent generates script + runs headlessly
  3. Distribute — Engine importers set up Unreal / Unity / Godot simultaneously
  4. Report   — Summary of what was created and where to find it
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .blender_asset import BlenderAssetAgent, AssetPackage
from .engine_importers import setup_unreal, setup_unity, setup_godot

# Engines we support
ENGINE_SETUP = {
    "unreal": setup_unreal,
    "unity":  setup_unity,
    "godot":  setup_godot,
}

PLAN_PROMPT = """\
You are a 3D Art Director planning game assets.

Request: "{request}"
Target engines: {engines}

Break this into a list of individual 3D assets to create.
For each asset give:
- A precise name (snake_case, no spaces)
- A detailed description for the Blender artist
- The asset type (weapon | character | prop | environment | vehicle | ui_element)

Output ONLY this JSON array:
[
  {{
    "name": "asset_name",
    "description": "detailed Blender-friendly description of shape, style, materials",
    "type": "prop"
  }}
]

Limit to 4 assets maximum. Prioritize the most important ones.\
"""

ENGINE_ALIASES = {
    "ue": "unreal", "ue5": "unreal", "unreal engine": "unreal", "unreal engine 5": "unreal",
    "unity3d": "unity", "unity 3d": "unity",
    "godot": "godot", "godot4": "godot", "godot 4": "godot", "grot": "godot", "grout": "godot",
}


class AssetPipelineAgent:
    def __init__(self, model_client, data_dir: Path):
        self._model = model_client
        self._data_dir = data_dir
        self._blender = BlenderAssetAgent(model_client, data_dir)
        self._print_fn = print

    def set_print(self, fn):
        self._print_fn = fn

    def _log(self, msg: str):
        self._print_fn(msg)

    # ── Main entry ────────────────────────────────────────────────────────

    def run(self, request: str, engines: list[str] = None) -> list[AssetPackage]:
        """
        Full pipeline: plan → create → distribute to engines.
        Returns list of AssetPackages, one per asset.
        """
        engines = self._resolve_engines(engines or ["unreal", "unity", "godot"])

        # 1. Plan
        self._log(f"\n[AssetPipeline] Planning assets for: \"{request}\"")
        self._log(f"[AssetPipeline] Target engines: {', '.join(engines)}")
        asset_plans = self._plan(request, engines)
        self._log(f"[AssetPipeline] Assets to build: {[p['name'] for p in asset_plans]}\n")

        blender_ok = self._blender.blender_available
        if blender_ok:
            self._log(f"[AssetPipeline] Blender found — will execute scripts headlessly")
        else:
            self._log(f"[AssetPipeline] Blender not found — generating scripts only (install Blender to run)")

        # 2. Create each asset (sequential — each is self-contained)
        packages: list[AssetPackage] = []
        for plan in asset_plans:
            self._log(f"\n🖌️  [Blender Artist] Creating '{plan['name']}' ({plan['type']})...")
            pkg = self._blender.create(
                description=plan["description"],
                asset_name=plan["name"],
                asset_type=plan["type"],
            )
            status = f"✓ executed ({len(pkg.files)} files exported)" if pkg.files else "✓ script generated (no Blender)"
            self._log(f"🖌️  [Blender Artist] {status}")
            packages.append(pkg)

        # 3. Distribute to engines (parallel per engine across all assets)
        self._log("\n[AssetPipeline] Distributing to engines...")
        self._distribute(packages, engines)

        # 4. Report
        self._log("\n[AssetPipeline] ✓ Asset pipeline complete!")
        self._print_summary(packages, engines)

        # 5. Store in agent memory if available
        self._store_memory(request, packages, engines)

        return packages

    # ── Stage 1: Plan ─────────────────────────────────────────────────────

    def _plan(self, request: str, engines: list[str]) -> list[dict]:
        prompt = PLAN_PROMPT.format(request=request, engines=", ".join(engines))
        response = self._model.chat([
            {"role": "system", "content": "You are a 3D Art Director. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ])
        try:
            m = re.search(r"\[.*\]", response, re.DOTALL)
            if m:
                plans = json.loads(m.group())
                # Validate each plan has required keys
                valid = []
                for p in plans[:4]:
                    if "name" in p and "description" in p:
                        p.setdefault("type", "prop")
                        valid.append(p)
                if valid:
                    return valid
        except Exception:
            pass
        # Fallback: single asset from the request
        return [{"name": re.sub(r"[^\w]", "_", request.lower())[:30],
                 "description": request,
                 "type": "prop"}]

    # ── Stage 2: Distribute ───────────────────────────────────────────────

    def _distribute(self, packages: list[AssetPackage], engines: list[str]):
        engine_emojis = {"unreal": "🔵", "unity": "⬜", "godot": "🟣"}
        engine_names  = {"unreal": "Unreal Engine 5", "unity": "Unity", "godot": "Godot 4"}

        def run_engine(engine: str, pkg: AssetPackage):
            fn = ENGINE_SETUP.get(engine)
            if not fn:
                return
            emoji = engine_emojis.get(engine, "•")
            name  = engine_names.get(engine, engine)
            try:
                out_dir = fn(pkg, model_client=self._model)
                self._log(f"  {emoji} [{name}] ✓ {pkg.asset_name} — {out_dir.relative_to(self._data_dir)}")
            except Exception as e:
                self._log(f"  {emoji} [{name}] ✗ {pkg.asset_name} failed: {e}")

        # Run all (asset × engine) combinations in parallel
        tasks = [(engine, pkg) for pkg in packages for engine in engines]
        with ThreadPoolExecutor(max_workers=min(len(tasks), 6)) as pool:
            futures = [pool.submit(run_engine, engine, pkg) for engine, pkg in tasks]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception:
                    pass

    # ── Summary ───────────────────────────────────────────────────────────

    def _print_summary(self, packages: list[AssetPackage], engines: list[str]):
        self._log("\n" + "─" * 60)
        for pkg in packages:
            self._log(f"  📦 {pkg.asset_name}")
            self._log(f"     Script:  {pkg.script_path.relative_to(self._data_dir)}")
            for f in pkg.files:
                self._log(f"     Export:  {f.path.relative_to(self._data_dir)} ({f.format.upper()})")
            for engine in engines:
                engine_dir = pkg.asset_dir / engine
                if engine_dir.exists():
                    guide = engine_dir / "IMPORT_GUIDE.md"
                    self._log(f"     {engine.title():8} → {engine_dir.relative_to(self._data_dir)}/")
        self._log("─" * 60)
        if not packages[0].blender_available:
            self._log(
                "\n  ℹ️  Blender not found — scripts generated but not executed.\n"
                "     To create real .fbx/.glb files:\n"
                "     1. Install Blender (blender.org)\n"
                "     2. Run: blender --background --python <script_path>"
            )

    def _store_memory(self, request: str, packages: list[AssetPackage], engines: list[str]):
        pass  # Will be wired to agent memory when called from NexusAgent

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_engines(raw: list[str]) -> list[str]:
        resolved = []
        for e in raw:
            e_lower = e.lower().strip()
            canonical = ENGINE_ALIASES.get(e_lower, e_lower)
            if canonical in ENGINE_SETUP and canonical not in resolved:
                resolved.append(canonical)
        return resolved or ["unreal", "unity", "godot"]
