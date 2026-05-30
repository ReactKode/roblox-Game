#!/usr/bin/env python3
"""Hermes — Local Self-Learning AI Agent"""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Hermes — Local Self-Learning AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Interactive CLI (recommended)
  python main.py --chat "How do I rig a 3D model in Blender?"
  python main.py --learn "Blender Python bpy scripting"
  python main.py --learn "Godot 4 GDScript game mechanics"
  python main.py --learn "FastAPI Python web server"
  python main.py --status
  python main.py --skills
  python main.py --model ollama               # Force Ollama backend
  python main.py --model anthropic            # Force Anthropic API
  python main.py --world "a medieval village with a castle and market"
  python main.py --world "dark ruins" --world-style "dark fantasy" --world-size 150
        """,
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--chat", metavar="MSG", help="Single message (non-interactive)")
    parser.add_argument("--learn", metavar="TOPIC", help="Learn a skill and exit")
    parser.add_argument("--project", metavar="GOAL", help="Run full Hive pipeline for a project goal")
    parser.add_argument("--asset", metavar="DESC", help="Create 3D asset: Blender script → engine importers")
    parser.add_argument("--engines", metavar="LIST", help="Comma-separated engines for --asset/--world: unreal,unity,godot")
    parser.add_argument("--world", metavar="DESC", help="Build a complete 3D world scene: terrain + buildings + props → engine importers")
    parser.add_argument("--world-style", metavar="STYLE", default="low poly stylized", help="Art style for --world (default: 'low poly stylized')")
    parser.add_argument("--world-size", metavar="METRES", type=float, default=100.0, help="World size in metres for --world (default: 100)")
    parser.add_argument("--status", action="store_true", help="Print status JSON and exit")
    parser.add_argument("--skills", action="store_true", help="List learned skills")
    parser.add_argument("--profile", action="store_true", help="Show agent self-model profile")
    parser.add_argument("--model", help="Force model backend: ollama/lmstudio/openai/anthropic")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show tool calls")
    args = parser.parse_args()

    if args.model:
        import os
        os.environ["NEXUS_MODEL_BACKEND"] = args.model

    try:
        from agent.core.agent import NexusAgent
        agent = NexusAgent(config_path=args.config)
    except RuntimeError as e:
        print(f"\n[ERROR] {e}\n", file=sys.stderr)
        print("Make sure a model backend is running:")
        print("  Ollama:    ollama serve  (then: ollama pull llama3.2)")
        print("  LM Studio: start server on port 1234")
        print("  Cloud:     set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")
        sys.exit(1)
    except Exception as e:
        print(f"Startup error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.status:
            print(json.dumps(agent.status(), indent=2, default=str))
        elif args.skills:
            skills = agent._skills.list_all()
            if not skills:
                print("No skills learned yet.")
            else:
                for s in skills:
                    conf = s.get("confidence", 0.5)
                    print(f"  [{conf:.0%}] [{s['domain']}] {s['name']}")
        elif args.profile:
            p = agent._self_model.profile
            print(json.dumps(p, indent=2))
        elif args.learn:
            result = agent.learn(args.learn)
            print(result)
        elif args.project:
            project = agent.run_project(args.project)
            print(f"\nProject '{project.name}' complete!")
            for role_id, deliv in project.deliverables.items():
                if not role_id.startswith("_"):
                    print(f"  ✓ {deliv.title} ({deliv.word_count} words)")
        elif args.asset:
            engines = [e.strip() for e in args.engines.split(",")] if args.engines else None
            pkgs = agent.create_assets(args.asset, engines)
            for pkg in pkgs:
                print(f"  ✓ {pkg.asset_name}: script={pkg.script_path.name}, "
                      f"exports={len(pkg.files)} files")
        elif args.world:
            engines = [e.strip() for e in args.engines.split(",")] if args.engines else None
            world = agent.build_world(
                args.world,
                style=args.world_style,
                size=args.world_size,
                engines=engines,
            )
            print(f"\nWorld '{world.name}' complete!")
            for pa in world.assets:
                files = len(pa.package.files) if pa.package else 0
                print(f"  • {pa.name} × {pa.count} [{pa.asset_type}]"
                      + (f" — {files} files exported" if files else " — script only"))
            if world.assembly_script:
                print(f"\nTo assemble in Blender:")
                print(f"  blender --background --python {world.assembly_script}")
        elif args.chat:
            agent._verbose = args.verbose
            response = agent.chat(args.chat)
            print(response)
        else:
            agent._verbose = args.verbose
            agent.run_cli()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        agent.shutdown()


if __name__ == "__main__":
    main()
