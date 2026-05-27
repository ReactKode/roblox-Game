#!/usr/bin/env python3
"""NexusAgent v2 — Local Self-Learning AI Agent"""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        description="NexusAgent v2 — Local Self-Learning AI Agent",
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
        """,
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--chat", metavar="MSG", help="Single message (non-interactive)")
    parser.add_argument("--learn", metavar="TOPIC", help="Learn a skill and exit")
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
