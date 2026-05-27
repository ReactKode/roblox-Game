#!/usr/bin/env python3
"""NexusAgent - Local Self-Learning AI Agent"""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        description="NexusAgent - Local Self-Learning AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Interactive CLI
  python main.py --chat "What is Blender?"
  python main.py --learn "Blender Python scripting"
  python main.py --learn "Unity game development"
  python main.py --learn "Godot GDScript"
  python main.py --status
  python main.py --model ollama           # Force Ollama backend
        """,
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--chat", metavar="MESSAGE", help="Single message (non-interactive)")
    parser.add_argument("--learn", metavar="TOPIC", help="Learn a specific skill")
    parser.add_argument("--status", action="store_true", help="Print agent status and exit")
    parser.add_argument("--skills", action="store_true", help="List learned skills")
    parser.add_argument("--model", help="Override model backend (ollama/lmstudio/openai/anthropic)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose tool call output")
    args = parser.parse_args()

    if args.model:
        import os
        os.environ["NEXUS_MODEL_BACKEND"] = args.model

    try:
        from agent.core.agent import NexusAgent
        agent = NexusAgent(config_path=args.config)
    except Exception as e:
        print(f"Failed to initialize NexusAgent: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.status:
            status = agent.status()
            print(json.dumps(status, indent=2, default=str))

        elif args.skills:
            skills = agent._skills.list_all()
            if not skills:
                print("No skills learned yet.")
            else:
                for s in skills:
                    print(f"[{s['domain']}] {s['name']} — {s['description']}")

        elif args.learn:
            result = agent.learn(args.learn)
            print(result)

        elif args.chat:
            response = agent.chat(args.chat, verbose=args.verbose)
            print(response)

        else:
            agent.run_cli()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        agent.shutdown()


if __name__ == "__main__":
    main()
