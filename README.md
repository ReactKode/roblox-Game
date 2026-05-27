# NexusAgent

**A local, self-learning AI agent with persistent memory, skill acquisition, and multi-backend model support.**

NexusAgent runs on your machine, learns new skills by researching the web, remembers things across sessions, and can use local models (Ollama, LM Studio) or cloud APIs.

---

## Features

| Feature | Description |
|---|---|
| **Self-Learning** | `/learn Blender scripting` — agent researches docs, stores skill |
| **Persistent Memory** | Short-term context + ChromaDB long-term vector store + SQLite episode log |
| **Heartbeat Daemon** | Background thread consolidates memory, writes status every 5 min |
| **Local-First** | Ollama + LM Studio supported; cloud APIs optional |
| **Tool Use** | Web search, web scraping, Python execution, file ops |
| **ReAct Loop** | Think → Tool → Observe → Repeat until final answer |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up a model backend (pick one)

**Ollama (recommended for local):**
```bash
# Install: https://ollama.com
ollama pull llama3.2
ollama serve
```

**LM Studio:**
Download from [lmstudio.ai](https://lmstudio.ai), load a model, start the local server on port 1234.

**Cloud API:**
```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY or OPENAI_API_KEY
```

### 3. Run

```bash
python main.py                            # Interactive CLI
python main.py --learn "Blender Python"   # Learn a skill
python main.py --chat "How do I rig a 3D model?"
python main.py --status                   # Agent status
```

---

## CLI Commands

In interactive mode:

| Command | Description |
|---|---|
| `/learn <topic>` | Research and learn a skill (Blender, Unity, Godot, etc.) |
| `/skills` | List all learned skills |
| `/skill <name>` | Show a skill's procedure and code template |
| `/status` | Agent uptime, model, memory stats |
| `/memory` | Memory statistics |
| `/episodes` | Recent action history |
| `/clear` | Clear short-term memory |
| `/model` | Show active model backend |
| `/help` | Show all commands |
| `/quit` | Exit |

---

## Architecture

```
NexusAgent
├── Models Layer          Ollama → LM Studio → OpenAI → Anthropic (auto-fallback)
├── Memory System
│   ├── Short-term        Sliding context window (last 20 messages)
│   ├── Long-term         ChromaDB vector store (semantic search)
│   └── Episodic          SQLite log of every action + outcome
├── Skill Library
│   ├── Registry          JSON files per skill in data/skills/
│   └── Learner           Auto-researches docs when learning new skill
├── Tool Registry
│   ├── web_search        DuckDuckGo
│   ├── web_scrape        requests + BeautifulSoup
│   ├── run_python        Subprocess sandbox
│   ├── run_shell         Shell commands
│   ├── read/write_file   File system
│   ├── remember          Store in long-term memory
│   ├── recall_memory     Search memories
│   ├── learn_skill       Trigger skill learner
│   └── recall_skill      Retrieve learned skill
├── ReAct Loop            Think → Tool → Observe → Final Answer
├── Task Planner          Decomposes complex tasks into subtasks
└── Heartbeat             Background daemon (memory consolidation, status)
```

---

## Skill Domains

NexusAgent auto-detects these domains when learning:

- `blender` — Blender Python API (bpy), 3D modeling, mesh operations
- `unity` — Unity C# scripting, MonoBehaviour, game systems
- `unreal_engine` — UE5 Blueprints, C++, game development
- `godot` — GDScript, Godot 4 API
- `game_dev` — Game design, mechanics, level design
- `python` — Python libraries, patterns, frameworks
- `javascript` — JS/TS, Node.js, React
- `machine_learning` — PyTorch, TensorFlow, model training
- `business` — Startup strategy, SaaS, marketing
- `research` — Information gathering, analysis
- `web` — HTML/CSS, REST APIs, web scraping

---

## Data Storage

All persistent data is in `./data/` (gitignored):

```
data/
├── status.json           Heartbeat status (updated every 5 min)
├── memory/
│   ├── chroma/           ChromaDB vector store
│   ├── vectors.json      Fallback JSON vector store
│   └── episodes.db       SQLite episodic memory
└── skills/
    ├── blender_python_scripting.json
    ├── unity_game_development.json
    └── ...
```

---

## Configuration

Edit `config.yaml` to customize:

```yaml
models:
  preferred_backend: "auto"   # auto | ollama | lmstudio | openai | anthropic
  ollama:
    model: "llama3.2"         # or mistral, codellama, etc.

heartbeat:
  interval: 300               # seconds between heartbeats

memory:
  short_term_limit: 20
  long_term_backend: "chromadb"
```

---

## Example Session

```
nexus> /learn Blender Python scripting
Researching 'Blender Python scripting'...
Learned skill: Blender Python scripting
Domain: blender

nexus> How do I create a cube in Blender with Python?
nexus> I'll use the learned Blender skill to answer...

To create a cube in Blender using Python (bpy):

import bpy
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
obj = bpy.context.active_object
obj.name = "MyCube"

nexus> /skills
Learned Skills (1):
• [blender] Blender Python scripting — Skills and techniques for Blender Python scripting
```
