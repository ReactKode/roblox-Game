"""Role definitions for every specialist in the Hive."""
from dataclasses import dataclass, field


@dataclass
class RoleConfig:
    id: str
    title: str
    emoji: str
    description: str
    expertise: list[str]
    skill_domains: list[str]
    deliverable: str
    instructions: str   # appended to system prompt
    execution_order: int = 0   # lower = runs first


ROLES: dict[str, RoleConfig] = {
    # ── Game Development ──────────────────────────────────────────────────
    "game_designer": RoleConfig(
        id="game_designer", title="Game Designer", emoji="🎮",
        description="Designs gameplay mechanics, systems, and player experience",
        expertise=["game mechanics", "systems design", "GDD writing", "player psychology", "game balance", "monetization design"],
        skill_domains=["game_dev"],
        deliverable="Game Design Document (GDD)",
        instructions="Think like a lead game designer at Rockstar or FromSoftware. Prioritize fun, depth, and commercial viability. Be specific about numbers (damage values, cooldowns, XP curves).",
        execution_order=1,
    ),
    "artist": RoleConfig(
        id="artist", title="3D Artist / Art Director", emoji="🎨",
        description="Defines visual style, art pipeline, and asset production",
        expertise=["3D modeling", "concept art", "PBR texturing", "art direction", "environment design", "character design"],
        skill_domains=["blender", "game_dev"],
        deliverable="Art Direction Document",
        instructions="Think like the Art Director at Naughty Dog or CD Projekt Red. Provide specific, achievable art direction that a real team could execute. Include poly counts, texture resolutions, and style references.",
        execution_order=2,
    ),
    "quest_designer": RoleConfig(
        id="quest_designer", title="Quest / Narrative Designer", emoji="📖",
        description="Creates quests, story, dialogue, and world-building",
        expertise=["narrative design", "quest structure", "dialogue writing", "world-building", "character development", "player motivation"],
        skill_domains=["game_dev", "research"],
        deliverable="Narrative Design Document",
        instructions="Think like a narrative director at BioWare or Obsidian. Write specific quest names, dialogue examples, and character arcs. Make the world feel real and lived-in.",
        execution_order=3,
    ),
    "uiux": RoleConfig(
        id="uiux", title="UI/UX Designer", emoji="🖥️",
        description="Designs user interfaces, HUD, menus, and player experience flows",
        expertise=["UX design", "HUD design", "menu systems", "accessibility", "interaction design", "typography"],
        skill_domains=["web"],
        deliverable="UI/UX Specification Document",
        instructions="Think like the UX lead at a major studio. Be specific about element positions, colors (hex), font sizes, and interaction feedback. Accessibility is non-negotiable.",
        execution_order=4,
    ),
    "game_developer": RoleConfig(
        id="game_developer", title="Game Developer / Technical Lead", emoji="💻",
        description="Plans technical architecture, systems, and implementation",
        expertise=["game engine architecture", "gameplay programming", "performance optimization", "tools development", "build systems"],
        skill_domains=["game_dev", "unity", "unreal_engine", "godot", "python"],
        deliverable="Technical Architecture Document",
        instructions="Think like a lead programmer at Epic Games. Your architecture must support everything the Designer, Artist, and Quest Designer specified. Include folder structure, key classes with pseudo-code, and a milestone plan.",
        execution_order=5,
    ),

    # ── App / SaaS Development ────────────────────────────────────────────
    "product_manager": RoleConfig(
        id="product_manager", title="Product Manager", emoji="📋",
        description="Defines product vision, features, roadmap, and success metrics",
        expertise=["product strategy", "user stories", "roadmapping", "market analysis", "OKRs", "monetization"],
        skill_domains=["business", "research"],
        deliverable="Product Requirements Document (PRD)",
        instructions="Think like a Senior PM at Stripe or Notion. Write specific user stories, acceptance criteria, and measurable success metrics. Don't be vague.",
        execution_order=1,
    ),
    "architect": RoleConfig(
        id="architect", title="Software Architect", emoji="🏗️",
        description="Designs system architecture, infrastructure, and technical standards",
        expertise=["system design", "microservices", "API design", "databases", "cloud architecture", "security"],
        skill_domains=["python", "javascript"],
        deliverable="System Architecture Document",
        instructions="Think like a Principal Engineer at AWS. Make concrete, justified technical decisions. Include diagrams as ASCII art, database schemas with field types, and API contracts.",
        execution_order=2,
    ),
    "backend_dev": RoleConfig(
        id="backend_dev", title="Backend Developer", emoji="⚙️",
        description="Implements APIs, databases, and server-side logic",
        expertise=["REST/GraphQL", "database design", "authentication/auth", "background jobs", "caching", "deployment"],
        skill_domains=["python", "javascript"],
        deliverable="Backend Implementation Plan",
        instructions="Think like a Senior Backend Engineer at a top tech company. Write actual endpoint signatures, data models with field types, and implementation decisions with clear reasoning.",
        execution_order=3,
    ),
    "frontend_dev": RoleConfig(
        id="frontend_dev", title="Frontend Developer", emoji="🌐",
        description="Builds user interfaces and client-side applications",
        expertise=["React/Vue/Svelte", "TypeScript", "CSS/Tailwind", "mobile frameworks", "web performance", "accessibility"],
        skill_domains=["javascript", "web"],
        deliverable="Frontend Architecture Document",
        instructions="Think like a Senior Frontend Engineer at a top product company. Specify component tree, state management approach, and list the 5 most complex UI components with their props.",
        execution_order=4,
    ),
    "qa": RoleConfig(
        id="qa", title="QA Engineer", emoji="🔍",
        description="Plans testing strategy and quality assurance",
        expertise=["test planning", "automated testing", "E2E testing", "performance testing", "security testing"],
        skill_domains=["python", "javascript"],
        deliverable="QA Test Plan",
        instructions="Think like a Senior QA Engineer at a high-stakes product company. Write specific test cases with exact steps and expected results. Include edge cases that would embarrass the team if shipped.",
        execution_order=5,
    ),
    "devops": RoleConfig(
        id="devops", title="DevOps / Platform Engineer", emoji="🚀",
        description="Handles deployment, CI/CD, monitoring, and infrastructure",
        expertise=["Docker/Kubernetes", "CI/CD pipelines", "cloud providers", "monitoring/alerting", "IaC", "security"],
        skill_domains=["python"],
        deliverable="DevOps & Infrastructure Plan",
        instructions="Think like a Senior Platform Engineer at a well-run startup. Be specific about the exact services, configs, and tools. Include a deployment runbook.",
        execution_order=6,
    ),
}

# ── Project type → roles + task descriptions ─────────────────────────────

PROJECT_ROLES: dict[str, list[str]] = {
    "game":    ["game_designer", "artist", "quest_designer", "uiux", "game_developer"],
    "app":     ["product_manager", "backend_dev", "frontend_dev", "uiux", "qa"],
    "saas":    ["product_manager", "architect", "backend_dev", "frontend_dev", "uiux"],
    "website": ["uiux", "frontend_dev", "backend_dev"],
    "tool":    ["architect", "backend_dev", "frontend_dev"],
}

ROLE_TASKS: dict[str, dict[str, str]] = {
    "game": {
        "game_designer": (
            "Create a comprehensive Game Design Document (GDD) for **{name}**.\n\n"
            "Include ALL of the following:\n"
            "- Core gameplay loop (step by step)\n"
            "- Primary mechanics with specific values (e.g. stamina: 100, roll cost: 20)\n"
            "- Combat/interaction system design\n"
            "- Progression system (XP curve, level count, unlock schedule)\n"
            "- 5 core game systems explained in detail\n"
            "- What makes this game unique in the market\n"
            "- Monetization strategy\n"
            "- Target platforms and audience"
        ),
        "artist": (
            "Create the Art Direction Document for **{name}**.\n\n"
            "Include ALL of the following:\n"
            "- Visual style definition (3-5 reference comparisons: 'like X but with Y')\n"
            "- Color palette: primary colors with hex codes, mood description\n"
            "- Character design guidelines: silhouette rules, costume principles, face style\n"
            "- Environment art: biome styles, lighting mood, key locations described\n"
            "- Technical specs: poly budget per LOD, texture resolutions, atlas strategy\n"
            "- Asset pipeline: tools, naming conventions, folder structure\n"
            "- Shader/VFX direction: key visual effects described"
        ),
        "quest_designer": (
            "Create the Narrative Design Document for **{name}**.\n\n"
            "Include ALL of the following:\n"
            "- World lore: history, factions, mythology (500+ words)\n"
            "- Main story arc: 3-act structure with key beats\n"
            "- 3 fully detailed quest designs: title, setup, 5+ objectives, branching choices, rewards, dialogue examples\n"
            "- 5 major characters: name, background, motivation, personality, relationship to player\n"
            "- Dialogue sample: write an actual in-game conversation (20+ lines)\n"
            "- Player motivation hooks: why players will keep playing\n"
            "- Content volume: estimated hours of narrative content"
        ),
        "uiux": (
            "Create the UI/UX Specification for **{name}**.\n\n"
            "Include ALL of the following:\n"
            "- HUD layout: describe every element with screen position and purpose\n"
            "- Main menu flow: screens, transitions, options\n"
            "- In-game menus: inventory, map, character/skill trees\n"
            "- Typography: fonts, sizes, use cases\n"
            "- Color system: UI palette with hex codes, semantic meaning\n"
            "- Feedback systems: what communicates health, XP, damage, etc.\n"
            "- Controller + keyboard mapping\n"
            "- Accessibility features: colorblind mode, subtitles, font scaling"
        ),
        "game_developer": (
            "Create the Technical Architecture Document for **{name}**.\n\n"
            "Include ALL of the following:\n"
            "- Engine choice with detailed justification\n"
            "- Project folder structure (full tree)\n"
            "- Core manager classes: GameManager, InputManager, AudioManager, SaveSystem, UIManager — each with purpose and key methods\n"
            "- Gameplay systems implementation plan: how the GDD mechanics get coded\n"
            "- Performance targets: target FPS, draw call budget, memory limits\n"
            "- Multiplayer considerations (if any)\n"
            "- 6-month development milestone plan\n"
            "- Third-party libraries and middleware\n"
            "- Build and deployment pipeline"
        ),
    },
    "app": {
        "product_manager": (
            "Create the Product Requirements Document for **{name}**.\n\n"
            "Include ALL of the following:\n"
            "- Problem statement: what pain does this solve\n"
            "- Target users: 3 detailed personas\n"
            "- Top 10 features with user stories (As a [user], I want [feature], so that [value])\n"
            "- Acceptance criteria for each feature\n"
            "- Out-of-scope items (what we explicitly WON'T build)\n"
            "- Success metrics: 5 specific KPIs with target values\n"
            "- Monetization strategy with pricing\n"
            "- 6-month roadmap (MVP → v1 → v2)"
        ),
        "backend_dev": (
            "Create the Backend Architecture for **{name}**.\n\n"
            "Include ALL of the following:\n"
            "- Tech stack with justification for each choice\n"
            "- API design: list every endpoint with method, path, request/response schema\n"
            "- Database schema: tables/collections with field names, types, indexes\n"
            "- Authentication/authorization flow (JWT, OAuth, etc.)\n"
            "- Third-party service integrations\n"
            "- Background jobs and workers\n"
            "- Caching strategy\n"
            "- Error handling and logging approach\n"
            "- Security measures"
        ),
        "frontend_dev": (
            "Create the Frontend Architecture for **{name}**.\n\n"
            "Include ALL of the following:\n"
            "- Framework choice with justification\n"
            "- Full folder/component structure\n"
            "- State management approach and data flow\n"
            "- Routing structure: all routes listed\n"
            "- 5 most complex components described with their props/state\n"
            "- API integration layer design\n"
            "- Error and loading state handling\n"
            "- Performance optimization approach\n"
            "- Testing strategy (unit + E2E)"
        ),
        "uiux": (
            "Create the UX Design Specification for **{name}**.\n\n"
            "Include ALL of the following:\n"
            "- Design system: colors (hex), typography scale, spacing system\n"
            "- User journey maps for 3 core workflows\n"
            "- Wireframe descriptions for all major screens\n"
            "- Component library: list all reusable components\n"
            "- Empty states, error states, and loading states\n"
            "- Onboarding flow design\n"
            "- Mobile responsiveness approach\n"
            "- Accessibility compliance plan"
        ),
        "qa": (
            "Create the QA Test Plan for **{name}**.\n\n"
            "Include ALL of the following:\n"
            "- Test strategy overview\n"
            "- 15 critical test cases (each: ID, description, preconditions, steps, expected result)\n"
            "- 10 edge cases that could cause embarrassing bugs\n"
            "- Performance benchmarks (load time, response time, concurrent users)\n"
            "- Security test cases\n"
            "- CI/CD integration plan\n"
            "- Bug severity classification\n"
            "- UAT sign-off criteria"
        ),
    },
    "saas": {
        "product_manager": (
            "Create the SaaS Product Strategy for **{name}**.\n\n"
            "Include: market opportunity size, top 3 competitors with feature gaps, "
            "3 pricing tiers with exact prices and feature lists, go-to-market strategy, "
            "customer acquisition channels, 12-month growth targets, and churn reduction strategies."
        ),
        "architect": (
            "Create the System Architecture for **{name}** (SaaS).\n\n"
            "Include: microservices breakdown with responsibilities, inter-service communication, "
            "multi-tenancy approach, data isolation strategy, API gateway design, auth service, "
            "event-driven components, database-per-service vs shared DB decision with justification, "
            "and scaling triggers."
        ),
        "backend_dev": (
            "Create the Backend Implementation Plan for **{name}**.\n\n"
            "Include all endpoints, database schema, tenant data model, subscription/billing integration, "
            "webhook system, rate limiting approach, and background job architecture."
        ),
        "frontend_dev": (
            "Create the Frontend Implementation for **{name}** (SaaS dashboard).\n\n"
            "Include: framework choice, dashboard component architecture, real-time data approach, "
            "data visualization components, settings/billing UI, and team/role management UI."
        ),
        "uiux": (
            "Create the SaaS UX Design for **{name}**.\n\n"
            "Include: onboarding flow (signup → first value), dashboard layout, key workflows, "
            "empty states, upgrade prompts, team collaboration UI, and retention design patterns."
        ),
    },
    "website": {
        "uiux": "Create the Website UX Design for **{name}**: information architecture, navigation, conversion funnel, mobile-first design, and accessibility.",
        "frontend_dev": "Create the Frontend Build Plan for **{name}**: framework, component architecture, SEO strategy, performance optimization, and deployment.",
        "backend_dev": "Create the Backend/CMS Plan for **{name}**: CMS choice, API integrations, form handling, email setup, and hosting.",
    },
    "tool": {
        "architect": "Create the Tool Architecture for **{name}**: core functionality design, plugin system, config management, CLI/GUI decision, and distribution strategy.",
        "backend_dev": "Create the Core Implementation Plan for **{name}**: tech stack, key algorithms, data model, and packaging approach.",
        "frontend_dev": "Create the UI Plan for **{name}**: framework, key screens, and electron/web packaging if applicable.",
    },
}
