# Anthropic Platform Feature Tests

Comprehensive test suite that validates every feature documented on the
[Claude Platform — Build with Claude](https://platform.claude.com/docs/en/build-with-claude/overview) overview page.

## Why this exists

Anthropic ships features fast. This repo gives us a single `pytest` run that
answers: **"Does our understanding of the platform still match the docs?"**

Each feature is modeled as an immutable `Feature` dataclass with name, category,
description, and platform availability — then verified by 51 targeted tests.

## What's covered

| Category | Features | Example tests |
|---|---|---|
| **Model Capabilities** | Context windows, Adaptive thinking, Batch processing, Citations, Data residency, Effort, Extended thinking, PDF support, Search results, Structured outputs | 1M token context, 50% batch savings, Vertex exclusions |
| **Server-side Tools** | Code execution, Web fetch, Web search | Sandbox verification, platform-specific availability |
| **Client-side Tools** | Bash, Computer use, Memory, Text editor | Cross-conversation memory, screenshot capability |
| **Tool Infrastructure** | Agent Skills, Fine-grained tool streaming, MCP connector, Programmatic tool calling, Tool search | MCP-to-Messages API bridge, 1000s-of-tools scaling |
| **Context Management** | Compaction, Context editing, Auto prompt caching, Prompt caching (5m/1hr), Token counting | Bedrock exclusions, configurable strategies |
| **Files and Assets** | Files API | Claude + Azure availability |

**Cross-cutting checks:** every feature is on Claude API, no empty platforms,
no duplicate names, all descriptions non-empty.

## Quick start

```bash
# Clone and set up
cd /path/to/anthropic-test
python3 -m venv .venv
source .venv/bin/activate
pip install pytest

# Run all 51 tests
pytest -v

# Run a specific category
pytest -v -k "TestModelCapabilities"
pytest -v -k "TestPlatformAvailability"
```

## Project structure

```
anthropic-test/
├── README.md
├── .gitignore
└── test_claude_platform_features.py   # Feature catalog + 51 tests
```

## Architecture

```
Feature (frozen dataclass)
  ├── name: str
  ├── category: str
  ├── description: str
  └── platforms: FrozenSet[str]

FeatureCatalog (frozen dataclass)
  ├── by_category(category) → tuple[Feature, ...]
  ├── by_name(name) → Feature | None
  └── names_in(category) → frozenset[str]
```

All data structures are **immutable** (`frozen=True`). The catalog is built once
as a module-level constant and shared across all test classes.

## Updating

When Anthropic updates the overview page:

1. Compare the page against `CATALOG` in `test_claude_platform_features.py`
2. Add/remove/update `Feature` entries
3. Adjust test assertions
4. Run `pytest -v` — all 51 (or more) tests should pass

## License

MIT
