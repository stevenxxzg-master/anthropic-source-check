# Anthropic Platform Feature Tests

Comprehensive test suite that validates every feature documented on the
[Claude Platform — Build with Claude](https://platform.claude.com/docs/en/build-with-claude/overview) overview page — plus live API integration tests against any Anthropic-compatible endpoint.

## Why this exists

Anthropic ships features fast. This repo gives us a single `pytest` run that
answers: **"Does our understanding of the platform still match the docs?"**
and **"Does this API endpoint actually work?"**

## Quick start

```bash
git clone https://github.com/stevenxxzg-master/anthropic-source-check.git
cd anthropic-source-check
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run catalog tests (no API key needed)

```bash
pytest test_claude_platform_features.py -v
```

### Run live API tests

```bash
# Option 1: environment variable
export ANTHROPIC_API_KEY=sk-ant-xxx
pytest test_api_live.py -v

# Option 2: CLI flags (great for testing proxies)
pytest test_api_live.py -v \
  --base-url=https://my-proxy.com \
  --api-key=sk-xxx \
  --model=claude-sonnet-4-6-20250514

# Option 3: .env file
cp .env.example .env
# edit .env with your credentials
pytest test_api_live.py -v

# Run everything
pytest -v
```

## Configuration

All settings can be provided via **CLI flags**, **environment variables**, or a **`.env` file**.
Priority: CLI > env vars > `.env` > defaults.

| Setting | CLI Flag | Env Var | Default |
|---|---|---|---|
| Base URL | `--base-url` | `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` |
| API Key | `--api-key` | `ANTHROPIC_API_KEY` | _(none — live tests skipped)_ |
| Model | `--model` | `ANTHROPIC_MODEL` | `claude-sonnet-4-6-20250514` |
| Timeout | `--timeout` | `ANTHROPIC_TIMEOUT` | `30` |

This makes it easy to test against different providers:

```bash
# Official Anthropic API
pytest test_api_live.py -v --api-key=sk-ant-xxx

# Custom proxy / gateway
pytest test_api_live.py -v --base-url=https://gateway.mycompany.com --api-key=my-key

# AWS Bedrock proxy
pytest test_api_live.py -v --base-url=https://bedrock-proxy.internal --api-key=xxx

# Different model
pytest test_api_live.py -v --api-key=sk-ant-xxx --model=claude-opus-4-6-20250514
```

## What's tested

### Catalog tests (`test_claude_platform_features.py`) — 51 tests

| Category | Features | Example tests |
|---|---|---|
| **Model Capabilities** | Context windows, Adaptive thinking, Batch processing, Citations, Data residency, Effort, Extended thinking, PDF support, Search results, Structured outputs | 1M token context, 50% batch savings, Vertex exclusions |
| **Server-side Tools** | Code execution, Web fetch, Web search | Sandbox verification, platform-specific availability |
| **Client-side Tools** | Bash, Computer use, Memory, Text editor | Cross-conversation memory, screenshot capability |
| **Tool Infrastructure** | Agent Skills, Fine-grained tool streaming, MCP connector, Programmatic tool calling, Tool search | MCP-to-Messages API bridge, 1000s-of-tools scaling |
| **Context Management** | Compaction, Context editing, Auto prompt caching, Prompt caching (5m/1hr), Token counting | Bedrock exclusions, configurable strategies |
| **Files and Assets** | Files API | Claude + Azure availability |

### Live API tests (`test_api_live.py`) — auto-skipped without API key

| Test Class | What it verifies |
|---|---|
| **TestMessagesAPI** | Simple messages, system prompts, multi-turn, max_tokens, response metadata |
| **TestStreaming** | Stream text, stream events |
| **TestStructuredOutputs** | JSON response parsing |
| **TestToolUse** | Tool invocation, tool result round-trip |
| **TestExtendedThinking** | Thinking blocks + final answer |
| **TestEffort** | Effort levels (low/medium/high) |
| **TestTokenCounting** | Token counting endpoint |
| **TestErrorHandling** | Invalid model, empty messages |

## Project structure

```
anthropic-test/
├── README.md
├── .env.example          # Template — copy to .env
├── .gitignore
├── requirements.txt      # pytest + anthropic SDK
├── config.py             # Configuration loader (CLI > env > .env > defaults)
├── conftest.py           # Pytest fixtures + CLI options
├── test_claude_platform_features.py  # Feature catalog + 51 tests
└── test_api_live.py      # Live API integration tests
```

## License

MIT
