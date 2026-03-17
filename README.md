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

### Live API tests (`test_api_live.py`) — aligned with official docs, auto-skipped without API key

Every test mirrors an official code example from the Anthropic documentation.

| Test Class | Doc Source | What it verifies |
|---|---|---|
| **TestMessagesAPI** | Messages API | Create message, system prompt, multi-turn, stop reasons, usage metadata |
| **TestStreaming** | [Streaming](https://platform.claude.com/docs/en/build-with-claude/streaming) | `text_stream` iteration, raw event types |
| **TestAdaptiveThinking** | [Adaptive Thinking](https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking) | `thinking.type: "adaptive"`, effort levels (low/medium/high), thinking blocks, streaming with thinking, `display: "omitted"` |
| **TestToolUse** | [Tool Use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview) | `get_weather` tool invocation, full tool result round-trip |
| **TestStructuredOutputs** | [Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) | `output_config` with `json_schema`, `strict: true` tool use |
| **TestCitations** | [Citations](https://platform.claude.com/docs/en/build-with-claude/citations) | Plain text citations with `char_location`, custom content with `content_block_location` |
| **TestVision** | [Vision](https://platform.claude.com/docs/en/build-with-claude/vision) | URL-based image, base64-encoded image |
| **TestPDFSupport** | [PDF Support](https://platform.claude.com/docs/en/build-with-claude/pdf-support) | URL-based PDF document analysis |
| **TestSearchResults** | [Search Results](https://platform.claude.com/docs/en/build-with-claude/search-results) | Top-level search results, search results from tool calls |
| **TestPromptCaching** | [Prompt Caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) | Automatic caching, explicit cache breakpoints |
| **TestExtendedThinking** | [Extended Thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking) | Manual `budget_tokens` mode, streaming with thinking, thinking + tool use round-trip |
| **TestEffortParameter** | [Effort](https://platform.claude.com/docs/en/build-with-claude/effort) | Standalone `output_config.effort` (low/medium/high), token usage comparison |
| **TestBatchProcessing** | [Batch Processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing) | Create, retrieve, list, cancel message batches |
| **TestFilesAPI** | [Files API](https://platform.claude.com/docs/en/build-with-claude/files) | Upload, list, use in message, delete (beta) |
| **TestTokenCounting** | Token Counting | `count_tokens` endpoint |
| **TestErrorHandling** | API Errors | `NotFoundError` for invalid model, `BadRequestError` for empty messages |

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
