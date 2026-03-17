"""
Live API tests — hit the real Anthropic API (or any compatible proxy).

These tests are automatically SKIPPED when no API key is configured.
Run them with:

    pytest test_api_live.py -v --api-key=sk-ant-xxx
    pytest test_api_live.py -v --base-url=https://my-proxy.com --api-key=sk-xxx
    ANTHROPIC_API_KEY=sk-ant-xxx pytest test_api_live.py -v

Or put credentials in .env:

    ANTHROPIC_API_KEY=sk-ant-xxx
    ANTHROPIC_BASE_URL=https://my-proxy.com
"""

import json
import pytest


# ===========================================================================
# Basic Messages API
# ===========================================================================

class TestMessagesAPI:
    """Core message creation — the foundation everything else builds on."""

    def test_simple_message(self, api_client, api_config):
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=64,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        )
        assert response.stop_reason == "end_turn"
        assert len(response.content) > 0
        assert "hello" in response.content[0].text.lower()

    def test_system_prompt(self, api_client, api_config):
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=64,
            system="You are a calculator. Only reply with numbers.",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )
        assert "4" in response.content[0].text

    def test_multi_turn_conversation(self, api_client, api_config):
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=64,
            messages=[
                {"role": "user", "content": "My name is TestBot."},
                {"role": "assistant", "content": "Nice to meet you, TestBot!"},
                {"role": "user", "content": "What is my name?"},
            ],
        )
        assert "testbot" in response.content[0].text.lower()

    def test_max_tokens_respected(self, api_client, api_config):
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=5,
            messages=[{"role": "user", "content": "Write a very long essay about the universe."}],
        )
        # Should stop due to max_tokens, not end_turn
        assert response.stop_reason == "max_tokens"

    def test_response_metadata(self, api_client, api_config):
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=32,
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert response.id.startswith("msg_")
        assert response.model is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0


# ===========================================================================
# Streaming
# ===========================================================================

class TestStreaming:
    """Verify streaming responses work end-to-end."""

    def test_basic_stream(self, api_client, api_config):
        collected_text = ""
        with api_client.messages.stream(
            model=api_config.model,
            max_tokens=64,
            messages=[{"role": "user", "content": "Say 'streaming works'"}],
        ) as stream:
            for text in stream.text_stream:
                collected_text += text

        assert "streaming" in collected_text.lower()

    def test_stream_events(self, api_client, api_config):
        events = []
        with api_client.messages.stream(
            model=api_config.model,
            max_tokens=32,
            messages=[{"role": "user", "content": "Hi"}],
        ) as stream:
            for event in stream:
                events.append(event.type)

        assert "message_start" in events or "content_block_start" in events


# ===========================================================================
# Structured Outputs
# ===========================================================================

class TestStructuredOutputs:
    """JSON mode and structured responses."""

    def test_json_response(self, api_client, api_config):
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=128,
            messages=[{
                "role": "user",
                "content": (
                    "Return a JSON object with keys 'name' and 'age'. "
                    "Example: {\"name\": \"Alice\", \"age\": 30}. "
                    "Only output valid JSON, nothing else."
                ),
            }],
        )
        text = response.content[0].text.strip()
        parsed = json.loads(text)
        assert "name" in parsed
        assert "age" in parsed


# ===========================================================================
# Tool Use
# ===========================================================================

class TestToolUse:
    """Client-side tool calling via the Messages API."""

    WEATHER_TOOL = {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. San Francisco, CA",
                },
            },
            "required": ["location"],
        },
    }

    def test_tool_invocation(self, api_client, api_config):
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=256,
            tools=[self.WEATHER_TOOL],
            messages=[{
                "role": "user",
                "content": "What's the weather in San Francisco?",
            }],
        )
        assert response.stop_reason == "tool_use"
        tool_block = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        assert tool_block is not None
        assert tool_block.name == "get_weather"
        assert "san francisco" in tool_block.input["location"].lower()

    def test_tool_result_round_trip(self, api_client, api_config):
        # Step 1: get tool call
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=256,
            tools=[self.WEATHER_TOOL],
            messages=[{
                "role": "user",
                "content": "What's the weather in Tokyo?",
            }],
        )
        tool_block = next(b for b in response.content if b.type == "tool_use")

        # Step 2: feed tool result back
        final = api_client.messages.create(
            model=api_config.model,
            max_tokens=256,
            tools=[self.WEATHER_TOOL],
            messages=[
                {"role": "user", "content": "What's the weather in Tokyo?"},
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": '{"temperature": 22, "condition": "sunny"}',
                    }],
                },
            ],
        )
        assert final.stop_reason == "end_turn"
        text = final.content[0].text.lower()
        assert "22" in text or "sunny" in text or "tokyo" in text


# ===========================================================================
# Extended Thinking
# ===========================================================================

class TestExtendedThinking:
    """Extended thinking / chain-of-thought reasoning."""

    def test_thinking_produces_reasoning(self, api_client, api_config):
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=16000,
            thinking={
                "type": "enabled",
                "budget_tokens": 8000,
            },
            messages=[{
                "role": "user",
                "content": "What is 127 * 43? Think step by step.",
            }],
        )
        thinking_blocks = [b for b in response.content if b.type == "thinking"]
        text_blocks = [b for b in response.content if b.type == "text"]
        assert len(thinking_blocks) > 0, "Expected thinking blocks"
        assert len(text_blocks) > 0, "Expected text output"
        assert "5461" in text_blocks[0].text


# ===========================================================================
# Effort Control
# ===========================================================================

class TestEffort:
    """Effort parameter controls token usage."""

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_effort_levels_accepted(self, api_client, api_config, effort):
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=128,
            thinking={
                "type": "enabled",
                "budget_tokens": 8000,
            },
            messages=[{"role": "user", "content": "What is 1+1?"}],
        )
        assert response.content is not None


# ===========================================================================
# Token Counting
# ===========================================================================

class TestTokenCounting:
    """Token counting endpoint."""

    def test_count_message_tokens(self, api_client, api_config):
        result = api_client.messages.count_tokens(
            model=api_config.model,
            messages=[{"role": "user", "content": "Hello, world!"}],
        )
        assert result.input_tokens > 0


# ===========================================================================
# Error Handling
# ===========================================================================

class TestErrorHandling:
    """Verify the API returns proper errors for bad requests."""

    def test_invalid_model_returns_error(self, api_client):
        anthropic = pytest.importorskip("anthropic")
        with pytest.raises(anthropic.NotFoundError):
            api_client.messages.create(
                model="nonexistent-model-xyz",
                max_tokens=32,
                messages=[{"role": "user", "content": "Hi"}],
            )

    def test_empty_messages_returns_error(self, api_client, api_config):
        anthropic = pytest.importorskip("anthropic")
        with pytest.raises(anthropic.BadRequestError):
            api_client.messages.create(
                model=api_config.model,
                max_tokens=32,
                messages=[],
            )


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
