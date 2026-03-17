"""
Live API tests aligned with official Anthropic documentation examples.

Source pages:
  - https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
  - https://platform.claude.com/docs/en/build-with-claude/structured-outputs
  - https://platform.claude.com/docs/en/build-with-claude/streaming
  - https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview

These tests are automatically SKIPPED when no API key is configured.
Run them with:

    pytest test_api_live.py -v --api-key=sk-ant-xxx
    pytest test_api_live.py -v --base-url=https://my-proxy.com --api-key=sk-xxx
    ANTHROPIC_API_KEY=sk-ant-xxx pytest test_api_live.py -v
"""

import json
import pytest


# ===========================================================================
# Messages API — basic usage from docs
# ===========================================================================

class TestMessagesAPI:
    """Core message creation from the Messages API documentation."""

    def test_create_message(self, api_client, api_config):
        """Docs: basic message creation."""
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=1024,
            messages=[
                {
                    "content": "Hello, world",
                    "role": "user",
                }
            ],
        )
        assert response.id.startswith("msg_")
        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) > 0
        assert response.content[0].type == "text"

    def test_create_message_with_system(self, api_client, api_config):
        """Docs: system prompt usage."""
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=1024,
            system="You are a calculator. Only reply with numbers.",
            messages=[
                {
                    "role": "user",
                    "content": "What is 2+2?",
                }
            ],
        )
        assert "4" in response.content[0].text

    def test_multi_turn_conversation(self, api_client, api_config):
        """Docs: multi-turn conversation pattern."""
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "My name is TestBot."},
                {"role": "assistant", "content": "Nice to meet you, TestBot!"},
                {"role": "user", "content": "What is my name?"},
            ],
        )
        assert "testbot" in response.content[0].text.lower()

    def test_stop_reason_end_turn(self, api_client, api_config):
        """Verify normal completion returns end_turn."""
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say hello."}],
        )
        assert response.stop_reason == "end_turn"

    def test_stop_reason_max_tokens(self, api_client, api_config):
        """Verify max_tokens truncation returns max_tokens stop reason."""
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=5,
            messages=[{"role": "user", "content": "Write a long essay about the universe."}],
        )
        assert response.stop_reason == "max_tokens"

    def test_usage_fields(self, api_client, api_config):
        """Verify usage metadata is returned."""
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=64,
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0


# ===========================================================================
# Streaming — from docs/en/build-with-claude/streaming
# ===========================================================================

class TestStreaming:
    """Streaming patterns from the official streaming documentation."""

    def test_stream_text(self, api_client, api_config):
        """Docs: basic streaming with text_stream."""
        collected_text = ""
        with api_client.messages.stream(
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
            model=api_config.model,
        ) as stream:
            for text in stream.text_stream:
                collected_text += text

        assert len(collected_text) > 0

    def test_stream_events(self, api_client, api_config):
        """Docs: raw event streaming."""
        event_types = []
        with api_client.messages.stream(
            max_tokens=64,
            messages=[{"role": "user", "content": "Hi"}],
            model=api_config.model,
        ) as stream:
            for event in stream:
                event_types.append(event.type)

        # SDK wraps raw SSE into typed events
        assert len(event_types) > 0


# ===========================================================================
# Adaptive Thinking — from docs/en/build-with-claude/adaptive-thinking
# ===========================================================================

class TestAdaptiveThinking:
    """Adaptive thinking from the official documentation.

    Docs: 'Adaptive thinking is the recommended way to use extended thinking
    with Claude Opus 4.6 and Sonnet 4.6.'
    """

    def test_adaptive_thinking_basic(self, api_client, api_config):
        """Docs: basic adaptive thinking example.

        Source: adaptive-thinking#how-to-use-adaptive-thinking
        """
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            messages=[
                {
                    "role": "user",
                    "content": "Explain why the sum of two even numbers is always even.",
                }
            ],
        )

        # Response should contain text blocks (and optionally thinking blocks)
        text_blocks = [b for b in response.content if b.type == "text"]
        assert len(text_blocks) > 0
        assert "even" in text_blocks[0].text.lower()

    def test_adaptive_thinking_with_effort_medium(self, api_client, api_config):
        """Docs: adaptive thinking with effort parameter.

        Source: adaptive-thinking#adaptive-thinking-with-the-effort-parameter
        """
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )

        text_blocks = [b for b in response.content if b.type == "text"]
        assert len(text_blocks) > 0
        assert "paris" in text_blocks[0].text.lower()

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_adaptive_thinking_effort_levels(self, api_client, api_config, effort):
        """Docs: effort levels table — low, medium, high all accepted."""
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            messages=[{"role": "user", "content": "What is 1+1?"}],
        )
        text_blocks = [b for b in response.content if b.type == "text"]
        assert len(text_blocks) > 0

    def test_adaptive_thinking_produces_thinking_blocks(self, api_client, api_config):
        """Docs: 'At the default effort level (high), Claude almost always thinks.'"""
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            messages=[
                {
                    "role": "user",
                    "content": "What is the greatest common divisor of 1071 and 462?",
                }
            ],
        )

        thinking_blocks = [b for b in response.content if b.type == "thinking"]
        text_blocks = [b for b in response.content if b.type == "text"]

        # At high effort (default), Claude almost always thinks
        assert len(thinking_blocks) > 0, "Expected thinking blocks at default (high) effort"
        assert len(text_blocks) > 0
        # GCD(1071, 462) = 21
        assert "21" in text_blocks[0].text

    def test_adaptive_thinking_streaming(self, api_client, api_config):
        """Docs: streaming with adaptive thinking.

        Source: adaptive-thinking#streaming-with-adaptive-thinking
        """
        collected_text = ""
        event_types = set()

        with api_client.messages.stream(
            model=api_config.model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            messages=[
                {
                    "role": "user",
                    "content": "What is the greatest common divisor of 1071 and 462?",
                }
            ],
        ) as stream:
            for event in stream:
                event_types.add(event.type)
                if hasattr(event, "type") and event.type == "content_block_delta":
                    if hasattr(event, "delta"):
                        if event.delta.type == "text_delta":
                            collected_text += event.delta.text

        assert len(collected_text) > 0

    def test_adaptive_thinking_display_omitted(self, api_client, api_config):
        """Docs: display: 'omitted' skips streaming thinking tokens.

        Source: adaptive-thinking#controlling-thinking-display
        """
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=16000,
            thinking={"type": "adaptive", "display": "omitted"},
            messages=[
                {
                    "role": "user",
                    "content": "What is 15 * 17?",
                }
            ],
        )

        text_blocks = [b for b in response.content if b.type == "text"]
        assert len(text_blocks) > 0
        assert "255" in text_blocks[0].text

        # Thinking blocks should have empty thinking text when display=omitted
        thinking_blocks = [b for b in response.content if b.type == "thinking"]
        for tb in thinking_blocks:
            assert tb.thinking == "" or tb.thinking is None


# ===========================================================================
# Tool Use — from docs/en/agents-and-tools/tool-use/overview
# ===========================================================================

class TestToolUse:
    """Tool use patterns from the official tool use documentation."""

    GET_WEATHER_TOOL = {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }
            },
            "required": ["location"],
        },
    }

    def test_tool_invocation(self, api_client, api_config):
        """Docs: basic tool use example from the overview page."""
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=1024,
            tools=[self.GET_WEATHER_TOOL],
            messages=[
                {"role": "user", "content": "What's the weather like in San Francisco?"}
            ],
        )
        assert response.stop_reason == "tool_use"

        tool_block = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        assert tool_block is not None
        assert tool_block.name == "get_weather"
        assert "san francisco" in tool_block.input["location"].lower()

    def test_tool_result_round_trip(self, api_client, api_config):
        """Docs: full tool use loop — request → tool_use → tool_result → final response."""
        # Step 1: Claude decides to call the tool
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=1024,
            tools=[self.GET_WEATHER_TOOL],
            messages=[
                {"role": "user", "content": "What's the weather like in San Francisco?"}
            ],
        )
        tool_block = next(b for b in response.content if b.type == "tool_use")

        # Step 2: Feed the tool result back
        final = api_client.messages.create(
            model=api_config.model,
            max_tokens=1024,
            tools=[self.GET_WEATHER_TOOL],
            messages=[
                {"role": "user", "content": "What's the weather like in San Francisco?"},
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": json.dumps({
                            "temperature": 65,
                            "unit": "fahrenheit",
                            "condition": "partly cloudy",
                        }),
                    }],
                },
            ],
        )
        assert final.stop_reason == "end_turn"
        text = final.content[0].text.lower()
        assert "65" in text or "cloudy" in text or "san francisco" in text


# ===========================================================================
# Structured Outputs — from docs/en/build-with-claude/structured-outputs
# ===========================================================================

class TestStructuredOutputs:
    """Structured output patterns from the official documentation."""

    def test_json_schema_output(self, api_client, api_config):
        """Docs: JSON outputs with output_config and json_schema.

        Source: structured-outputs#quick-start---basic-json-output
        """
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Extract the key information from this email: "
                        "John Smith (john@example.com) is interested in our "
                        "Enterprise plan and wants to schedule a demo for "
                        "next Tuesday at 2pm."
                    ),
                }
            ],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                            "plan_interest": {"type": "string"},
                            "demo_requested": {"type": "boolean"},
                        },
                        "required": ["name", "email", "plan_interest", "demo_requested"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        parsed = json.loads(response.content[0].text)
        assert parsed["name"] == "John Smith"
        assert parsed["email"] == "john@example.com"
        assert parsed["demo_requested"] is True

    def test_strict_tool_use(self, api_client, api_config):
        """Docs: strict tool use with strict: True.

        Source: structured-outputs#quick-start---basic-strict-tool
        """
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "What's the weather like in San Francisco?"}
            ],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "strict": True,
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The unit of temperature",
                            },
                        },
                        "required": ["location"],
                        "additionalProperties": False,
                    },
                }
            ],
        )
        tool_block = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        assert tool_block is not None
        assert tool_block.name == "get_weather"
        # With strict mode, input must conform to schema
        assert "location" in tool_block.input


# ===========================================================================
# Citations — from docs/en/build-with-claude/citations
# ===========================================================================

class TestCitations:
    """Citation patterns from the official citations documentation."""

    def test_plain_text_citations(self, api_client, api_config):
        """Docs: basic citation example with plain text document.

        Source: citations#how-citations-work
        """
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": "The grass is green. The sky is blue.",
                            },
                            "title": "My Document",
                            "context": "This is a trustworthy document.",
                            "citations": {"enabled": True},
                        },
                        {"type": "text", "text": "What color is the grass and sky?"},
                    ],
                }
            ],
        )
        # Response should contain text blocks, some with citations
        text_blocks = [b for b in response.content if b.type == "text"]
        assert len(text_blocks) > 0

        # At least one block should have citations
        cited_blocks = [b for b in text_blocks if hasattr(b, "citations") and b.citations]
        assert len(cited_blocks) > 0, "Expected at least one cited text block"

        # Check citation structure
        citation = cited_blocks[0].citations[0]
        assert citation.type == "char_location"
        assert hasattr(citation, "cited_text")
        assert citation.document_index == 0

    def test_custom_content_citations(self, api_client, api_config):
        """Docs: custom content documents give control over citation granularity.

        Source: citations#custom-content-documents
        """
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "content",
                                "content": [
                                    {"type": "text", "text": "Python was created by Guido van Rossum."},
                                    {"type": "text", "text": "JavaScript was created by Brendan Eich."},
                                ],
                            },
                            "title": "Programming Languages",
                            "citations": {"enabled": True},
                        },
                        {"type": "text", "text": "Who created Python?"},
                    ],
                }
            ],
        )
        text_blocks = [b for b in response.content if b.type == "text"]
        assert len(text_blocks) > 0

        # Should cite from content blocks
        cited_blocks = [b for b in text_blocks if hasattr(b, "citations") and b.citations]
        assert len(cited_blocks) > 0
        citation = cited_blocks[0].citations[0]
        assert citation.type == "content_block_location"


# ===========================================================================
# Vision — from docs/en/build-with-claude/vision
# ===========================================================================

class TestVision:
    """Vision patterns from the official vision documentation."""

    def test_url_image(self, api_client, api_config):
        """Docs: URL-based image example.

        Source: vision#url-based-image-example
        """
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg",
                            },
                        },
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ],
        )
        assert response.stop_reason == "end_turn"
        text = response.content[0].text.lower()
        assert "ant" in text or "insect" in text

    def test_base64_image(self, api_client, api_config):
        """Docs: base64-encoded image example.

        Source: vision#base64-encoded-image-example
        """
        # Minimal 1x1 PNG (from docs)
        image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"

        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ],
        )
        assert response.stop_reason == "end_turn"
        assert len(response.content[0].text) > 0


# ===========================================================================
# PDF Support — from docs/en/build-with-claude/pdf-support
# ===========================================================================

class TestPDFSupport:
    """PDF processing from the official PDF support documentation."""

    def test_url_pdf_document(self, api_client, api_config):
        """Docs: URL-based PDF document example.

        Source: pdf-support#option-1-url-based-pdf-document
        """
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "url",
                                "url": "https://assets.anthropic.com/m/1cd9d098ac3e6467/original/Claude-3-Model-Card-October-Addendum.pdf",
                            },
                        },
                        {"type": "text", "text": "What are the key findings in this document?"},
                    ],
                }
            ],
        )
        assert response.stop_reason == "end_turn"
        assert len(response.content[0].text) > 0


# ===========================================================================
# Search Results — from docs/en/build-with-claude/search-results
# ===========================================================================

class TestSearchResults:
    """Search results for RAG from the official documentation."""

    def test_search_results_in_user_message(self, api_client, api_config):
        """Docs: top-level search results in user messages.

        Source: search-results#as-top-level-content
        """
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "server_tool_use",
                            "id": "srvtoolu_search_001",
                            "name": "search",
                            "input": {"query": "What is Python?"},
                        },
                        {
                            "type": "server_tool_result",
                            "tool_use_id": "srvtoolu_search_001",
                            "content": [
                                {
                                    "type": "search_result",
                                    "source": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                                    "title": "Python (programming language)",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "Python is a high-level, general-purpose programming language created by Guido van Rossum.",
                                        }
                                    ],
                                },
                            ],
                        },
                        {
                            "type": "text",
                            "text": "What is Python?",
                        },
                    ],
                }
            ],
        )
        assert response.stop_reason == "end_turn"
        text = response.content[0].text.lower()
        assert "python" in text

    def test_search_results_from_tool_call(self, api_client, api_config):
        """Docs: search results returned from a custom tool.

        Source: search-results#from-tool-calls
        """
        # Step 1: Claude calls a search tool
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=1024,
            tools=[
                {
                    "name": "search_knowledge_base",
                    "description": "Search the knowledge base for relevant information",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            }
                        },
                        "required": ["query"],
                    },
                }
            ],
            messages=[
                {"role": "user", "content": "Search for info about the Eiffel Tower"}
            ],
        )
        assert response.stop_reason == "tool_use"

        tool_block = next(b for b in response.content if b.type == "tool_use")
        assert tool_block.name == "search_knowledge_base"

        # Step 2: Return search results
        final = api_client.messages.create(
            model=api_config.model,
            max_tokens=1024,
            tools=[
                {
                    "name": "search_knowledge_base",
                    "description": "Search the knowledge base for relevant information",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query"}
                        },
                        "required": ["query"],
                    },
                }
            ],
            messages=[
                {"role": "user", "content": "Search for info about the Eiffel Tower"},
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": [
                                {
                                    "type": "search_result",
                                    "source": "https://en.wikipedia.org/wiki/Eiffel_Tower",
                                    "title": "Eiffel Tower",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was constructed from 1887 to 1889 and is 330 metres tall.",
                                        }
                                    ],
                                },
                            ],
                        }
                    ],
                },
            ],
        )
        assert final.stop_reason == "end_turn"
        text = final.content[0].text.lower()
        assert "eiffel" in text or "paris" in text or "tower" in text


# ===========================================================================
# Prompt Caching — from docs/en/build-with-claude/prompt-caching
# ===========================================================================

class TestPromptCaching:
    """Prompt caching patterns from the official documentation."""

    def test_automatic_caching(self, api_client, api_config):
        """Docs: automatic caching with cache_control parameter.

        Source: prompt-caching#automatic-caching
        """
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=1024,
            cache_control={"type": "ephemeral"},
            system="You are an AI assistant tasked with analyzing literary works.",
            messages=[
                {
                    "role": "user",
                    "content": "Analyze the major themes in 'Pride and Prejudice'.",
                }
            ],
        )
        assert response.stop_reason == "end_turn"
        assert response.usage.input_tokens > 0

    def test_explicit_cache_breakpoint(self, api_client, api_config):
        """Docs: explicit cache_control on system prompt blocks.

        Source: prompt-caching#large-context-caching-example
        """
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": "You are an AI assistant tasked with analyzing documents.",
                },
                {
                    "type": "text",
                    "text": "Here is background context: " + ("x " * 500),
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=[
                {
                    "role": "user",
                    "content": "Summarize the key points.",
                }
            ],
        )
        assert response.stop_reason == "end_turn"
        assert len(response.content[0].text) > 0


# ===========================================================================
# Token Counting — from docs
# ===========================================================================

class TestTokenCounting:
    """Token counting endpoint."""

    def test_count_message_tokens(self, api_client, api_config):
        """Docs: count tokens before sending."""
        result = api_client.messages.count_tokens(
            model=api_config.model,
            messages=[{"role": "user", "content": "Hello, world!"}],
        )
        assert result.input_tokens > 0


# ===========================================================================
# Error Handling — API error responses
# ===========================================================================

class TestErrorHandling:
    """Verify the API returns proper typed errors."""

    def test_invalid_model_returns_not_found(self, api_client):
        """Invalid model ID should return NotFoundError."""
        anthropic = pytest.importorskip("anthropic")
        with pytest.raises(anthropic.NotFoundError):
            api_client.messages.create(
                model="nonexistent-model-xyz",
                max_tokens=64,
                messages=[{"role": "user", "content": "Hi"}],
            )

    def test_empty_messages_returns_bad_request(self, api_client, api_config):
        """Empty messages array should return BadRequestError."""
        anthropic = pytest.importorskip("anthropic")
        with pytest.raises(anthropic.BadRequestError):
            api_client.messages.create(
                model=api_config.model,
                max_tokens=64,
                messages=[],
            )


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
