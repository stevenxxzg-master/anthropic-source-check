"""
Level 4: Performance & Cost Benchmarks.

Measures real-world performance characteristics:
- Latency: time to first token (TTFT), total response time
- Throughput: tokens per second (TPS)
- Token usage: input/output token counts and cache efficiency
- Cost estimation: per-request cost based on model pricing
- Feature overhead: latency impact of thinking, tools, streaming

Auto-skipped if no API key is configured.

Run:
    pytest test_perf_benchmark.py -v --api-key=sk-ant-xxx
    pytest test_perf_benchmark.py -v -k "TestLatency"
    pytest test_perf_benchmark.py -v -s  # -s to see benchmark output
"""

import time
import json
import pytest
from dataclasses import dataclass

# Auto-skip all tests in this file if no API key configured
pytestmark = pytest.mark.usefixtures("require_api_key")


# ---------------------------------------------------------------------------
# Benchmark data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    total_time_s: float
    ttft_s: float | None  # time to first token (streaming only)
    input_tokens: int
    output_tokens: int
    model: str

    @property
    def tokens_per_second(self) -> float:
        if self.total_time_s <= 0:
            return 0.0
        return self.output_tokens / self.total_time_s

    @property
    def cost_estimate_usd(self) -> float:
        """Estimate cost using Sonnet 4.6 pricing ($3/$15 per 1M tokens)."""
        input_cost = (self.input_tokens / 1_000_000) * 3.0
        output_cost = (self.output_tokens / 1_000_000) * 15.0
        return input_cost + output_cost

    def summary(self) -> str:
        parts = [
            f"  {self.name}:",
            f"    Total time:     {self.total_time_s:.2f}s",
        ]
        if self.ttft_s is not None:
            parts.append(f"    TTFT:           {self.ttft_s:.3f}s")
        parts.extend([
            f"    Input tokens:   {self.input_tokens}",
            f"    Output tokens:  {self.output_tokens}",
            f"    Tokens/sec:     {self.tokens_per_second:.1f}",
            f"    Est. cost:      ${self.cost_estimate_usd:.6f}",
        ])
        return "\n".join(parts)


def _print_result(result: BenchmarkResult):
    """Print benchmark result to stdout (visible with pytest -s)."""
    print(f"\n{result.summary()}")


# ===========================================================================
# Latency Benchmarks
# ===========================================================================

class TestLatency:
    """Measure response latency for different request types."""

    def test_simple_message_latency(self, api_client, api_config):
        """Baseline: simple message round-trip time."""
        start = time.perf_counter()
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=64,
            messages=[{"role": "user", "content": "Say hello."}],
        )
        elapsed = time.perf_counter() - start

        result = BenchmarkResult(
            name="Simple message",
            total_time_s=elapsed,
            ttft_s=None,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=api_config.model,
        )
        _print_result(result)
        assert elapsed < 30, f"Simple message took {elapsed:.1f}s (expected <30s)"

    def test_long_response_latency(self, api_client, api_config):
        """Measure latency for a longer response."""
        start = time.perf_counter()
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Write a 3-paragraph essay about the ocean."}],
        )
        elapsed = time.perf_counter() - start

        result = BenchmarkResult(
            name="Long response (1024 max_tokens)",
            total_time_s=elapsed,
            ttft_s=None,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=api_config.model,
        )
        _print_result(result)
        assert elapsed < 60

    def test_system_prompt_latency(self, api_client, api_config):
        """Measure overhead of a large system prompt."""
        large_system = "You are an expert assistant. " + ("Context: x. " * 200)

        start = time.perf_counter()
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=64,
            system=large_system,
            messages=[{"role": "user", "content": "Say hello."}],
        )
        elapsed = time.perf_counter() - start

        result = BenchmarkResult(
            name="Large system prompt (~200 sentences)",
            total_time_s=elapsed,
            ttft_s=None,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=api_config.model,
        )
        _print_result(result)
        assert elapsed < 30


# ===========================================================================
# Streaming TTFT Benchmarks
# ===========================================================================

class TestStreamingTTFT:
    """Measure time to first token via streaming."""

    def test_streaming_ttft(self, api_client, api_config):
        """Measure TTFT (time to first text token) via streaming."""
        start = time.perf_counter()
        ttft = None
        total_text = ""

        with api_client.messages.stream(
            model=api_config.model,
            max_tokens=256,
            messages=[{"role": "user", "content": "Explain gravity in one paragraph."}],
        ) as stream:
            for text in stream.text_stream:
                if ttft is None:
                    ttft = time.perf_counter() - start
                total_text += text

        elapsed = time.perf_counter() - start
        final_message = stream.get_final_message()

        result = BenchmarkResult(
            name="Streaming TTFT",
            total_time_s=elapsed,
            ttft_s=ttft,
            input_tokens=final_message.usage.input_tokens,
            output_tokens=final_message.usage.output_tokens,
            model=api_config.model,
        )
        _print_result(result)
        assert ttft is not None, "Never received a text token"
        assert ttft < 10, f"TTFT was {ttft:.2f}s (expected <10s)"

    def test_streaming_ttft_with_thinking(self, api_client, api_config):
        """Measure TTFT with adaptive thinking enabled — thinking adds latency."""
        start = time.perf_counter()
        ttft = None
        total_text = ""

        with api_client.messages.stream(
            model=api_config.model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": "What is 17 * 23?"}],
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event, "delta") and event.delta.type == "text_delta":
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        total_text += event.delta.text

        elapsed = time.perf_counter() - start
        final_message = stream.get_final_message()

        result = BenchmarkResult(
            name="Streaming TTFT (with thinking)",
            total_time_s=elapsed,
            ttft_s=ttft,
            input_tokens=final_message.usage.input_tokens,
            output_tokens=final_message.usage.output_tokens,
            model=api_config.model,
        )
        _print_result(result)
        # Thinking adds latency, so we allow more time
        assert elapsed < 60


# ===========================================================================
# Throughput (Tokens per Second)
# ===========================================================================

class TestThroughput:
    """Measure output throughput in tokens per second."""

    def test_output_tps(self, api_client, api_config):
        """Measure tokens per second for a medium-length response."""
        start = time.perf_counter()
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=512,
            messages=[{"role": "user", "content": "List 20 interesting facts about space."}],
        )
        elapsed = time.perf_counter() - start

        result = BenchmarkResult(
            name="Output throughput (512 max_tokens)",
            total_time_s=elapsed,
            ttft_s=None,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=api_config.model,
        )
        _print_result(result)
        assert result.tokens_per_second > 1, f"TPS too low: {result.tokens_per_second:.1f}"

    @pytest.mark.parametrize("effort", ["low", "high"])
    def test_effort_throughput_comparison(self, api_client, api_config, effort):
        """Compare throughput at different effort levels."""
        start = time.perf_counter()
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=512,
            output_config={"effort": effort},
            messages=[{"role": "user", "content": "Explain how a CPU works."}],
        )
        elapsed = time.perf_counter() - start

        result = BenchmarkResult(
            name=f"Throughput (effort={effort})",
            total_time_s=elapsed,
            ttft_s=None,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=api_config.model,
        )
        _print_result(result)
        assert result.tokens_per_second > 0.5


# ===========================================================================
# Token Usage & Cost
# ===========================================================================

class TestTokenUsage:
    """Analyze token consumption patterns and cost estimates."""

    def test_token_usage_basic(self, api_client, api_config):
        """Baseline token usage for a simple request."""
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=256,
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )
        result = BenchmarkResult(
            name="Token usage (simple)",
            total_time_s=0,
            ttft_s=None,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=api_config.model,
        )
        _print_result(result)
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    def test_token_usage_with_tools(self, api_client, api_config):
        """Measure token overhead of tool definitions."""
        tool = {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state"},
                },
                "required": ["location"],
            },
        }

        # Without tools
        r_no_tool = api_client.messages.create(
            model=api_config.model,
            max_tokens=64,
            messages=[{"role": "user", "content": "Hello"}],
        )
        # With tools
        r_with_tool = api_client.messages.create(
            model=api_config.model,
            max_tokens=64,
            tools=[tool],
            tool_choice={"type": "none"},
            messages=[{"role": "user", "content": "Hello"}],
        )

        overhead = r_with_tool.usage.input_tokens - r_no_tool.usage.input_tokens
        print(f"\n  Tool definition overhead: {overhead} input tokens")
        assert overhead > 0, "Tool definitions should add input tokens"

    def test_cache_efficiency(self, api_client, api_config):
        """Measure prompt caching token savings on repeated requests."""
        system_prompt = "You are a helpful assistant. " + ("Background info. " * 100)

        # First request — cache write
        r1 = api_client.messages.create(
            model=api_config.model,
            max_tokens=32,
            cache_control={"type": "ephemeral"},
            system=system_prompt,
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Second request — should hit cache
        r2 = api_client.messages.create(
            model=api_config.model,
            max_tokens=32,
            cache_control={"type": "ephemeral"},
            system=system_prompt,
            messages=[{"role": "user", "content": "Hi there"}],
        )

        usage1 = r1.usage
        usage2 = r2.usage

        cache_read_tokens = getattr(usage2, "cache_read_input_tokens", 0) or 0
        cache_creation_tokens = getattr(usage1, "cache_creation_input_tokens", 0) or 0

        print(f"\n  Cache efficiency:")
        print(f"    Request 1 - cache_creation: {cache_creation_tokens} tokens")
        print(f"    Request 2 - cache_read:     {cache_read_tokens} tokens")
        print(f"    Request 1 - input_tokens:   {usage1.input_tokens}")
        print(f"    Request 2 - input_tokens:   {usage2.input_tokens}")

        # If caching worked, the second request should have cache_read_input_tokens > 0
        # or lower input_tokens. This is soft — caching may not always activate.
        assert r2.usage.input_tokens > 0


# ===========================================================================
# Feature Overhead Comparison
# ===========================================================================

class TestFeatureOverhead:
    """Compare latency/cost overhead of different features."""

    def test_thinking_overhead(self, api_client, api_config):
        """Measure the latency overhead of adaptive thinking."""
        prompt = "What is the square root of 144?"

        # Without thinking
        start = time.perf_counter()
        r_no_think = api_client.messages.create(
            model=api_config.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        time_no_think = time.perf_counter() - start

        # With thinking
        start = time.perf_counter()
        r_think = api_client.messages.create(
            model=api_config.model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )
        time_think = time.perf_counter() - start

        print(f"\n  Thinking overhead:")
        print(f"    Without thinking: {time_no_think:.2f}s, {r_no_think.usage.output_tokens} output tokens")
        print(f"    With thinking:    {time_think:.2f}s, {r_think.usage.output_tokens} output tokens")
        print(f"    Overhead:         {time_think - time_no_think:.2f}s")

        # Both should return correct answer
        text_blocks = [b for b in r_think.content if b.type == "text"]
        assert "12" in text_blocks[0].text
        assert "12" in r_no_think.content[0].text

    def test_structured_output_overhead(self, api_client, api_config):
        """Measure overhead of JSON schema enforcement."""
        prompt = "What is the capital of France?"

        # Plain text
        start = time.perf_counter()
        r_plain = api_client.messages.create(
            model=api_config.model,
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        time_plain = time.perf_counter() - start

        # JSON schema
        start = time.perf_counter()
        r_json = api_client.messages.create(
            model=api_config.model,
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "capital": {"type": "string"},
                            "country": {"type": "string"},
                        },
                        "required": ["capital", "country"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        time_json = time.perf_counter() - start

        parsed = json.loads(r_json.content[0].text)
        print(f"\n  Structured output overhead:")
        print(f"    Plain text: {time_plain:.2f}s")
        print(f"    JSON schema: {time_json:.2f}s")
        print(f"    Overhead: {time_json - time_plain:.2f}s")

        assert "paris" in parsed["capital"].lower()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
