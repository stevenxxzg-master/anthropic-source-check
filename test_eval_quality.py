"""
Level 3: Quality & Correctness Evaluation Tests.

Tests response quality using DeepEval metrics:
- Correctness: factual accuracy against expected answers
- Hallucination: grounded in provided context
- Answer Relevancy: response addresses the question
- Faithfulness: claims traceable to source context

Auto-skipped if DeepEval is not installed or no API key is configured.

Run:
    pip install deepeval
    pytest test_eval_quality.py -v --api-key=sk-ant-xxx
    pytest test_eval_quality.py -v -k "TestCorrectness"
"""

import json
import os
import pytest
import time

# Auto-skip all tests in this file if no API key configured
pytestmark = pytest.mark.usefixtures("require_api_key")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_claude(api_client, api_config, prompt, system=None, max_tokens=1024):
    """Send a message to Claude and return the text response."""
    kwargs = {
        "model": api_config.model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    response = api_client.messages.create(**kwargs)
    return response.content[0].text


def _get_judge_model():
    """Get the DeepEval judge model, configured via env vars.

    Supports custom OpenAI-compatible endpoints via:
        OPENAI_BASE_URL — custom base URL (e.g. proxy)
        OPENAI_MODEL_NAME — model name (default: gpt-4.1)
    """
    from deepeval.models import GPTModel

    base_url = os.environ.get("OPENAI_BASE_URL")
    model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-4.1")

    kwargs = {"model": model_name}
    if base_url:
        kwargs["base_url"] = base_url

    return GPTModel(**kwargs)


def _skip_no_deepeval():
    """Skip if deepeval is not installed or no OpenAI key configured."""
    pytest.importorskip("deepeval", reason="deepeval not installed — pip install deepeval")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("No OPENAI_API_KEY configured — needed for DeepEval judge model")


# ===========================================================================
# Correctness — factual accuracy
# ===========================================================================

class TestCorrectness:
    """Verify Claude gives factually correct answers using GEval metric."""

    FACT_QUESTIONS = [
        {
            "input": "What is the chemical formula for water?",
            "expected": "H2O",
        },
        {
            "input": "What year did World War II end?",
            "expected": "1945",
        },
        {
            "input": "What is the speed of light in a vacuum in meters per second?",
            "expected": "approximately 299,792,458 meters per second",
        },
        {
            "input": "Who wrote Romeo and Juliet?",
            "expected": "William Shakespeare",
        },
        {
            "input": "What is the capital of Australia?",
            "expected": "Canberra",
        },
    ]

    @pytest.mark.parametrize("qa", FACT_QUESTIONS, ids=[q["input"][:40] for q in FACT_QUESTIONS])
    def test_factual_correctness(self, api_client, api_config, qa):
        """Assert Claude's answer is factually correct using GEval."""
        _skip_no_deepeval()
        from deepeval import assert_test
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
        from deepeval.metrics import GEval

        actual_output = _call_claude(api_client, api_config, qa["input"])

        correctness = GEval(
            name="Correctness",
            criteria="Determine if the 'actual output' is factually correct based on the 'expected output'.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=0.5,
            model=_get_judge_model(),
        )
        test_case = LLMTestCase(
            input=qa["input"],
            actual_output=actual_output,
            expected_output=qa["expected"],
        )
        assert_test(test_case, [correctness])


# ===========================================================================
# Hallucination — grounded in provided context
# ===========================================================================

class TestHallucination:
    """Verify Claude doesn't hallucinate beyond provided context."""

    def test_stays_grounded_in_context(self, api_client, api_config):
        """Claude should only state facts from the provided context."""
        _skip_no_deepeval()
        from deepeval import assert_test
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import HallucinationMetric

        context = [
            "The Eiffel Tower is located in Paris, France.",
            "It was constructed from 1887 to 1889.",
            "The tower is 330 metres tall.",
            "It was designed by Gustave Eiffel's engineering company.",
        ]
        context_text = "\n".join(context)

        actual_output = _call_claude(
            api_client,
            api_config,
            f"Based ONLY on this context, describe the Eiffel Tower:\n\n{context_text}",
            system="Only state facts explicitly mentioned in the provided context. Do not add any outside knowledge.",
        )

        metric = HallucinationMetric(threshold=0.5, model=_get_judge_model())
        test_case = LLMTestCase(
            input="Describe the Eiffel Tower based on the context.",
            actual_output=actual_output,
            context=context,
        )
        assert_test(test_case, [metric])

    def test_admits_unknown(self, api_client, api_config):
        """Claude should say it doesn't know when context lacks the answer."""
        _skip_no_deepeval()
        from deepeval import assert_test
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import HallucinationMetric

        context = [
            "Python is a programming language created by Guido van Rossum.",
            "Python was first released in 1991.",
        ]
        context_text = "\n".join(context)

        actual_output = _call_claude(
            api_client,
            api_config,
            f"Based ONLY on this context, what is the population of Paris?\n\n{context_text}",
            system="Only answer from the provided context. If the answer is not in the context, say 'I don't have that information in the provided context.'",
        )

        metric = HallucinationMetric(threshold=0.5, model=_get_judge_model())
        test_case = LLMTestCase(
            input="What is the population of Paris?",
            actual_output=actual_output,
            context=context,
        )
        assert_test(test_case, [metric])


# ===========================================================================
# Answer Relevancy — response addresses the question
# ===========================================================================

class TestAnswerRelevancy:
    """Verify Claude's response is relevant to the question asked."""

    RELEVANCY_CASES = [
        {
            "input": "What are the benefits of exercise?",
            "check_words": ["health", "fitness", "physical", "mental", "benefit"],
        },
        {
            "input": "Explain how photosynthesis works.",
            "check_words": ["light", "plant", "carbon", "oxygen", "energy", "chlorophyll"],
        },
        {
            "input": "What is machine learning?",
            "check_words": ["algorithm", "data", "learn", "model", "pattern", "predict"],
        },
    ]

    @pytest.mark.parametrize("case", RELEVANCY_CASES, ids=[c["input"][:40] for c in RELEVANCY_CASES])
    def test_answer_is_relevant(self, api_client, api_config, case):
        """Assert Claude's response is relevant using AnswerRelevancyMetric."""
        _skip_no_deepeval()
        from deepeval import assert_test
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import AnswerRelevancyMetric

        actual_output = _call_claude(api_client, api_config, case["input"])

        metric = AnswerRelevancyMetric(threshold=0.5, model=_get_judge_model())
        test_case = LLMTestCase(
            input=case["input"],
            actual_output=actual_output,
        )
        assert_test(test_case, [metric])

    def test_refuses_irrelevant_topic(self, api_client, api_config):
        """When asked to stay on topic, Claude shouldn't drift."""
        actual_output = _call_claude(
            api_client,
            api_config,
            "What is the capital of France?",
            system="You are a geography assistant. Only answer geography questions. Be concise.",
        )
        text = actual_output.lower()
        assert "paris" in text


# ===========================================================================
# Faithfulness — claims traceable to provided context (RAG)
# ===========================================================================

class TestFaithfulness:
    """Verify Claude's claims are traceable to retrieval context."""

    def test_faithful_to_retrieval_context(self, api_client, api_config):
        """All claims in the response should be traceable to the context."""
        _skip_no_deepeval()
        from deepeval import assert_test
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import FaithfulnessMetric

        retrieval_context = [
            "Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning.",
            "Elon Musk joined as chairman of the board in 2004 after leading the Series A funding round.",
            "Tesla's first car, the Roadster, was released in 2008.",
            "The Model S sedan was launched in 2012.",
        ]
        context_text = "\n".join(retrieval_context)

        actual_output = _call_claude(
            api_client,
            api_config,
            f"Based on this context, give a brief history of Tesla:\n\n{context_text}",
            system="Only use facts from the provided context.",
        )

        metric = FaithfulnessMetric(threshold=0.5, model=_get_judge_model())
        test_case = LLMTestCase(
            input="Give a brief history of Tesla.",
            actual_output=actual_output,
            retrieval_context=retrieval_context,
        )
        assert_test(test_case, [metric])


# ===========================================================================
# Structured Output Correctness
# ===========================================================================

class TestStructuredOutputCorrectness:
    """Verify structured outputs contain correct data, not just valid JSON."""

    def test_json_extraction_accuracy(self, api_client, api_config):
        """Extracted JSON should match the source data accurately."""
        source_text = (
            "Meeting Notes - March 15, 2026\n"
            "Attendees: Alice Chen, Bob Kumar, Carol Zhang\n"
            "Topic: Q1 Budget Review\n"
            "Decision: Approved $50,000 for marketing campaign\n"
            "Next meeting: March 22, 2026"
        )
        response = api_client.messages.create(
            model=api_config.model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"Extract structured data from these meeting notes:\n\n{source_text}",
            }],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string"},
                            "attendees": {"type": "array", "items": {"type": "string"}},
                            "topic": {"type": "string"},
                            "decision": {"type": "string"},
                            "budget_amount": {"type": "number"},
                            "next_meeting": {"type": "string"},
                        },
                        "required": ["date", "attendees", "topic", "decision", "budget_amount"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        parsed = json.loads(response.content[0].text)

        assert len(parsed["attendees"]) == 3
        assert "Alice Chen" in parsed["attendees"]
        assert parsed["budget_amount"] == 50000
        assert "budget" in parsed["topic"].lower() or "q1" in parsed["topic"].lower()

    def test_classification_accuracy(self, api_client, api_config):
        """Sentiment classification should be correct for clear-cut cases."""
        test_cases = [
            ("I absolutely love this product! Best purchase ever!", "positive"),
            ("This is terrible. Complete waste of money.", "negative"),
            ("The package arrived on Tuesday.", "neutral"),
        ]
        for text, expected_sentiment in test_cases:
            response = api_client.messages.create(
                model=api_config.model,
                max_tokens=64,
                messages=[{
                    "role": "user",
                    "content": f"Classify the sentiment as positive, negative, or neutral. Reply with one word only.\n\n{text}",
                }],
            )
            result = response.content[0].text.strip().lower()
            assert expected_sentiment in result, (
                f"Expected '{expected_sentiment}' for '{text[:30]}...', got '{result}'"
            )


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
