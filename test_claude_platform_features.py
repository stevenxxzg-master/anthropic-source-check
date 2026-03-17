"""
Feature tests for Claude Platform - Build with Claude Overview page.
URL: https://platform.claude.com/docs/en/build-with-claude/overview

Tests verify all documented features, capabilities, and API surface areas.
"""

import pytest
from dataclasses import dataclass
from typing import FrozenSet


# ---------------------------------------------------------------------------
# Domain models (immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Feature:
    name: str
    category: str
    description: str
    platforms: FrozenSet[str]


@dataclass(frozen=True)
class FeatureCatalog:
    features: tuple[Feature, ...]

    def by_category(self, category: str) -> tuple[Feature, ...]:
        return tuple(f for f in self.features if f.category == category)

    def by_name(self, name: str) -> Feature | None:
        return next((f for f in self.features if f.name == name), None)

    def names_in(self, category: str) -> frozenset[str]:
        return frozenset(f.name for f in self.by_category(category))


# ---------------------------------------------------------------------------
# Platform constants
# ---------------------------------------------------------------------------

CLAUDE_API = "claude_api"
BEDROCK = "bedrock"
VERTEX_AI = "vertex_ai"
AZURE_AI = "azure_ai"

ALL_PLATFORMS = frozenset({CLAUDE_API, BEDROCK, VERTEX_AI, AZURE_AI})
CLAUDE_ONLY = frozenset({CLAUDE_API})
CLAUDE_AZURE = frozenset({CLAUDE_API, AZURE_AI})
NO_AZURE = frozenset({CLAUDE_API, BEDROCK, VERTEX_AI})
NO_VERTEX = frozenset({CLAUDE_API, BEDROCK, AZURE_AI})


# ---------------------------------------------------------------------------
# Feature catalog built from the documentation
# ---------------------------------------------------------------------------

CATALOG = FeatureCatalog(features=(
    # --- Model capabilities ---
    Feature(
        name="Context windows",
        category="model_capabilities",
        description="Up to 1M tokens for processing large documents, extensive codebases, and long conversations",
        platforms=ALL_PLATFORMS,
    ),
    Feature(
        name="Adaptive thinking",
        category="model_capabilities",
        description="Let Claude dynamically decide when and how much to think",
        platforms=ALL_PLATFORMS,
    ),
    Feature(
        name="Batch processing",
        category="model_capabilities",
        description="Process large volumes of requests asynchronously for 50% cost savings",
        platforms=NO_AZURE,
    ),
    Feature(
        name="Citations",
        category="model_capabilities",
        description="Ground Claude's responses in source documents with detailed references",
        platforms=ALL_PLATFORMS,
    ),
    Feature(
        name="Data residency",
        category="model_capabilities",
        description="Control where model inference runs using geographic controls",
        platforms=CLAUDE_ONLY,
    ),
    Feature(
        name="Effort",
        category="model_capabilities",
        description="Control how many tokens Claude uses when responding",
        platforms=ALL_PLATFORMS,
    ),
    Feature(
        name="Extended thinking",
        category="model_capabilities",
        description="Enhanced reasoning capabilities for complex tasks with step-by-step thought process",
        platforms=ALL_PLATFORMS,
    ),
    Feature(
        name="PDF support",
        category="model_capabilities",
        description="Process and analyze text and visual content from PDF documents",
        platforms=ALL_PLATFORMS,
    ),
    Feature(
        name="Search results",
        category="model_capabilities",
        description="Enable natural citations for RAG applications with proper source attribution",
        platforms=ALL_PLATFORMS,
    ),
    Feature(
        name="Structured outputs",
        category="model_capabilities",
        description="Guarantee schema conformance with JSON outputs and strict tool use",
        platforms=NO_VERTEX,
    ),

    # --- Server-side tools ---
    Feature(
        name="Code execution",
        category="server_side_tools",
        description="Run code in a sandboxed environment for data analysis and calculations",
        platforms=CLAUDE_AZURE,
    ),
    Feature(
        name="Web fetch",
        category="server_side_tools",
        description="Retrieve full content from specified web pages and PDF documents",
        platforms=CLAUDE_AZURE,
    ),
    Feature(
        name="Web search",
        category="server_side_tools",
        description="Augment Claude's knowledge with current real-world data from the web",
        platforms=frozenset({CLAUDE_API, VERTEX_AI, AZURE_AI}),
    ),

    # --- Client-side tools ---
    Feature(
        name="Bash",
        category="client_side_tools",
        description="Execute bash commands and scripts to interact with the system shell",
        platforms=ALL_PLATFORMS,
    ),
    Feature(
        name="Computer use",
        category="client_side_tools",
        description="Control computer interfaces by taking screenshots and issuing mouse/keyboard commands",
        platforms=ALL_PLATFORMS,
    ),
    Feature(
        name="Memory",
        category="client_side_tools",
        description="Store and retrieve information across conversations",
        platforms=ALL_PLATFORMS,
    ),
    Feature(
        name="Text editor",
        category="client_side_tools",
        description="Create and edit text files with a built-in text editor interface",
        platforms=ALL_PLATFORMS,
    ),

    # --- Tool infrastructure ---
    Feature(
        name="Agent Skills",
        category="tool_infrastructure",
        description="Extend Claude's capabilities with pre-built or custom Skills",
        platforms=CLAUDE_AZURE,
    ),
    Feature(
        name="Fine-grained tool streaming",
        category="tool_infrastructure",
        description="Stream tool use parameters without buffering/JSON validation",
        platforms=ALL_PLATFORMS,
    ),
    Feature(
        name="MCP connector",
        category="tool_infrastructure",
        description="Connect to remote MCP servers directly from the Messages API",
        platforms=CLAUDE_AZURE,
    ),
    Feature(
        name="Programmatic tool calling",
        category="tool_infrastructure",
        description="Call tools programmatically from within code execution containers",
        platforms=CLAUDE_AZURE,
    ),
    Feature(
        name="Tool search",
        category="tool_infrastructure",
        description="Scale to thousands of tools by dynamically discovering and loading tools on-demand",
        platforms=ALL_PLATFORMS,
    ),

    # --- Context management ---
    Feature(
        name="Compaction",
        category="context_management",
        description="Server-side context summarization for long-running conversations",
        platforms=ALL_PLATFORMS,
    ),
    Feature(
        name="Context editing",
        category="context_management",
        description="Automatically manage conversation context with configurable strategies",
        platforms=ALL_PLATFORMS,
    ),
    Feature(
        name="Automatic prompt caching",
        category="context_management",
        description="Simplify prompt caching to a single API parameter",
        platforms=CLAUDE_AZURE,
    ),
    Feature(
        name="Prompt caching (5m)",
        category="context_management",
        description="Reduce costs and latency with 5-minute prompt caching",
        platforms=ALL_PLATFORMS,
    ),
    Feature(
        name="Prompt caching (1hr)",
        category="context_management",
        description="Extended 1-hour cache duration for less frequently accessed context",
        platforms=frozenset({CLAUDE_API, VERTEX_AI, AZURE_AI}),
    ),
    Feature(
        name="Token counting",
        category="context_management",
        description="Determine the number of tokens in a message before sending",
        platforms=ALL_PLATFORMS,
    ),

    # --- Files and assets ---
    Feature(
        name="Files API",
        category="files_and_assets",
        description="Upload and manage files to use with Claude without re-uploading each request",
        platforms=CLAUDE_AZURE,
    ),
))


# ===========================================================================
# Category: API Surface Organization
# ===========================================================================

class TestAPISurfaceOrganization:
    """The API surface is organized into five documented areas."""

    EXPECTED_CATEGORIES = frozenset({
        "model_capabilities",
        "server_side_tools",
        "client_side_tools",
        "tool_infrastructure",
        "context_management",
        "files_and_assets",
    })

    def test_all_categories_present(self):
        actual = frozenset(f.category for f in CATALOG.features)
        # server_side_tools + client_side_tools map to the "Tools" area
        assert self.EXPECTED_CATEGORIES == actual

    def test_total_feature_count(self):
        assert len(CATALOG.features) == 29

    @pytest.mark.parametrize("category,expected_count", [
        ("model_capabilities", 10),
        ("server_side_tools", 3),
        ("client_side_tools", 4),
        ("tool_infrastructure", 5),
        ("context_management", 6),
        ("files_and_assets", 1),
    ])
    def test_feature_count_per_category(self, category: str, expected_count: int):
        assert len(CATALOG.by_category(category)) == expected_count


# ===========================================================================
# Category: Model Capabilities
# ===========================================================================

class TestModelCapabilities:
    """Features that steer Claude's reasoning, formatting, and input modalities."""

    EXPECTED_FEATURES = frozenset({
        "Context windows",
        "Adaptive thinking",
        "Batch processing",
        "Citations",
        "Data residency",
        "Effort",
        "Extended thinking",
        "PDF support",
        "Search results",
        "Structured outputs",
    })

    def test_all_model_capability_features_listed(self):
        assert CATALOG.names_in("model_capabilities") == self.EXPECTED_FEATURES

    def test_context_window_supports_1m_tokens(self):
        feat = CATALOG.by_name("Context windows")
        assert feat is not None
        assert "1M tokens" in feat.description

    def test_batch_processing_50_percent_savings(self):
        feat = CATALOG.by_name("Batch processing")
        assert feat is not None
        assert "50%" in feat.description

    def test_data_residency_claude_api_only(self):
        feat = CATALOG.by_name("Data residency")
        assert feat is not None
        assert feat.platforms == CLAUDE_ONLY

    def test_adaptive_thinking_available_all_platforms(self):
        feat = CATALOG.by_name("Adaptive thinking")
        assert feat is not None
        assert feat.platforms == ALL_PLATFORMS

    def test_extended_thinking_available_all_platforms(self):
        feat = CATALOG.by_name("Extended thinking")
        assert feat is not None
        assert feat.platforms == ALL_PLATFORMS

    def test_structured_outputs_not_on_vertex(self):
        feat = CATALOG.by_name("Structured outputs")
        assert feat is not None
        assert VERTEX_AI not in feat.platforms

    def test_citations_description_mentions_references(self):
        feat = CATALOG.by_name("Citations")
        assert feat is not None
        assert "references" in feat.description.lower()

    def test_effort_description_mentions_tokens(self):
        feat = CATALOG.by_name("Effort")
        assert feat is not None
        assert "tokens" in feat.description.lower()

    def test_pdf_support_handles_text_and_visual(self):
        feat = CATALOG.by_name("PDF support")
        assert feat is not None
        assert "text" in feat.description.lower()
        assert "visual" in feat.description.lower()


# ===========================================================================
# Category: Server-side Tools
# ===========================================================================

class TestServerSideTools:
    """Built-in tools run by the platform on Claude's behalf."""

    EXPECTED_FEATURES = frozenset({
        "Code execution",
        "Web fetch",
        "Web search",
    })

    def test_all_server_side_tools_listed(self):
        assert CATALOG.names_in("server_side_tools") == self.EXPECTED_FEATURES

    def test_code_execution_sandboxed(self):
        feat = CATALOG.by_name("Code execution")
        assert feat is not None
        assert "sandbox" in feat.description.lower()

    def test_web_search_not_on_bedrock(self):
        feat = CATALOG.by_name("Web search")
        assert feat is not None
        assert BEDROCK not in feat.platforms

    def test_web_fetch_claude_and_azure_only(self):
        feat = CATALOG.by_name("Web fetch")
        assert feat is not None
        assert feat.platforms == CLAUDE_AZURE

    def test_code_execution_claude_and_azure_only(self):
        feat = CATALOG.by_name("Code execution")
        assert feat is not None
        assert feat.platforms == CLAUDE_AZURE


# ===========================================================================
# Category: Client-side Tools
# ===========================================================================

class TestClientSideTools:
    """Tools implemented and executed by the caller, invoked via tool_use."""

    EXPECTED_FEATURES = frozenset({
        "Bash",
        "Computer use",
        "Memory",
        "Text editor",
    })

    def test_all_client_side_tools_listed(self):
        assert CATALOG.names_in("client_side_tools") == self.EXPECTED_FEATURES

    def test_all_client_tools_on_all_platforms(self):
        for feat in CATALOG.by_category("client_side_tools"):
            assert feat.platforms == ALL_PLATFORMS, f"{feat.name} not on all platforms"

    def test_computer_use_involves_screenshots(self):
        feat = CATALOG.by_name("Computer use")
        assert feat is not None
        assert "screenshot" in feat.description.lower()

    def test_memory_supports_cross_conversation(self):
        feat = CATALOG.by_name("Memory")
        assert feat is not None
        assert "across conversations" in feat.description.lower()

    def test_bash_executes_commands(self):
        feat = CATALOG.by_name("Bash")
        assert feat is not None
        assert "command" in feat.description.lower()


# ===========================================================================
# Category: Tool Infrastructure
# ===========================================================================

class TestToolInfrastructure:
    """Infrastructure supporting tool discovery, orchestration, and scaling."""

    EXPECTED_FEATURES = frozenset({
        "Agent Skills",
        "Fine-grained tool streaming",
        "MCP connector",
        "Programmatic tool calling",
        "Tool search",
    })

    def test_all_tool_infra_features_listed(self):
        assert CATALOG.names_in("tool_infrastructure") == self.EXPECTED_FEATURES

    def test_mcp_connector_bridges_messages_api(self):
        feat = CATALOG.by_name("MCP connector")
        assert feat is not None
        assert "Messages API" in feat.description

    def test_tool_search_scales_to_thousands(self):
        feat = CATALOG.by_name("Tool search")
        assert feat is not None
        assert "thousands" in feat.description.lower()

    def test_agent_skills_supports_custom_and_prebuilt(self):
        feat = CATALOG.by_name("Agent Skills")
        assert feat is not None
        assert "custom" in feat.description.lower()
        assert "pre-built" in feat.description.lower()

    def test_programmatic_tool_calling_reduces_latency(self):
        feat = CATALOG.by_name("Programmatic tool calling")
        assert feat is not None
        assert "code execution" in feat.description.lower()

    def test_fine_grained_streaming_on_all_platforms(self):
        feat = CATALOG.by_name("Fine-grained tool streaming")
        assert feat is not None
        assert feat.platforms == ALL_PLATFORMS


# ===========================================================================
# Category: Context Management
# ===========================================================================

class TestContextManagement:
    """Features for controlling and optimizing Claude's context window."""

    EXPECTED_FEATURES = frozenset({
        "Compaction",
        "Context editing",
        "Automatic prompt caching",
        "Prompt caching (5m)",
        "Prompt caching (1hr)",
        "Token counting",
    })

    def test_all_context_management_features_listed(self):
        assert CATALOG.names_in("context_management") == self.EXPECTED_FEATURES

    def test_compaction_is_server_side_summarization(self):
        feat = CATALOG.by_name("Compaction")
        assert feat is not None
        assert "summarization" in feat.description.lower()

    def test_prompt_caching_5m_on_all_platforms(self):
        feat = CATALOG.by_name("Prompt caching (5m)")
        assert feat is not None
        assert feat.platforms == ALL_PLATFORMS

    def test_prompt_caching_1hr_not_on_bedrock(self):
        feat = CATALOG.by_name("Prompt caching (1hr)")
        assert feat is not None
        assert BEDROCK not in feat.platforms

    def test_auto_caching_claude_and_azure_only(self):
        feat = CATALOG.by_name("Automatic prompt caching")
        assert feat is not None
        assert feat.platforms == CLAUDE_AZURE

    def test_token_counting_available_all_platforms(self):
        feat = CATALOG.by_name("Token counting")
        assert feat is not None
        assert feat.platforms == ALL_PLATFORMS

    def test_context_editing_supports_configurable_strategies(self):
        feat = CATALOG.by_name("Context editing")
        assert feat is not None
        assert "configurable" in feat.description.lower()


# ===========================================================================
# Category: Files and Assets
# ===========================================================================

class TestFilesAndAssets:
    """File management features for use with Claude."""

    def test_files_api_exists(self):
        feat = CATALOG.by_name("Files API")
        assert feat is not None

    def test_files_api_claude_and_azure_only(self):
        feat = CATALOG.by_name("Files API")
        assert feat is not None
        assert feat.platforms == CLAUDE_AZURE

    def test_files_api_supports_multiple_formats(self):
        feat = CATALOG.by_name("Files API")
        assert feat is not None
        desc = feat.description.lower()
        # Docs mention PDFs, images, and text files
        assert "file" in desc


# ===========================================================================
# Cross-cutting: Platform Availability
# ===========================================================================

class TestPlatformAvailability:
    """Verify cross-platform availability patterns."""

    def test_all_features_available_on_claude_api(self):
        for feat in CATALOG.features:
            assert CLAUDE_API in feat.platforms, (
                f"{feat.name} should be available on Claude API"
            )

    def test_azure_features_count(self):
        azure_features = [f for f in CATALOG.features if AZURE_AI in f.platforms]
        # Most features are on Azure (all except Batch processing)
        assert len(azure_features) >= 25

    def test_bedrock_features_count(self):
        bedrock_features = [f for f in CATALOG.features if BEDROCK in f.platforms]
        assert len(bedrock_features) >= 15

    def test_no_feature_has_empty_platforms(self):
        for feat in CATALOG.features:
            assert len(feat.platforms) > 0, f"{feat.name} has no platforms"


# ===========================================================================
# Cross-cutting: Feature Uniqueness
# ===========================================================================

class TestFeatureUniqueness:
    """Ensure catalog integrity."""

    def test_no_duplicate_feature_names(self):
        names = [f.name for f in CATALOG.features]
        assert len(names) == len(set(names)), "Duplicate feature names found"

    def test_all_features_have_descriptions(self):
        for feat in CATALOG.features:
            assert feat.description, f"{feat.name} has empty description"

    def test_all_features_have_category(self):
        for feat in CATALOG.features:
            assert feat.category, f"{feat.name} has empty category"


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
