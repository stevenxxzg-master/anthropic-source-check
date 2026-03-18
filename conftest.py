"""
Pytest configuration — shared fixtures and CLI options.

Adds --base-url, --api-key, --model CLI flags so you can point tests
at any Anthropic-compatible endpoint without touching env vars.

Reports are auto-generated in output-check/ after every run.
"""

import pytest
from config import ApiConfig, load_config

# Register the report plugin
pytest_plugins = ["report_plugin"]


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("anthropic", "Anthropic API options")
    group.addoption(
        "--base-url",
        dest="base_url",
        default=None,
        help="Anthropic API base URL (default: https://api.anthropic.com)",
    )
    group.addoption(
        "--api-key",
        dest="api_key",
        default=None,
        help="Anthropic API key (default: $ANTHROPIC_API_KEY or .env)",
    )
    group.addoption(
        "--model",
        dest="model",
        default=None,
        help="Model to test against (default: claude-sonnet-4-6-20250514)",
    )
    group.addoption(
        "--timeout",
        dest="timeout",
        type=int,
        default=None,
        help="Request timeout in seconds (default: 30)",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def api_config(request: pytest.FixtureRequest) -> ApiConfig:
    """Resolved API configuration available to all tests."""
    return load_config(
        base_url=request.config.getoption("base_url"),
        api_key=request.config.getoption("api_key"),
        model=request.config.getoption("model"),
        timeout=request.config.getoption("timeout"),
    )


@pytest.fixture(scope="session")
def require_api_key(api_config: ApiConfig) -> ApiConfig:
    """Skip the test if no API key is configured."""
    if not api_config.is_configured:
        pytest.skip("No ANTHROPIC_API_KEY configured — skipping live API test")
    return api_config


@pytest.fixture(scope="session")
def api_client(require_api_key: ApiConfig):
    """
    Return an Anthropic client pointed at the configured base URL.
    Requires the `anthropic` package — tests using this fixture are
    skipped automatically if the package is missing.
    """
    anthropic = pytest.importorskip("anthropic")
    return anthropic.Anthropic(
        api_key=require_api_key.api_key,
        base_url=require_api_key.base_url,
        timeout=require_api_key.timeout,
    )
