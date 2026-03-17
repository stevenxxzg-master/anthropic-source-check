"""
Configuration for Anthropic API testing.

Priority: CLI args > env vars > .env file > defaults.

Usage:
    # Environment variables
    export ANTHROPIC_API_KEY=sk-ant-xxx
    export ANTHROPIC_BASE_URL=https://api.anthropic.com

    # Or .env file
    echo "ANTHROPIC_API_KEY=sk-ant-xxx" > .env

    # Or pytest CLI
    pytest --base-url=https://custom-proxy.com --api-key=sk-ant-xxx
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    api_key: str
    model: str
    timeout: int

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_MODEL = "claude-sonnet-4-6-20250514"
DEFAULT_TIMEOUT = 30


def _load_dotenv() -> dict[str, str]:
    """Load .env file from project root without external dependencies."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return {}

    result: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        result[key] = value
    return result


def load_config(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
) -> ApiConfig:
    """
    Build config with priority: explicit args > env vars > .env file > defaults.
    All parameters are optional — unset values fall through to the next source.
    """
    dotenv = _load_dotenv()

    return ApiConfig(
        base_url=(
            base_url
            or os.environ.get("ANTHROPIC_BASE_URL")
            or dotenv.get("ANTHROPIC_BASE_URL")
            or DEFAULT_BASE_URL
        ),
        api_key=(
            api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or dotenv.get("ANTHROPIC_API_KEY")
            or ""
        ),
        model=(
            model
            or os.environ.get("ANTHROPIC_MODEL")
            or dotenv.get("ANTHROPIC_MODEL")
            or DEFAULT_MODEL
        ),
        timeout=(
            timeout
            or int(os.environ.get("ANTHROPIC_TIMEOUT", 0))
            or int(dotenv.get("ANTHROPIC_TIMEOUT", 0))
            or DEFAULT_TIMEOUT
        ),
    )
