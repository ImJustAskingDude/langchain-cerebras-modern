import os
from typing import Any

import pytest

DEFAULT_ZAI_REASONING_MODEL = "zai-glm-4.7"


def require_cerebras_api_key() -> str:
    """Return the API key for live integration tests or skip the module."""
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        pytest.skip(
            "CEREBRAS_API_KEY is not set; skipping live Cerebras integration tests.",
            allow_module_level=True,
        )
    return api_key


def get_standard_chat_model_params() -> dict[str, Any]:
    """Return model params for the provider-agnostic standard test suite."""
    require_cerebras_api_key()
    model = os.environ.get("CEREBRAS_TEST_STANDARD_MODEL") or DEFAULT_ZAI_REASONING_MODEL
    params: dict[str, Any] = {"model": model}

    # Default to plain text responses for the generic LangChain conformance suite.
    if model.startswith("zai-glm-"):
        params["disable_reasoning"] = True

    return params


def get_zai_reasoning_model() -> str:
    """Return the configured ZAI reasoning model used in live tests."""
    require_cerebras_api_key()
    return os.environ.get("CEREBRAS_TEST_ZAI_MODEL") or DEFAULT_ZAI_REASONING_MODEL


def get_non_reasoning_model_params() -> dict[str, Any]:
    """Return a model config that should produce standard text output."""
    require_cerebras_api_key()
    model = os.environ.get("CEREBRAS_TEST_NON_REASONING_MODEL")
    if model:
        return {"model": model}
    return {
        "model": get_zai_reasoning_model(),
        "disable_reasoning": True,
    }
