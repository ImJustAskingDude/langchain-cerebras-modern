"""Test reasoning parameters for configured Cerebras models."""

import os

import pytest

from langchain_cerebras import ChatCerebras
from tests.integration_tests import get_zai_reasoning_model, require_cerebras_api_key

require_cerebras_api_key()
ZAI_REASONING_MODEL = get_zai_reasoning_model()
GPT_OSS_MODEL = os.environ.get("CEREBRAS_TEST_GPT_OSS_MODEL")


class TestReasoningParameters:
    """Test suite for reasoning parameters."""

    @pytest.mark.skipif(
        not GPT_OSS_MODEL,
        reason="CEREBRAS_TEST_GPT_OSS_MODEL not set for GPT-OSS-specific tests",
    )
    def test_gpt_oss_reasoning_effort_low(self) -> None:
        """Test a configured GPT-OSS model with low reasoning effort."""
        chat = ChatCerebras(
            model=GPT_OSS_MODEL,  # type: ignore[arg-type]
            reasoning_effort="low",
            temperature=0.7,
        )

        # Verify the parameter is set
        assert chat.reasoning_effort == "low"

        # Test with a simple query
        response = chat.invoke("What is 2+2?")
        assert response.content

    @pytest.mark.skipif(
        not GPT_OSS_MODEL,
        reason="CEREBRAS_TEST_GPT_OSS_MODEL not set for GPT-OSS-specific tests",
    )
    def test_gpt_oss_reasoning_effort_medium(self) -> None:
        """Test a configured GPT-OSS model with medium reasoning effort."""
        chat = ChatCerebras(
            model=GPT_OSS_MODEL,  # type: ignore[arg-type]
            reasoning_effort="medium",
            temperature=0.7,
        )

        # Verify the parameter is set
        assert chat.reasoning_effort == "medium"

        # Test with a simple query
        response = chat.invoke("What is 2+2?")
        assert response.content

    @pytest.mark.skipif(
        not GPT_OSS_MODEL,
        reason="CEREBRAS_TEST_GPT_OSS_MODEL not set for GPT-OSS-specific tests",
    )
    def test_gpt_oss_reasoning_effort_high(self) -> None:
        """Test a configured GPT-OSS model with high reasoning effort."""
        chat = ChatCerebras(
            model=GPT_OSS_MODEL,  # type: ignore[arg-type]
            reasoning_effort="high",
            temperature=0.7,
        )

        # Verify the parameter is set
        assert chat.reasoning_effort == "high"

        # Test with a simple query
        response = chat.invoke("What is 2+2?")
        assert response.content

    @pytest.mark.skipif(
        not GPT_OSS_MODEL,
        reason="CEREBRAS_TEST_GPT_OSS_MODEL not set for GPT-OSS-specific tests",
    )
    def test_gpt_oss_without_reasoning_effort(self) -> None:
        """Test a configured GPT-OSS model without reasoning effort."""
        chat = ChatCerebras(
            model=GPT_OSS_MODEL,  # type: ignore[arg-type]
            temperature=0.7,
        )

        # Verify the parameter is None (default)
        assert chat.reasoning_effort is None

        # Test with a simple query
        response = chat.invoke("What is 2+2?")
        assert response.content

    def test_zai_disable_reasoning_true(self) -> None:
        """Test the configured ZAI model with reasoning disabled."""
        chat = ChatCerebras(
            model=ZAI_REASONING_MODEL,
            disable_reasoning=True,
            temperature=0.7,
        )

        # Verify the parameter is set
        assert chat.disable_reasoning is True

        # Test with a simple query
        response = chat.invoke("What is 2+2?")
        assert response.content

    def test_zai_disable_reasoning_false(self) -> None:
        """Test the configured ZAI model with reasoning enabled."""
        chat = ChatCerebras(
            model=ZAI_REASONING_MODEL,
            disable_reasoning=False,
            temperature=0.7,
        )

        # Verify the parameter is set
        assert chat.disable_reasoning is False

        # Test with a simple query
        response = chat.invoke("What is 2+2?")
        assert response.content

    def test_zai_without_disable_reasoning(self) -> None:
        """Test the configured ZAI model without disable_reasoning."""
        chat = ChatCerebras(
            model=ZAI_REASONING_MODEL,
            temperature=0.7,
        )

        # Verify the parameter is None (default)
        assert chat.disable_reasoning is None

        # Test with a simple query
        response = chat.invoke("What is 2+2?")
        assert response.content

    @pytest.mark.skipif(
        not GPT_OSS_MODEL,
        reason="CEREBRAS_TEST_GPT_OSS_MODEL not set for GPT-OSS-specific tests",
    )
    def test_default_params_includes_reasoning_effort(self) -> None:
        """Test that _default_params includes reasoning_effort when set."""
        chat = ChatCerebras(
            model=GPT_OSS_MODEL,  # type: ignore[arg-type]
            reasoning_effort="high",
        )

        params = chat._default_params
        assert "reasoning_effort" in params
        assert params["reasoning_effort"] == "high"

    def test_default_params_includes_disable_reasoning(self) -> None:
        """Test that _default_params includes disable_reasoning when set."""
        chat = ChatCerebras(
            model=ZAI_REASONING_MODEL,
            disable_reasoning=True,
        )

        params = chat._default_params
        # disable_reasoning is a nonstandard parameter and must be in extra_body
        assert "extra_body" in params
        assert "disable_reasoning" in params["extra_body"]
        assert params["extra_body"]["disable_reasoning"] is True

    def test_default_params_excludes_none_values(self) -> None:
        """Test that _default_params excludes reasoning params when None."""
        chat = ChatCerebras(
            model=ZAI_REASONING_MODEL,
        )

        # When None, the parameters should not be in the dict
        assert chat.reasoning_effort is None
        assert chat.disable_reasoning is None


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
