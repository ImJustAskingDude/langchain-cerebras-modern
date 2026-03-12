import pytest

from langchain_cerebras import ChatCerebras


def test_missing_api_key_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)

    with pytest.raises(ValueError, match="CEREBRAS_API_KEY"):
        ChatCerebras(model="llama3.1-8b")
