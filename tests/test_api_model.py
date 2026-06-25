from types import SimpleNamespace

import pytest

from api_model import LLMResponseError, OpenRouter


def _openrouter_model() -> OpenRouter:
    model = OpenRouter.__new__(OpenRouter)
    model.model_checkpoint = "provider/test-model"
    model.reasoning_effort = None
    model.max_completion_tokens = None
    return model


def _response(*, content, finish_reason="stop", usage=None, reasoning=None):
    message = SimpleNamespace(
        content=content,
        reasoning=reasoning,
        reasoning_details=[],
        tool_calls=[],
    )
    choice = SimpleNamespace(
        message=message,
        finish_reason=finish_reason,
        native_finish_reason=None,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def test_openrouter_extracts_text_content():
    model = _openrouter_model()

    assert model._extract_response_text(_response(content="Translated text.")) == "Translated text."


def test_openrouter_rejects_missing_content():
    model = _openrouter_model()

    with pytest.raises(LLMResponseError, match="did not contain text content") as exc:
        model._extract_response_text(_response(content=None, reasoning="thinking"))

    assert "reasoning_chars" in str(exc.value)


def test_openrouter_rejects_truncated_response():
    model = _openrouter_model()

    with pytest.raises(LLMResponseError, match="did not finish cleanly") as exc:
        model._extract_response_text(_response(content="partial", finish_reason="length"))

    assert "finish_reason='length'" in str(exc.value)
