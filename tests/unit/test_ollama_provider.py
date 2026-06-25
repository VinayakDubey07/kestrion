"""
Tests for OllamaProvider's response parsing, especially the malformed
tool-call-arguments recovery path. No real Ollama server needed — these
tests fake the HTTP response shape directly.
"""

import httpx
import pytest

from kestrion.llm.ollama_provider import OllamaProvider


class _FakeResponse:
    def __init__(self, json_data):
        self._json_data = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


class _FakeAsyncClient:
    def __init__(self, response_data):
        self._response_data = response_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, url, json):
        return _FakeResponse(self._response_data)


def _patch_client(monkeypatch, response_data):
    monkeypatch.setattr(httpx, "AsyncClient", lambda timeout=None: _FakeAsyncClient(response_data))


async def test_ollama_provider_parses_well_formed_tool_call(monkeypatch):
    _patch_client(monkeypatch, {
        "message": {
            "content": None,
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": '{"city": "Bangalore"}'}}
            ],
        },
        "prompt_eval_count": 30,
        "eval_count": 10,
    })

    provider = OllamaProvider(model="llama3.1")
    response = await provider.complete(messages=[], tools=[])

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "Bangalore"}
    assert response.tokens_in == 30
    assert response.tokens_out == 10
    assert response.cost_usd == 0.0


async def test_ollama_provider_recovers_from_malformed_tool_call_json(monkeypatch):
    """
    Regression test: a small local model returning syntactically invalid
    JSON in tool_call arguments must not crash the whole provider call —
    it should surface as an empty-arguments tool call with a visible
    note, so the agent loop can continue (and the LLM can see the
    failure and retry) instead of an unhandled exception taking down
    the run.
    """
    _patch_client(monkeypatch, {
        "message": {
            "content": "Let me check that for you.",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": '{"city": "Bangalore"'}}  # missing closing brace
            ],
        },
        "prompt_eval_count": 30,
        "eval_count": 10,
    })

    provider = OllamaProvider(model="llama3.1")
    response = await provider.complete(messages=[], tools=[])  # must not raise

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {}  # malformed args -> empty, not a crash
    assert "malformed tool-call" in response.text
    assert "get_weather" in response.text


async def test_ollama_provider_handles_dict_arguments_without_double_parsing(monkeypatch):
    """Some Ollama models return arguments as a dict directly, not a JSON string."""
    _patch_client(monkeypatch, {
        "message": {
            "content": None,
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Bangalore"}}}
            ],
        },
    })

    provider = OllamaProvider(model="llama3.1")
    response = await provider.complete(messages=[], tools=[])

    assert response.tool_calls[0].arguments == {"city": "Bangalore"}


async def test_ollama_provider_handles_no_tool_calls(monkeypatch):
    _patch_client(monkeypatch, {
        "message": {"content": "It's sunny today."},
    })

    provider = OllamaProvider(model="llama3.1")
    response = await provider.complete(messages=[], tools=[])

    assert response.tool_calls == []
    assert response.text == "It's sunny today."
    assert response.stop_reason == "end_turn"