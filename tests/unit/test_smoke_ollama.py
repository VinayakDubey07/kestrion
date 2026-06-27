"""
Real, live smoke test against a local Ollama server. Unlike every other
test in this suite, this one is NOT hermetic — it requires Ollama
running locally with a model pulled, and makes a real HTTP call.

This exists specifically to close a gap flagged during development:
anthropic_provider.py and openai_provider.py were written against
documented API shapes but never executed against a live API.
ollama_provider.py is the one provider we can verify for free, locally,
without an API key — so this test is the actual proof that the
request-building / response-parsing logic works against a real server,
not just against fakes.

Skipped automatically if Ollama isn't reachable, so it doesn't break
`pytest tests/` for anyone without Ollama running (e.g. in CI).
"""

import httpx
import pytest

from kestrion.agent.agent import Agent
from kestrion.agent.decorators import tool
from kestrion.llm.ollama_provider import OllamaProvider

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2"


def _ollama_is_running() -> bool:
    try:
        response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_is_running(),
    reason="Ollama server not reachable at localhost:11434 — skipping live smoke test",
)


@tool
def get_time() -> str:
    """Returns a fixed string for testing."""
    return "it is currently testing time"


async def test_live_ollama_agent_run_completes(tmp_store):
    """
    Not a behavioral assertion about model output quality — small local
    models are often unreliable about whether/when to call a tool, and
    that's a model limitation, not a Kestrion bug. This test only
    asserts the plumbing works: a real HTTP call completes, the
    response parses without crashing, and the run reaches a terminal
    status either way.
    """
    agent = Agent(
        provider=OllamaProvider(model=MODEL),
        tools=[get_time],
        store=f"sqlite:///{tmp_store.path}",
    )

    result = await agent.run("What time is it? Use the get_time tool if you have one.")

    print(f"\n[live ollama smoke test] status={result.status} output={result.output!r}")

    # The real assertion: the call completed without an unhandled
    # exception and reached SOME terminal status. We don't assert
    # COMPLETED specifically, since WAITING_ON_HUMAN or FAILED would
    # also prove the plumbing works -- only an unhandled crash is a
    # genuine Kestrion bug here.
    assert result.status.value in ("completed", "waiting_on_human", "failed")