import asyncio
import pytest
from unittest.mock import AsyncMock

from kestrion.core.engine import Engine
from kestrion.core.nodes import SummarizationNode
from kestrion.core.types import (
    AgentState,
    Event,
    EventType,
    Node,
    NodeResult,
    RunStatus,
)
from kestrion.llm.anthropic_provider import AnthropicProvider
from kestrion.llm.base import ImageBlock, LLMResponse, Message
from kestrion.llm.ollama_provider import OllamaProvider
from kestrion.llm.openai_provider import OpenAIProvider
from kestrion.scheduler.rate_limiter import RateLimiter, RateLimiterConfig
from kestrion.store.memory_store import MemoryCheckpointStore
from kestrion.tools.sandbox import CodeSandboxToolkit
from kestrion.adapters.vercel import _format_vercel_events


class DummyNode(Node):
    name = "dummy"

    async def run(self, state: AgentState) -> NodeResult:
        return NodeResult(next_node=None, state_updates={})


# ----------------------------------------------------------------------
# Issue #18: fork() scratchpad restoration
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_engine_fork_preserves_scratch_and_checkpoints():
    store = MemoryCheckpointStore()
    engine = Engine(nodes={"dummy": DummyNode()}, tools={}, store=store, entry_node="dummy")

    run_id = "run_orig"
    state = AgentState(
        run_id=run_id,
        status=RunStatus.RUNNING,
        current_node="dummy",
        scratch={"counter": 100, "_messages": [{"role": "user", "content": "hello"}]},
    )

    await engine._emit(state, EventType.RUN_STARTED, {"entry_node": "dummy"})
    await engine._checkpoint(state)
    await engine._emit(state, EventType.LLM_CALL_STARTED, {})

    forked_id = await engine.fork(
        run_id,
        at_seq=2,
        new_run_id="run_forked",
        scratch_override={"counter": 200, "new_var": True},
    )

    ckpt = await store.latest(forked_id)
    assert ckpt is not None
    forked_state = ckpt.state
    assert forked_state.run_id == "run_forked"
    assert forked_state.scratch["counter"] == 200
    assert forked_state.scratch["new_var"] is True
    assert forked_state.scratch["_messages"] == [{"role": "user", "content": "hello"}]


# ----------------------------------------------------------------------
# Issue #19: RateLimiter deadlock when tokens > tokens_per_minute
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_rate_limiter_exceeds_tpm_no_deadlock():
    limiter = RateLimiter(
        RateLimiterConfig(
            requests_per_minute=6000.0,
            tokens_per_minute=500.0,
        )
    )
    # Asking for 2000 tokens when bucket capacity is 500
    # Must drain available tokens and succeed without infinite loop
    await asyncio.wait_for(limiter.acquire(tokens=2000), timeout=1.0)
    assert limiter._tok_tokens == 0.0


# ----------------------------------------------------------------------
# Issue #20: Sandbox unique script naming & concurrency
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sandbox_concurrent_script_execution(tmp_path):
    toolkit = CodeSandboxToolkit(workspace_dir=tmp_path, mode="subprocess")

    results = await asyncio.gather(
        toolkit.execute_python("print('SCRIPT_1')"),
        toolkit.execute_python("print('SCRIPT_2')"),
        toolkit.execute_python("print('SCRIPT_3')"),
    )

    assert any("SCRIPT_1" in r for r in results)
    assert any("SCRIPT_2" in r for r in results)
    assert any("SCRIPT_3" in r for r in results)
    # Verify no leaked _exec_ script files in workspace
    assert len(list(tmp_path.glob("_exec_*.py"))) == 0


# ----------------------------------------------------------------------
# Issue #21: Multimodal ImageBlock tool outputs preserved
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_multimodal_imageblock_tool_output():
    img = ImageBlock(data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==", media_type="image/png")
    tool_msg = Message(role="tool", tool_call_id="call_123", content=[img])

    provider = AnthropicProvider(model="claude-3-5-sonnet-20241022", api_key="dummy")
    anthropic_msgs = provider._to_anthropic_messages([tool_msg])

    assert len(anthropic_msgs) == 1
    assert anthropic_msgs[0]["role"] == "user"
    assert anthropic_msgs[0]["content"][0]["type"] == "tool_result"
    assert anthropic_msgs[0]["content"][0]["tool_use_id"] == "call_123"
    result_content = anthropic_msgs[0]["content"][0]["content"]
    assert isinstance(result_content, list)
    assert result_content[0]["type"] == "image"
    assert result_content[0]["source"]["media_type"] == "image/png"


# ----------------------------------------------------------------------
# Issue #22: SummarizationNode system prompt & tool pair atomicity
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_summarization_node_preserves_system_prompt_and_tool_pairs():
    mock_provider = AsyncMock()
    mock_provider.complete.return_value = LLMResponse(text="Summary of previous turns")

    node = SummarizationNode(
        provider=mock_provider,
        next_node=None,
        max_history_turns=2,
        keep_turns=2,
    )

    messages = [
        {"role": "system", "content": "System prompt instructions"},
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Call tool", "tool_calls": [{"id": "t1", "name": "lookup"}]},
        {"role": "tool", "content": "Result 1", "tool_call_id": "t1"},
        {"role": "user", "content": "Question 2"},
    ]

    state = AgentState(
        run_id="run_sum",
        status=RunStatus.RUNNING,
        scratch={"_messages": messages},
    )

    result = await node.run(state)
    compacted = result.state_updates["_messages"]

    # Original system prompt must remain at index 0
    assert compacted[0]["role"] == "system"
    assert compacted[0]["content"] == "System prompt instructions"

    # Tool result must never appear without its preceding assistant tool call
    for idx, m in enumerate(compacted):
        if m.get("role") == "tool":
            assert idx > 0
            assert compacted[idx - 1].get("role") == "assistant"


# ----------------------------------------------------------------------
# Issue #23: Vercel adapter toolCallId matching
# ----------------------------------------------------------------------
def test_vercel_adapter_tool_call_id_matching():
    ev_completed = Event.create(
        run_id="run_v",
        type=EventType.TOOL_CALL_COMPLETED,
        payload={"tool": "weather", "output": "sunny", "tool_call_id": "call_abc123"},
    )
    chunks = _format_vercel_events([ev_completed])
    assert len(chunks) == 1
    assert "call_abc123" in chunks[0]
    assert '"toolCallId": "call_abc123"' in chunks[0]


# ----------------------------------------------------------------------
# Issue #24: LLM providers system prompt deduplication
# ----------------------------------------------------------------------
def test_llm_provider_system_prompt_deduplication():
    messages = [
        Message(role="system", content="Existing system instruction"),
        Message(role="user", content="Hello"),
    ]

    # OpenAI
    openai_prov = OpenAIProvider(model="gpt-4o", api_key="dummy")
    openai_msgs = openai_prov._to_openai_messages(messages, system="Existing system instruction")
    sys_count_openai = sum(1 for m in openai_msgs if m["role"] == "system")
    assert sys_count_openai == 1

    # Ollama
    ollama_prov = OllamaProvider(model="llama3.1")
    ollama_msgs = ollama_prov._to_ollama_messages(messages, system="Existing system instruction")
    sys_count_ollama = sum(1 for m in ollama_msgs if m["role"] == "system")
    assert sys_count_ollama == 1


# ----------------------------------------------------------------------
# Issue #25: Engine _emit live folds TOOL_CALL_COMPLETED
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_engine_emit_live_folds_tool_call_completed():
    store = MemoryCheckpointStore()
    engine = Engine(nodes={"dummy": DummyNode()}, tools={}, store=store, entry_node="dummy")
    state = AgentState(run_id="run_fold", status=RunStatus.RUNNING, current_node="dummy")

    await engine._emit(
        state,
        EventType.TOOL_CALL_COMPLETED,
        {"tool": "calculator", "output": "42", "tool_call_id": "c1"},
    )

    assert len(state.history) == 1
    assert state.history[0]["type"] == "tool_result"
    assert state.history[0]["tool"] == "calculator"
    assert state.history[0]["output"] == "42"
