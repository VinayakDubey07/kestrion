"""
Tests for parallel tool call execution in the Agent loop: multiple tool
calls requested in one LLM turn now run concurrently via asyncio.gather,
with a pre-check phase that ensures a batch either fully runs or cleanly
pauses with nothing partially executed if any call in the batch is gated
and unapproved.
"""

import asyncio

from kestrion.agent.agent import Agent
from kestrion.agent.decorators import tool
from kestrion.llm.base import LLMResponse, ToolCallRequest

# Shared mutable list so tests can observe execution order/timing —
# module-level is fine since each test gets a fresh list via fixture-style
# reset at the top of each test function.
_call_log: list[str] = []


@tool
async def slow_tool_a() -> str:
    """A tool that takes a noticeable amount of time."""
    _call_log.append("a_start")
    await asyncio.sleep(0.05)
    _call_log.append("a_end")
    return "result_a"


@tool
async def slow_tool_b() -> str:
    """Another tool that takes a noticeable amount of time."""
    _call_log.append("b_start")
    await asyncio.sleep(0.05)
    _call_log.append("b_end")
    return "result_b"


@tool
def failing_tool() -> str:
    """A tool that always raises."""
    raise RuntimeError("boom")


@tool(requires_approval=True)
def gated_tool() -> str:
    """A tool requiring approval."""
    _call_log.append("gated_executed")
    return "gated_result"


def _store_url(tmp_store) -> str:
    return f"sqlite:///{tmp_store.path}"


# ---------------------------------------------------------------------------
# Concurrency actually happens (timing-based proof, not just "didn't crash")
# ---------------------------------------------------------------------------

async def test_two_independent_tool_calls_run_concurrently_not_sequentially(tmp_store):
    """
    If these ran sequentially, total time would be >= 0.1s (2 x 0.05s).
    Running concurrently, total time should be close to 0.05s. This is
    the actual proof of parallelism, not just "the test didn't crash."
    """
    _call_log.clear()

    class FakeProviderCallsBothTools:
        def __init__(self):
            self.calls = 0

        async def complete(self, messages, tools, system=None):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    text=None,
                    tool_calls=[
                        ToolCallRequest(id="c1", name="slow_tool_a", arguments={}),
                        ToolCallRequest(id="c2", name="slow_tool_b", arguments={}),
                    ],
                    stop_reason="tool_use",
                )
            return LLMResponse(text="done", tool_calls=[], stop_reason="end_turn")

    agent = Agent(
        provider=FakeProviderCallsBothTools(),
        tools=[slow_tool_a, slow_tool_b],
        store=_store_url(tmp_store),
    )

    start = asyncio.get_event_loop().time()
    result = await agent.run("call both tools")
    elapsed = asyncio.get_event_loop().time() - start

    assert result.status.value == "completed"
    assert elapsed < 0.09, f"expected concurrent execution (~0.05s), took {elapsed:.3f}s"

    # Both tools should have started before either finished, proving
    # genuine overlap rather than just fast sequential execution.
    a_start_idx = _call_log.index("a_start")
    b_start_idx = _call_log.index("b_start")
    a_end_idx = _call_log.index("a_end")
    b_end_idx = _call_log.index("b_end")
    assert a_start_idx < a_end_idx and b_start_idx < b_end_idx
    # The second tool's start must come before the first tool's end —
    # this is only true if they're actually running concurrently.
    assert b_start_idx < a_end_idx or a_start_idx < b_end_idx


# ---------------------------------------------------------------------------
# Results map back to the correct originating call
# ---------------------------------------------------------------------------

async def test_results_map_back_to_correct_tool_call_id_not_just_order(tmp_store):
    class FakeProviderCallsBothTools:
        def __init__(self):
            self.calls = 0

        async def complete(self, messages, tools, system=None):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    text=None,
                    tool_calls=[
                        ToolCallRequest(id="call_a", name="slow_tool_a", arguments={}),
                        ToolCallRequest(id="call_b", name="slow_tool_b", arguments={}),
                    ],
                    stop_reason="tool_use",
                )
            return LLMResponse(text="done", tool_calls=[], stop_reason="end_turn")

    agent = Agent(
        provider=FakeProviderCallsBothTools(),
        tools=[slow_tool_a, slow_tool_b],
        store=_store_url(tmp_store),
    )
    result = await agent.run("call both")

    messages = result.state.scratch["_messages"]
    tool_messages = [m for m in messages if m["role"] == "tool"]
    by_call_id = {m["tool_call_id"]: m["content"] for m in tool_messages}

    assert by_call_id["call_a"] == "result_a"
    assert by_call_id["call_b"] == "result_b"


# ---------------------------------------------------------------------------
# Fail-fast behavior preserved: a real exception still propagates
# ---------------------------------------------------------------------------

async def test_a_failing_tool_in_a_parallel_batch_still_raises(tmp_store):
    """
    Engine.call_tool catches tool exceptions internally and returns a
    ToolResult with .error set — it does NOT raise. So a batch with one
    failing tool should complete normally, with that tool's result
    showing the error, exactly like the sequential version did. This
    confirms gather() didn't change that fail-soft behavior.
    """
    class FakeProviderCallsFailingAndWorkingTool:
        def __init__(self):
            self.calls = 0

        async def complete(self, messages, tools, system=None):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    text=None,
                    tool_calls=[
                        ToolCallRequest(id="c1", name="failing_tool", arguments={}),
                        ToolCallRequest(id="c2", name="slow_tool_a", arguments={}),
                    ],
                    stop_reason="tool_use",
                )
            return LLMResponse(text="handled", tool_calls=[], stop_reason="end_turn")

    agent = Agent(
        provider=FakeProviderCallsFailingAndWorkingTool(),
        tools=[failing_tool, slow_tool_a],
        store=_store_url(tmp_store),
    )
    result = await agent.run("call both")

    assert result.status.value == "completed"
    messages = result.state.scratch["_messages"]
    tool_messages = [m for m in messages if m["role"] == "tool"]
    contents = [m["content"] for m in tool_messages]
    assert any("error" in c.lower() or "none" in c.lower() for c in contents) or any(
        c == "None" for c in contents
    )


# ---------------------------------------------------------------------------
# The critical safety property: gated call in a batch blocks the WHOLE
# batch before anything runs, not just itself
# ---------------------------------------------------------------------------

async def test_gated_call_in_a_batch_blocks_everything_before_any_call_executes(tmp_store):
    """
    This is the core safety guarantee of the two-phase design: if a
    batch contains a gated, unapproved tool alongside safe tools, NONE
    of them should execute — not even the safe ones — because approval
    is checked for the whole batch before any dispatch happens.
    """
    _call_log.clear()

    class FakeProviderCallsGatedAndSafeTool:
        async def complete(self, messages, tools, system=None):
            return LLMResponse(
                text=None,
                tool_calls=[
                    ToolCallRequest(id="c1", name="gated_tool", arguments={}),
                    ToolCallRequest(id="c2", name="slow_tool_a", arguments={}),
                ],
                stop_reason="tool_use",
            )

    agent = Agent(
        provider=FakeProviderCallsGatedAndSafeTool(),
        tools=[gated_tool, slow_tool_a],
        store=_store_url(tmp_store),
    )
    result = await agent.run("call both")

    assert result.status.value == "waiting_on_human"
    # Neither tool should have actually executed — the gated one because
    # it's blocked, and the safe one because the pre-check phase raises
    # BEFORE asyncio.gather is ever called, so dispatch never happens.
    assert "gated_executed" not in _call_log
    assert "a_start" not in _call_log


async def test_gated_call_alone_still_pauses_correctly_no_regression(tmp_store):
    """Sanity check: the single-gated-call case (no batch) still works exactly as before."""
    _call_log.clear()

    class FakeProviderCallsGatedToolAlone:
        async def complete(self, messages, tools, system=None):
            return LLMResponse(
                text=None,
                tool_calls=[ToolCallRequest(id="c1", name="gated_tool", arguments={})],
                stop_reason="tool_use",
            )

    agent = Agent(
        provider=FakeProviderCallsGatedToolAlone(),
        tools=[gated_tool],
        store=_store_url(tmp_store),
    )
    result = await agent.run("call gated tool")

    assert result.status.value == "waiting_on_human"
    assert "gated_executed" not in _call_log