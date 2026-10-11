"""
Tests for the Multiverse Execution Engine.
"""

import pytest

from kestrion.agent.agent import Agent, RunResult
from kestrion.core.types import AgentState, RunStatus
from kestrion.multiverse import (
    MultiverseAgent,
    LLMJudge,
    FirstCompletedJudge,
    LowestCostJudge,
    UniverseConfig,
    UniverseResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class MockProvider:
    """A mock LLM provider that returns deterministic responses."""

    def __init__(self, response_text="Mock response"):
        self.response_text = response_text

    async def complete(self, **kwargs):
        from kestrion.llm.base import LLMResponse
        return LLMResponse(
            text=self.response_text,
            stop_reason="end_turn",
            tool_calls=[],
            tokens_in=10,
            tokens_out=20,
            cost_usd=0.001,
        )


class MockJudgeProvider:
    """Returns "1", "2", or "3" to simulate judge decisions."""

    def __init__(self, choice: str = "1"):
        self.choice = choice

    async def complete(self, **kwargs):
        from kestrion.llm.base import LLMResponse
        return LLMResponse(
            text=self.choice,
            stop_reason="end_turn",
            tool_calls=[],
            tokens_in=5,
            tokens_out=1,
            cost_usd=0.0001,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiverse_basic_three_branches():
    """Three universes should all complete and a winner should be selected."""
    provider = MockProvider(response_text="The answer is 42.")
    agent = Agent(
        provider=provider,
        tools=[],
        store="sqlite:///test_multiverse.db",
        system_prompt="You are a helpful assistant.",
    )

    mv = MultiverseAgent(
        agent,
        judge=FirstCompletedJudge(),
    )

    result = await mv.run("What is the meaning of life?")

    assert result.winner is not None
    assert len(result.all_branches) == 3
    assert result.winner.result.status == RunStatus.COMPLETED
    assert result.judge_strategy == "FirstCompletedJudge"

    # All branches should have completed
    for branch in result.all_branches:
        assert branch.result.status == RunStatus.COMPLETED
        assert branch.strategy_hint in ("analytical", "creative", "conservative")


@pytest.mark.asyncio
async def test_multiverse_custom_branches():
    """Custom universe configurations should be respected."""
    provider = MockProvider()
    agent = Agent(
        provider=provider,
        tools=[],
        store="sqlite:///test_multiverse_custom.db",
    )

    custom_branches = [
        UniverseConfig(
            strategy_hint="fast",
            system_prompt_suffix="\n[Be fast]",
            temperature=0.1,
        ),
        UniverseConfig(
            strategy_hint="thorough",
            system_prompt_suffix="\n[Be thorough]",
            temperature=0.5,
        ),
    ]

    mv = MultiverseAgent(
        agent,
        judge=FirstCompletedJudge(),
        branches=custom_branches,
    )

    result = await mv.run("Test prompt")

    assert len(result.all_branches) == 2
    hints = {b.strategy_hint for b in result.all_branches}
    assert hints == {"fast", "thorough"}


@pytest.mark.asyncio
async def test_llm_judge_selects_best():
    """LLMJudge should use an LLM to pick the winner."""
    provider = MockProvider()
    judge_provider = MockJudgeProvider(choice="2")

    agent = Agent(
        provider=provider,
        tools=[],
        store="sqlite:///test_multiverse_judge.db",
    )

    mv = MultiverseAgent(
        agent,
        judge=LLMJudge(provider=judge_provider),
    )

    result = await mv.run("Analyze this data")

    # Judge said "2", so universe index 1 should win
    assert result.winner.universe_id == 1
    assert result.judge_strategy == "LLMJudge"


@pytest.mark.asyncio
async def test_lowest_cost_judge():
    """LowestCostJudge should pick the cheapest universe."""
    # Create fake UniverseResults with different costs
    results = []
    for i, cost in enumerate([0.05, 0.01, 0.03]):
        state = AgentState(run_id=f"run_{i}", status=RunStatus.COMPLETED)
        state.total_cost_usd = cost
        rr = RunResult(run_id=f"run_{i}", status=RunStatus.COMPLETED, output=f"Answer {i}", state=state)
        results.append(UniverseResult(
            universe_id=i,
            run_id=f"run_{i}",
            result=rr,
            strategy_hint=f"strategy_{i}",
        ))

    judge = LowestCostJudge()
    winner = await judge.evaluate("test", results)

    assert winner.run_id == "run_1"  # $0.01 is cheapest
    assert winner.universe_id == 1


@pytest.mark.asyncio
async def test_multiverse_result_structure():
    """MultiverseResult should have all expected fields."""
    provider = MockProvider()
    agent = Agent(
        provider=provider,
        tools=[],
        store="sqlite:///test_multiverse_struct.db",
    )

    mv = MultiverseAgent(agent, judge=FirstCompletedJudge())
    result = await mv.run("Hello")

    assert result.parent_run_id.startswith("mv_parent_")
    assert result.winner in result.all_branches
    assert isinstance(result.judge_strategy, str)
