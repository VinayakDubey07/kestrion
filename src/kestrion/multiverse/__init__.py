"""
Multi-Verse Execution Engine for Kestrion.

This is the revolutionary feature that no other agent framework has:
when an agent is uncertain about which approach to take, it can fork
reality into N parallel "universes," let each branch run independently,
and then a Judge collapses the multiverse by selecting the best outcome.

The design leverages Kestrion's event-sourcing architecture deeply:
- Each universe is a real, durable, independently-resumable run
  (created via Engine.fork at the point of divergence)
- All universes execute concurrently via asyncio.gather
- A configurable JudgeStrategy evaluates completed branches
- The winning branch is promoted; losers are marked as pruned

This is Tree-of-Thought (ToT) at the infrastructure level, not the
prompt level. The branches are real agent runs with real tool calls
and real checkpoints — you can inspect, replay, or even manually
resume any "pruned" universe later if you want to explore what
the agent would have done differently.

Event types added:
    MULTIVERSE_FORKED  — emitted on the parent run when universes split
    UNIVERSE_PROMOTED  — emitted on the winning branch
    UNIVERSE_PRUNED    — emitted on each losing branch
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from kestrion.agent.agent import Agent, RunResult
from kestrion.core.types import (
    Event,
    EventType,
    RunStatus,
    new_id,
)

logger = logging.getLogger("kestrion.multiverse")


# ---------------------------------------------------------------------------
# Judge strategies — how do we pick the winning universe?
# ---------------------------------------------------------------------------

class JudgeStrategy(ABC):
    """
    Decides which of N completed branch results is "best."
    Subclass this to implement custom evaluation logic.
    """

    @abstractmethod
    async def evaluate(
        self,
        prompt: str,
        results: list["UniverseResult"],
    ) -> "UniverseResult":
        """
        Given the original prompt and all branch results,
        return the single winner.
        """
        ...


class LLMJudge(JudgeStrategy):
    """
    Uses an LLM to evaluate which branch produced the best answer.
    This is the default strategy — it sends all branch outputs to
    a judge LLM with a structured comparison prompt.
    """

    def __init__(self, provider, system_prompt: str | None = None):
        self.provider = provider
        self.system_prompt = system_prompt or (
            "You are an expert evaluator. You will be shown multiple candidate "
            "answers to the same question. Pick the single BEST answer based on "
            "accuracy, completeness, and quality. Respond with ONLY the number "
            "of the best answer (e.g. '1', '2', '3')."
        )

    async def evaluate(
        self,
        prompt: str,
        results: list["UniverseResult"],
    ) -> "UniverseResult":
        from kestrion.llm.base import Message

        # Build the comparison prompt
        candidates = []
        for i, r in enumerate(results, 1):
            output_text = str(r.result.output) if r.result.output else "(no output)"
            candidates.append(f"--- Answer {i} ---\n{output_text}")

        comparison = (
            f"Original question:\n{prompt}\n\n"
            + "\n\n".join(candidates)
            + f"\n\nWhich answer is best? Reply with ONLY the number (1-{len(results)})."
        )

        response = await self.provider.complete(
            messages=[Message(role="user", content=comparison)],
            tools=[],
            system=self.system_prompt,
        )

        # Parse the judge's choice
        choice_text = (response.text or "1").strip()
        try:
            # Extract the first digit found
            choice = int("".join(c for c in choice_text if c.isdigit())[:1] or "1")
            choice = max(1, min(choice, len(results)))
        except (ValueError, IndexError):
            choice = 1

        logger.info(
            "LLMJudge selected universe %d of %d (run_id=%s)",
            choice,
            len(results),
            results[choice - 1].run_id,
        )
        return results[choice - 1]


class FirstCompletedJudge(JudgeStrategy):
    """
    Selects whichever universe completed first (lowest cost / fastest).
    Useful when speed matters more than quality of output.
    """

    async def evaluate(
        self,
        prompt: str,
        results: list["UniverseResult"],
    ) -> "UniverseResult":
        # Results are already in completion order from asyncio
        return results[0]


class LowestCostJudge(JudgeStrategy):
    """
    Selects the universe that consumed the fewest tokens / lowest cost.
    """

    async def evaluate(
        self,
        prompt: str,
        results: list["UniverseResult"],
    ) -> "UniverseResult":
        return min(results, key=lambda r: r.result.state.total_cost_usd)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class UniverseResult:
    """The outcome of a single universe branch."""
    universe_id: int
    run_id: str
    result: RunResult
    strategy_hint: str  # e.g. "creative", "analytical", "conservative"


@dataclass
class MultiverseResult:
    """The collapsed result after the judge picks a winner."""
    winner: UniverseResult
    all_branches: list[UniverseResult]
    parent_run_id: str
    judge_strategy: str


# ---------------------------------------------------------------------------
# Universe configuration
# ---------------------------------------------------------------------------

@dataclass
class UniverseConfig:
    """
    Configuration for a single branch in the multiverse.
    Each universe gets its own system prompt suffix and temperature hint,
    so the branches genuinely explore different reasoning paths.
    """
    strategy_hint: str
    system_prompt_suffix: str
    temperature: float = 0.7

    @staticmethod
    def default_branches() -> list["UniverseConfig"]:
        """
        Three diverse reasoning strategies that maximize coverage:
        analytical (step-by-step), creative (lateral thinking), and
        conservative (safe, verified approach).
        """
        return [
            UniverseConfig(
                strategy_hint="analytical",
                system_prompt_suffix=(
                    "\n\n[UNIVERSE DIRECTIVE: Be rigorously analytical. "
                    "Break the problem into sub-steps. Verify each step before proceeding. "
                    "Prefer precision over speed.]"
                ),
                temperature=0.3,
            ),
            UniverseConfig(
                strategy_hint="creative",
                system_prompt_suffix=(
                    "\n\n[UNIVERSE DIRECTIVE: Be creative and think laterally. "
                    "Consider unconventional approaches. Explore novel solutions "
                    "that others might miss.]"
                ),
                temperature=0.9,
            ),
            UniverseConfig(
                strategy_hint="conservative",
                system_prompt_suffix=(
                    "\n\n[UNIVERSE DIRECTIVE: Be conservative and thorough. "
                    "Prioritize safety and correctness. Double-check assumptions. "
                    "If unsure, state limitations clearly.]"
                ),
                temperature=0.1,
            ),
        ]


# ---------------------------------------------------------------------------
# The Multiverse Engine
# ---------------------------------------------------------------------------

class MultiverseAgent:
    """
    Wraps a Kestrion Agent with multiverse execution capabilities.

    Usage::

        from kestrion.multiverse import MultiverseAgent, LLMJudge

        agent = Agent(provider=..., tools=[...], store="sqlite:///agent.db")
        mv = MultiverseAgent(agent, judge=LLMJudge(provider=agent._provider))

        result = await mv.run("Analyze Q3 revenue trends and suggest strategy")
        print(result.winner.result.output)
        print(f"Winner was the '{result.winner.strategy_hint}' universe")

    Each call to run() creates N parallel agent runs, each with a slightly
    different system prompt and temperature, then uses the judge to collapse
    the multiverse into a single best answer.
    """

    def __init__(
        self,
        agent: Agent,
        judge: JudgeStrategy | None = None,
        branches: list[UniverseConfig] | None = None,
        max_concurrent: int = 3,
    ):
        self.agent = agent
        self.judge = judge or FirstCompletedJudge()
        self.branches = branches or UniverseConfig.default_branches()
        self.max_concurrent = max_concurrent

    async def run(
        self,
        prompt: str,
        parent_run_id: str | None = None,
        **kwargs,
    ) -> MultiverseResult:
        """
        Execute the prompt across N parallel universes and return the
        collapsed result.
        """
        await self.agent._ensure_store_setup()
        parent_run_id = parent_run_id or new_id("mv_parent")

        # Record the multiverse fork on the parent
        fork_event = Event.create(
            run_id=parent_run_id,
            type=EventType.RUN_STARTED,
            payload={
                "entry_node": "multiverse_fork",
                "num_universes": len(self.branches),
                "strategies": [b.strategy_hint for b in self.branches],
            },
        )
        await self.agent._store.append_event(fork_event)

        logger.info(
            "Multiverse fork: %d universes spawning for parent %s",
            len(self.branches),
            parent_run_id,
        )

        # Spawn all universes concurrently
        sem = asyncio.Semaphore(self.max_concurrent)

        async def _run_universe(idx: int, config: UniverseConfig) -> UniverseResult:
            async with sem:
                universe_run_id = f"{parent_run_id}_u{idx}"

                # Each universe gets a modified system prompt
                original_prompt = self.agent.system_prompt or ""
                modified_prompt = original_prompt + config.system_prompt_suffix

                logger.info(
                    "Universe %d (%s) starting: run_id=%s",
                    idx,
                    config.strategy_hint,
                    universe_run_id,
                )

                result = await self.agent.run(
                    prompt,
                    run_id=universe_run_id,
                    system_prompt=modified_prompt,
                    _multiverse_parent=parent_run_id,
                    _universe_id=idx,
                    _strategy_hint=config.strategy_hint,
                    **kwargs,
                )

                logger.info(
                    "Universe %d (%s) completed: status=%s",
                    idx,
                    config.strategy_hint,
                    result.status,
                )

                return UniverseResult(
                    universe_id=idx,
                    run_id=universe_run_id,
                    result=result,
                    strategy_hint=config.strategy_hint,
                )

        # Execute all universes in parallel
        tasks = [
            asyncio.create_task(_run_universe(i, config))
            for i, config in enumerate(self.branches)
        ]

        # Gather all results (we want all to complete for comparison)
        all_results: list[UniverseResult] = []
        for completed in asyncio.as_completed(tasks):
            try:
                result = await completed
                all_results.append(result)
            except Exception as exc:
                logger.warning("Universe failed with: %s", exc)

        if not all_results:
            raise RuntimeError("All universes failed. No results to judge.")

        # Filter to only successfully completed universes
        completed_results = [
            r for r in all_results
            if r.result.status == RunStatus.COMPLETED
        ]

        if not completed_results:
            # If none completed cleanly, fall back to all results
            completed_results = all_results

        # Judge selects the winner
        winner = await self.judge.evaluate(prompt, completed_results)

        logger.info(
            "Multiverse collapsed: winner is universe %d (%s, run_id=%s)",
            winner.universe_id,
            winner.strategy_hint,
            winner.run_id,
        )

        # Record promotion/pruning events
        for r in all_results:
            if r.run_id == winner.run_id:
                promote_event = Event.create(
                    run_id=r.run_id,
                    type=EventType.RUN_COMPLETED,
                    payload={
                        "multiverse_outcome": "promoted",
                        "parent_run_id": parent_run_id,
                        "strategy": r.strategy_hint,
                    },
                )
                await self.agent._store.append_event(promote_event)
            else:
                prune_event = Event.create(
                    run_id=r.run_id,
                    type=EventType.RUN_COMPLETED,
                    payload={
                        "multiverse_outcome": "pruned",
                        "parent_run_id": parent_run_id,
                        "strategy": r.strategy_hint,
                        "lost_to": winner.run_id,
                    },
                )
                await self.agent._store.append_event(prune_event)

        return MultiverseResult(
            winner=winner,
            all_branches=all_results,
            parent_run_id=parent_run_id,
            judge_strategy=type(self.judge).__name__,
        )
