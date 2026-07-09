"""
Tests for the kestrion.scheduler package.

All tests use AsyncMock agents — no real LLM calls required. The goal is
to verify the scheduler's structural behaviour:
  - RateLimiter token-bucket mechanics and backoff
  - WorkerPool concurrency bounding
  - Pipeline DAG validation (cycle detection, missing deps)
  - Pipeline dependency-gated execution order
  - Pipeline fail_fast vs. continue-on-failure behaviour
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrion.scheduler import (
    AgentTask,
    Pipeline,
    RateLimiter,
    RateLimiterConfig,
    TaskStatus,
    WorkerPool,
)
from kestrion.agent.agent import RunResult
from kestrion.core.types import RunStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_agent(output: str = "done", status: RunStatus = RunStatus.COMPLETED) -> MagicMock:
    """Return a mock Agent whose .run() resolves immediately with a RunResult."""
    agent = MagicMock()
    result = RunResult(
        run_id="run_test_abc123",
        status=status,
        output=output,
        state=MagicMock(),
    )
    agent.run = AsyncMock(return_value=result)
    return agent


def make_failing_agent() -> MagicMock:
    agent = MagicMock()
    agent.run = AsyncMock(side_effect=RuntimeError("LLM error"))
    return agent


# ---------------------------------------------------------------------------
# RateLimiter tests
# ---------------------------------------------------------------------------

class TestRateLimiter:

    @pytest.mark.asyncio
    async def test_acquire_no_limits_immediate(self):
        """With no limits configured, acquire() returns immediately."""
        limiter = RateLimiter(RateLimiterConfig())
        t0 = time.monotonic()
        await limiter.acquire(tokens=1000)
        assert time.monotonic() - t0 < 0.1

    @pytest.mark.asyncio
    async def test_on_rate_limited_sets_backoff(self):
        """on_rate_limited() with explicit retry_after sets a backoff window."""
        limiter = RateLimiter(RateLimiterConfig())
        await limiter.on_rate_limited(retry_after=0.2)
        # _backoff_until should be in the future
        assert limiter._backoff_until > time.monotonic()

    @pytest.mark.asyncio
    async def test_on_rate_limited_exponential_growth(self):
        """Consecutive 429s produce increasing backoffs (before jitter)."""
        config = RateLimiterConfig(base_backoff_seconds=1.0, max_backoff_seconds=60.0)
        limiter = RateLimiter(config)
        # Simulate 3 consecutive 429s and record each backoff window
        backoffs = []
        for _ in range(3):
            t0 = time.monotonic()
            await limiter.on_rate_limited()
            backoffs.append(limiter._backoff_until - t0)
        # Each backoff should be larger than the previous (net of jitter)
        # Base * 2^0 = 1, Base * 2^1 = 2, Base * 2^2 = 4 — allow for jitter
        assert backoffs[1] > backoffs[0] * 0.5
        assert backoffs[2] > backoffs[0] * 0.5

    def test_reset_backoff_clears_counter(self):
        """reset_backoff() resets the consecutive-429 counter."""
        limiter = RateLimiter()
        limiter._consecutive_429s = 5
        limiter.reset_backoff()
        assert limiter._consecutive_429s == 0

    @pytest.mark.asyncio
    async def test_rpm_bucket_refills_over_time(self):
        """RPM bucket: tokens are consumed and refill based on elapsed time."""
        # 60 RPM = 1 token per second
        config = RateLimiterConfig(requests_per_minute=60.0)
        limiter = RateLimiter(config)
        # Drain the bucket (it starts full = 60 tokens)
        limiter._req_tokens = 0.5  # half a token — not enough for one request
        # Simulate 1 second of elapsed time via _refill
        limiter._last_refill = time.monotonic() - 1.0
        now = time.monotonic()
        limiter._refill(now)
        # Should have refilled ~1 token (60 RPM / 60 seconds = 1/s)
        assert limiter._req_tokens >= 1.0


# ---------------------------------------------------------------------------
# WorkerPool tests
# ---------------------------------------------------------------------------

class TestWorkerPool:

    @pytest.mark.asyncio
    async def test_submit_runs_coroutine(self):
        """submit() executes and returns the coroutine's result."""
        pool = WorkerPool(max_workers=2)

        async def work():
            return 42

        result = await pool.submit(work())
        assert result == 42

    @pytest.mark.asyncio
    async def test_max_workers_limits_concurrency(self):
        """At most max_workers coroutines run at the same time."""
        max_workers = 2
        pool = WorkerPool(max_workers=max_workers)
        active = 0
        peak = 0

        async def track():
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.05)
            active -= 1

        # Submit more tasks than max_workers allows simultaneously
        await pool.map([track() for _ in range(6)])
        assert peak <= max_workers

    @pytest.mark.asyncio
    async def test_map_preserves_order(self):
        """map() returns results in submission order, not completion order."""
        pool = WorkerPool(max_workers=5)

        async def delayed(val: int, delay: float):
            await asyncio.sleep(delay)
            return val

        # Reverse delay so last submitted finishes first
        results = await pool.map([
            delayed(0, 0.04),
            delayed(1, 0.02),
            delayed(2, 0.01),
        ])
        assert results == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_submit_propagates_exception(self):
        """submit() re-raises exceptions from the coroutine."""
        pool = WorkerPool()

        async def boom():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await pool.submit(boom())


# ---------------------------------------------------------------------------
# Pipeline DAG validation tests
# ---------------------------------------------------------------------------

class TestPipelineValidation:

    def test_rejects_missing_dependency(self):
        """Pipeline raises ValueError if a depends_on name doesn't exist."""
        with pytest.raises(ValueError, match="no task with that name exists"):
            Pipeline([
                AgentTask("a", agent=make_mock_agent(), prompt="hi"),
                AgentTask("b", agent=make_mock_agent(), prompt="hi", depends_on=["nonexistent"]),
            ])

    def test_rejects_direct_cycle(self):
        """Pipeline raises ValueError on a direct A->B->A cycle."""
        with pytest.raises(ValueError, match="Cycle detected"):
            Pipeline([
                AgentTask("a", agent=make_mock_agent(), prompt="hi", depends_on=["b"]),
                AgentTask("b", agent=make_mock_agent(), prompt="hi", depends_on=["a"]),
            ])

    def test_rejects_indirect_cycle(self):
        """Pipeline raises ValueError on an indirect A->B->C->A cycle."""
        with pytest.raises(ValueError, match="Cycle detected"):
            Pipeline([
                AgentTask("a", agent=make_mock_agent(), prompt="hi", depends_on=["c"]),
                AgentTask("b", agent=make_mock_agent(), prompt="hi", depends_on=["a"]),
                AgentTask("c", agent=make_mock_agent(), prompt="hi", depends_on=["b"]),
            ])

    def test_accepts_valid_dag(self):
        """Pipeline accepts a valid DAG without raising."""
        pipeline = Pipeline([
            AgentTask("a", agent=make_mock_agent(), prompt="hi"),
            AgentTask("b", agent=make_mock_agent(), prompt="hi"),
            AgentTask("c", agent=make_mock_agent(), prompt="hi", depends_on=["a", "b"]),
        ])
        assert pipeline is not None


# ---------------------------------------------------------------------------
# Pipeline execution tests
# ---------------------------------------------------------------------------

class TestPipelineExecution:

    @pytest.mark.asyncio
    async def test_single_task_completes(self):
        """A single-task pipeline runs and returns COMPLETED."""
        agent = make_mock_agent(output="result text")
        pipeline = Pipeline([
            AgentTask("solo", agent=agent, prompt="do something"),
        ])
        results = await pipeline.run()

        assert "solo" in results
        assert results["solo"].status == TaskStatus.COMPLETED
        assert results["solo"].run_result.output == "result text"
        agent.run.assert_called_once_with("do something", run_id=None)

    @pytest.mark.asyncio
    async def test_independent_tasks_all_complete(self):
        """Three independent tasks all complete."""
        pipeline = Pipeline([
            AgentTask("a", agent=make_mock_agent("a_out"), prompt="a"),
            AgentTask("b", agent=make_mock_agent("b_out"), prompt="b"),
            AgentTask("c", agent=make_mock_agent("c_out"), prompt="c"),
        ])
        results = await pipeline.run()

        for name in ["a", "b", "c"]:
            assert results[name].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_dependency_order_respected(self):
        """Task C should start only after tasks A and B complete."""
        execution_order = []

        def make_tracking_agent(label):
            async def _run(prompt, run_id=None):
                execution_order.append(label)
                await asyncio.sleep(0.01)
                return RunResult("run_x", RunStatus.COMPLETED, label, MagicMock())
            agent = MagicMock()
            agent.run = _run
            return agent

        pipeline = Pipeline([
            AgentTask("a", agent=make_tracking_agent("A"), prompt="A"),
            AgentTask("b", agent=make_tracking_agent("B"), prompt="B"),
            AgentTask("c", agent=make_tracking_agent("C"), prompt="C", depends_on=["a", "b"]),
        ])
        await pipeline.run()

        # C must come after both A and B
        assert execution_order.index("C") > execution_order.index("A")
        assert execution_order.index("C") > execution_order.index("B")

    @pytest.mark.asyncio
    async def test_failed_task_marks_dependents_skipped(self):
        """When a task fails, its dependents are SKIPPED (fail_fast=False)."""
        pipeline = Pipeline([
            AgentTask("a", agent=make_failing_agent(), prompt="a"),
            AgentTask("b", agent=make_mock_agent(), prompt="b", depends_on=["a"]),
        ], fail_fast=False)
        results = await pipeline.run()

        assert results["a"].status == TaskStatus.FAILED
        assert results["b"].status == TaskStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_independent_task_runs_despite_sibling_failure(self):
        """An independent task completes even if an unrelated task fails."""
        pipeline = Pipeline([
            AgentTask("a", agent=make_failing_agent(), prompt="a"),
            AgentTask("independent", agent=make_mock_agent("ok"), prompt="b"),
        ], fail_fast=False)
        results = await pipeline.run()

        assert results["a"].status == TaskStatus.FAILED
        assert results["independent"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_fail_fast_aborts_remaining_tasks(self):
        """With fail_fast=True, remaining tasks are skipped after first failure."""
        pipeline = Pipeline([
            AgentTask("a", agent=make_failing_agent(), prompt="a"),
            AgentTask("b", agent=make_mock_agent(), prompt="b", depends_on=["a"]),
            AgentTask("c", agent=make_mock_agent(), prompt="c", depends_on=["b"]),
        ], fail_fast=True)
        results = await pipeline.run()

        assert results["a"].status == TaskStatus.FAILED
        # b and c should be skipped (not run)
        assert results["b"].status in (TaskStatus.SKIPPED, TaskStatus.FAILED)
        assert results["c"].status in (TaskStatus.SKIPPED, TaskStatus.FAILED)

    @pytest.mark.asyncio
    async def test_status_summary_includes_all_tasks(self):
        """status_summary() returns a string covering all task names."""
        pipeline = Pipeline([
            AgentTask("a", agent=make_mock_agent("hello"), prompt="a"),
        ])
        results = await pipeline.run()
        summary = pipeline.status_summary(results)
        assert "a" in summary
        assert "completed" in summary

    @pytest.mark.asyncio
    async def test_task_result_timing(self):
        """TaskResult.duration_seconds is populated after run."""
        pipeline = Pipeline([
            AgentTask("a", agent=make_mock_agent(), prompt="a"),
        ])
        results = await pipeline.run()
        assert results["a"].duration_seconds is not None
        assert results["a"].duration_seconds >= 0.0
