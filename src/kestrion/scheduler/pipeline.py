"""
DAG-based multi-agent orchestration pipeline.

This is the main user-facing API for the scheduler. A Pipeline accepts a
list of AgentTasks — each declaring its own dependencies — and executes
them concurrently, respecting both the dependency order and the shared
rate/concurrency limits.

Key design decisions:
  - Dependency resolution uses asyncio.Event per task. When a task
    completes, it sets its event; downstream tasks atomically unblock and
    move to the worker pool queue. No polling, no busy-wait.
  - Cycle detection is done eagerly at pipeline creation time (not at
    runtime), using DFS. A cyclic dependency raises ValueError immediately
    so the user knows before any agent is invoked.
  - Task failures don't abort the whole pipeline by default (fail_fast=False).
    Independent branches continue running. This is the safer default for
    long-running pipelines where partial results are still valuable.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kestrion.agent.agent import Agent, RunResult

from .rate_limiter import RateLimiter, RateLimiterConfig
from .worker_pool import WorkerPool

logger = logging.getLogger("kestrion.scheduler.pipeline")


class TaskStatus(str, Enum):
    PENDING   = "pending"
    WAITING   = "waiting"   # dependencies not yet met
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    SKIPPED   = "skipped"   # a dependency failed and fail_fast=False skips dependents


@dataclass
class AgentTask:
    """
    One node in the Pipeline DAG.

    Parameters
    ----------
    name:
        Unique name for this task within the pipeline.
    agent:
        The Kestrion Agent instance to run.
    prompt:
        The user prompt sent to the agent.
    depends_on:
        Names of tasks that must successfully complete before this task
        can start. Tasks with no dependencies start immediately.
    run_id:
        Optional explicit run_id. Auto-generated if not provided.
    estimated_tokens:
        Hint for the rate limiter — expected output tokens for this task.
        Affects TPM bucket draw; ignored if no TPM limit is configured.
    metadata:
        Arbitrary key-value pairs stored on the task for caller use.
        Not used by the pipeline engine itself.
    """
    name: str
    agent: Agent
    prompt: str
    depends_on: list[str] = field(default_factory=list)
    run_id: str | None = None
    estimated_tokens: int = 100
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """
    The outcome of one AgentTask execution.

    Wraps RunResult with pipeline-level metadata (timing, status, errors).
    """
    name: str
    status: TaskStatus
    run_result: RunResult | None = None
    error: str | None = None
    started_at: float | None = None   # monotonic time
    finished_at: float | None = None  # monotonic time

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is not None and self.finished_at is not None:
            return self.finished_at - self.started_at
        return None


class Pipeline:
    """
    Executes a list of AgentTasks as a DAG, with bounded concurrency and
    optional rate limiting.

    Usage::

        pipeline = Pipeline(
            tasks=[
                AgentTask("researcher_a", agent=agent_a, prompt="Research X"),
                AgentTask("researcher_b", agent=agent_b, prompt="Research Y"),
                AgentTask(
                    "synthesizer",
                    agent=synth_agent,
                    prompt="Combine the research findings",
                    depends_on=["researcher_a", "researcher_b"],
                ),
            ],
            max_workers=3,
            rate_limiter_config=RateLimiterConfig(requests_per_minute=60),
        )

        results = await pipeline.run()
        for name, task_result in results.items():
            print(f"{name}: {task_result.status} — {task_result.run_result.output}")

    Parameters
    ----------
    tasks:
        List of AgentTasks forming the DAG.
    max_workers:
        Maximum concurrent agent runs. Defaults to 5.
    rate_limiter_config:
        Rate limiting configuration. Pass None to disable rate limiting
        (appropriate for local Ollama or unlimited providers).
    fail_fast:
        If True, the pipeline aborts as soon as any task fails.
        If False (default), independent branches continue running;
        tasks that depend on a failed task are marked SKIPPED.
    """

    def __init__(
        self,
        tasks: list[AgentTask],
        max_workers: int = 5,
        rate_limiter_config: RateLimiterConfig | None = None,
        fail_fast: bool = False,
    ):
        self._tasks: dict[str, AgentTask] = {t.name: t for t in tasks}
        self._max_workers = max_workers
        self._rate_limiter = RateLimiter(rate_limiter_config) if rate_limiter_config else None
        self._fail_fast = fail_fast

        # Validate the dependency graph eagerly
        self._validate_dag()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> dict[str, TaskResult]:
        """
        Execute the pipeline to completion.

        Returns a dict mapping task name -> TaskResult for every task.
        The call blocks until all tasks are either COMPLETED, FAILED, or SKIPPED.
        """
        pool = WorkerPool(
            max_workers=self._max_workers,
            rate_limiter=self._rate_limiter,
        )

        # One asyncio.Event per task — set when the task finishes (any outcome)
        done_events: dict[str, asyncio.Event] = {
            name: asyncio.Event() for name in self._tasks
        }

        # Shared result dict — written by individual task coroutines
        results: dict[str, TaskResult] = {}

        # Shared abort flag — set when fail_fast is triggered
        abort_event = asyncio.Event()

        async def run_task(task: AgentTask) -> None:
            name = task.name

            # Wait for all dependencies to finish (any outcome)
            for dep_name in task.depends_on:
                await done_events[dep_name].wait()

            # If abort is signalled or any dependency failed, skip this task
            if abort_event.is_set():
                results[name] = TaskResult(name=name, status=TaskStatus.SKIPPED)
                done_events[name].set()
                return

            for dep_name in task.depends_on:
                dep_result = results.get(dep_name)
                if dep_result and dep_result.status in (TaskStatus.FAILED, TaskStatus.SKIPPED):
                    results[name] = TaskResult(
                        name=name,
                        status=TaskStatus.SKIPPED,
                        error=f"Dependency '{dep_name}' did not complete successfully",
                    )
                    done_events[name].set()
                    logger.info("Task '%s' skipped (dependency '%s' failed/skipped).", name, dep_name)
                    return

            results[name] = TaskResult(name=name, status=TaskStatus.RUNNING, started_at=time.monotonic())
            logger.info("Task '%s' starting.", name)

            try:
                run_result = await pool.submit(
                    task.agent.run(task.prompt, run_id=task.run_id),
                    estimated_tokens=task.estimated_tokens,
                )
                results[name] = TaskResult(
                    name=name,
                    status=TaskStatus.COMPLETED,
                    run_result=run_result,
                    started_at=results[name].started_at,
                    finished_at=time.monotonic(),
                )
                logger.info(
                    "Task '%s' completed (run_id=%s, status=%s).",
                    name,
                    run_result.run_id,
                    run_result.status.value,
                )
            except Exception as exc:
                results[name] = TaskResult(
                    name=name,
                    status=TaskStatus.FAILED,
                    error=str(exc),
                    started_at=results[name].started_at,
                    finished_at=time.monotonic(),
                )
                logger.error("Task '%s' failed: %s", name, exc)
                if self._fail_fast:
                    abort_event.set()
            finally:
                done_events[name].set()

        # Launch all tasks concurrently — each will self-gate on its
        # dependencies' events before actually doing work.
        await asyncio.gather(
            *[run_task(task) for task in self._tasks.values()],
            return_exceptions=True,  # don't let one task's exception kill gather()
        )

        return results

    def status_summary(self, results: dict[str, TaskResult]) -> str:
        """
        Human-readable summary of a results dict (for logging/printing).
        """
        lines = ["Pipeline results:"]
        for name, r in results.items():
            duration = f"({r.duration_seconds:.1f}s)" if r.duration_seconds else ""
            output_preview = ""
            if r.run_result and r.run_result.output:
                preview = r.run_result.output[:80].replace("\n", " ")
                output_preview = f" | {preview}..."
            error_info = f" | ERROR: {r.error}" if r.error else ""
            lines.append(f"  {name}: {r.status.value} {duration}{output_preview}{error_info}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal: DAG validation
    # ------------------------------------------------------------------

    def _validate_dag(self) -> None:
        """
        Validate all dependency names exist and detect cycles using DFS.
        Raises ValueError on the first problem found.
        """
        for task in self._tasks.values():
            for dep in task.depends_on:
                if dep not in self._tasks:
                    raise ValueError(
                        f"Task '{task.name}' depends on '{dep}', but no task "
                        f"with that name exists in this pipeline. "
                        f"Available tasks: {list(self._tasks.keys())}"
                    )

        # DFS cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {name: WHITE for name in self._tasks}

        def dfs(name: str) -> None:
            color[name] = GRAY
            for dep in self._tasks[name].depends_on:
                if color[dep] == GRAY:
                    raise ValueError(
                        f"Cycle detected in pipeline dependency graph: "
                        f"'{name}' -> '{dep}' creates a cycle. "
                        f"DAG dependencies must be acyclic."
                    )
                if color[dep] == WHITE:
                    dfs(dep)
            color[name] = BLACK

        for name in self._tasks:
            if color[name] == WHITE:
                dfs(name)
