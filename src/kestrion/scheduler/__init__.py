"""
Kestrion Scheduler — multi-agent orchestration with DAG-based pipelines.

Public API::

    from kestrion.scheduler import Pipeline, AgentTask, RateLimiter, RateLimiterConfig, WorkerPool

Example::

    pipeline = Pipeline(
        tasks=[
            AgentTask("researcher_a", agent=agent_a, prompt="Research X"),
            AgentTask("researcher_b", agent=agent_b, prompt="Research Y"),
            AgentTask(
                "synthesizer",
                agent=synth_agent,
                prompt="Combine research findings",
                depends_on=["researcher_a", "researcher_b"],
            ),
        ],
        max_workers=3,
    )

    results = await pipeline.run()
"""

from .pipeline import AgentTask, Pipeline, TaskResult, TaskStatus
from .rate_limiter import RateLimiter, RateLimiterConfig
from .worker_pool import WorkerPool

__all__ = [
    "AgentTask",
    "Pipeline",
    "TaskResult",
    "TaskStatus",
    "RateLimiter",
    "RateLimiterConfig",
    "WorkerPool",
]
