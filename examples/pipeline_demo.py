"""
Pipeline Demo — Multi-Agent Research Team using Kestrion Scheduler

This demo shows the DAG-based orchestrator in action with three Ollama agents:

    researcher_a  ──────┬──── synthesizer
    researcher_b  ──────┘

researcher_a and researcher_b run CONCURRENTLY (no dependencies).
synthesizer waits for both to finish, then combines their outputs.

Run with:
    PYTHONPATH=src .venv/bin/python examples/pipeline_demo.py

Requires Ollama running locally:
    ollama serve
    ollama pull llama3.2
"""

from __future__ import annotations

import asyncio

from kestrion.agent.agent import Agent
from kestrion.llm.ollama_provider import OllamaProvider
from kestrion.scheduler import AgentTask, Pipeline


STORE = "sqlite:///pipeline_demo.db"
MODEL = "llama3.2"

TOPIC_A = "the history and main use cases of Python programming language"
TOPIC_B = "the history and main use cases of Rust programming language"
SYNTHESIS_PROMPT = (
    "You have been given research on two programming languages. "
    "Write a 3-paragraph comparison highlighting: "
    "(1) key similarities, (2) key differences, (3) when to choose each."
)


def make_researcher(topic: str) -> Agent:
    return Agent(
        provider=OllamaProvider(model=MODEL),
        store=STORE,
        system_prompt=(
            "You are a concise technical researcher. "
            "Given a topic, write a clear 2-paragraph summary covering "
            "its history and the 3 most important use cases. "
            "Be factual and brief."
        ),
    )


def make_synthesizer() -> Agent:
    return Agent(
        provider=OllamaProvider(model=MODEL),
        store=STORE,
        system_prompt=(
            "You are a senior technical writer. "
            "Synthesize research into clear, well-structured comparisons."
        ),
    )


async def main() -> None:
    print("=" * 60)
    print("Kestrion Pipeline Demo — Research Team")
    print("=" * 60)
    print(f"  Model: {MODEL}")
    print(f"  Store: {STORE}")
    print()
    print("DAG topology:")
    print("  researcher_a (Python) ──┬── synthesizer")
    print("  researcher_b (Rust)   ──┘")
    print()

    researcher_a = make_researcher(TOPIC_A)
    researcher_b = make_researcher(TOPIC_B)
    synthesizer  = make_synthesizer()

    pipeline = Pipeline(
        tasks=[
            AgentTask(
                name="researcher_a",
                agent=researcher_a,
                prompt=f"Research this topic: {TOPIC_A}",
                estimated_tokens=300,
            ),
            AgentTask(
                name="researcher_b",
                agent=researcher_b,
                prompt=f"Research this topic: {TOPIC_B}",
                estimated_tokens=300,
            ),
            AgentTask(
                name="synthesizer",
                agent=synthesizer,
                prompt=SYNTHESIS_PROMPT,
                depends_on=["researcher_a", "researcher_b"],
                estimated_tokens=500,
            ),
        ],
        max_workers=3,   # researcher_a and researcher_b run concurrently
        # No rate_limiter_config — Ollama is local/unlimited
    )

    print("Starting pipeline...\n")
    results = await pipeline.run()
    print()

    # Print each result
    for task_name in ["researcher_a", "researcher_b", "synthesizer"]:
        r = results[task_name]
        duration = f"({r.duration_seconds:.1f}s)" if r.duration_seconds else ""
        print(f"─── {task_name.upper()} {duration} [{r.status.value}]")
        if r.run_result and r.run_result.output:
            print(r.run_result.output)
        elif r.error:
            print(f"[Error: {r.error}]")
        print()

    print("=" * 60)
    print(pipeline.status_summary(results))
    print("=" * 60)
    print("\nAll run IDs saved to pipeline_demo.db — inspect them in the dashboard!")
    print("Run: kestrion dashboard --store pipeline_demo.db --port 8080")


if __name__ == "__main__":
    asyncio.run(main())
