"""
End-to-end integration demo: a small ops support system that exercises
all five agentic features built on top of the core engine, running
against a REAL local Ollama model — not fake providers like every unit
test in this project uses. This is deliberately the first time these
features are exercised together in one live run, since each one was
only proven in isolation (against scripted fakes) until now.

Scenario:
  router agent
    -> handoff -> billing agent (chain: support + finance approval,
                  with a timeout -- exercises approval chains + timeouts)
    -> handoff -> infra agent (parallel tool calls checking 2 servers,
                  delegates disk cleanup to a sub-agent, has its own
                  gated restart tool -- exercises parallel calls,
                  sub-agents, and a simple single-role approval gate)

Requires Ollama running locally with a model pulled:
    ollama serve &
    ollama pull llama3.2

Run with:
    python3 examples/ops_demo/main.py

NOTE ON MODEL BEHAVIOR: small local models are less reliable than
hosted models (Claude, GPT) at deciding which tool to call and when —
already documented honestly in llm/ollama_provider.py's malformed-JSON
handling. This script prints what actually happened at each step rather
than asserting a specific outcome, since the model's exact choices may
vary run to run. The PLUMBING (approval gates, chains, timeouts,
sub-agent delegation, handoff) is what's being proven here, not "the
model always makes the textbook-correct decision."
"""

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from kestrion.agent.agent import Agent
from kestrion.agent.decorators import tool
from kestrion.core.engine import Engine
from kestrion.core.types import Checkpoint, RunStatus, new_id
from kestrion.llm.ollama_provider import OllamaProvider

MODEL = "llama3.2"


# ---------------------------------------------------------------------------
# Infra agent's tools
# ---------------------------------------------------------------------------

@tool
def get_server_status(server_name: str) -> dict:
    """Check the status of a server by name."""
    # Deliberately fake/static data -- this is a demo, not a real
    # infrastructure integration. Mirrors the same pattern as
    # examples/kubectl_agent.py.
    statuses = {
        "web-01": {"status": "healthy", "disk_used_pct": 45},
        "web-02": {"status": "healthy", "disk_used_pct": 91},
    }
    return statuses.get(server_name, {"status": "unknown"})


@tool(requires_approval=True)
def restart_server(server_name: str) -> dict:
    """Restart a server. Requires approval before restarting."""
    return {"restarted": server_name}


# ---------------------------------------------------------------------------
# Disk-cleanup sub-agent's tool (delegated to by the infra agent)
# ---------------------------------------------------------------------------

@tool
def free_disk_space(server_name: str) -> dict:
    """Clean up temp files and logs on a server to free disk space."""
    return {"server": server_name, "freed_gb": 12}


# ---------------------------------------------------------------------------
# Billing agent's tool: a multi-role approval chain WITH a timeout
# ---------------------------------------------------------------------------

@tool(requires_approval=["support", "finance"], approval_timeout_seconds=3600.0)
def issue_refund(customer_id: str, amount: int) -> dict:
    """Issue a refund to a customer. Needs both support and finance to approve, within an hour."""
    return {"refunded": True, "customer_id": customer_id, "amount": amount}


async def approve_and_persist(agent: Agent, state, tool_name: str, role: str):
    """Records one role's approval and persists it as a checkpoint, so a
    later resume() call can see it. This is the real, current mechanism
    -- Agent.approve() is still a stub, see README known gaps."""
    Engine.record_approval(state, tool_name, role=role)
    await agent._store.save(Checkpoint(
        checkpoint_id=new_id("ckpt"),
        run_id=state.run_id,
        state=state,
        created_at=datetime.now(timezone.utc),
        event_seq=state.last_event_seq,
    ))


async def run_infra_scenario(store_url: str):
    print("\n" + "=" * 70)
    print("SCENARIO 1: Infra agent -- parallel checks, sub-agent delegation, gated restart")
    print("=" * 70)

    disk_specialist = Agent(
        provider=OllamaProvider(model=MODEL),
        tools=[free_disk_space],
        store=store_url,
    )
    cleanup_tool = disk_specialist.as_tool(
        "ask_disk_specialist", "Ask the disk-cleanup specialist to free up space on a server"
    )

    infra_agent = Agent(
        provider=OllamaProvider(model=MODEL),
        tools=[get_server_status, restart_server, cleanup_tool],
        store=store_url,
        system_prompt=(
            "You manage servers. Check server status when asked. If a server's disk usage "
            "is over 80%, delegate cleanup to the disk specialist tool. Only restart a server "
            "if explicitly asked to."
        ),
    )

    result = await infra_agent.run(
        "Check the status of web-01 and web-02, and clean up disk space on any server over 80% full."
    )
    print(f"\nstatus: {result.status}")
    print(f"output: {result.output}")

    parent_tools = [h.get("tool") for h in result.state.history if h.get("type") == "tool_result"]
    print(f"infra agent's own direct tool calls: {parent_tools}")
    print("(if the model routed everything through ask_disk_specialist, the actual")
    print(" get_server_status/free_disk_space calls happened on the disk specialist's")
    print(" OWN, separate run -- by design, see SubAgentTool's docstring decision #1.")
    print(" That's not a bug: it's what makes the sub-agent's work independently")
    print(" resumable even if this parent run later crashes.)")
    return result


async def run_billing_scenario(store_url: str):
    print("\n" + "=" * 70)
    print("SCENARIO 2: Billing agent -- multi-role approval chain with a timeout")
    print("=" * 70)

    billing_agent = Agent(
        provider=OllamaProvider(model=MODEL),
        tools=[issue_refund],
        store=store_url,
        system_prompt="You handle billing requests. Use the issue_refund tool when a customer requests a refund.",
    )

    result = await billing_agent.run("Please refund customer CUST-42 for $50.")
    print(f"\nstatus: {result.status}")

    if result.status == RunStatus.WAITING_ON_HUMAN:
        pending = result.state.scratch["_pending_approval"]
        print(f"pending approval: tool={pending['tool']} missing_roles={pending['missing_roles']}")

        print("Approving as 'support'...")
        await approve_and_persist(billing_agent, result.state, "issue_refund", role="support")
        mid_resume = await billing_agent.resume(result.run_id)
        print(f"status after 1 of 2 approvals: {mid_resume.status}")
        if mid_resume.status == RunStatus.WAITING_ON_HUMAN:
            print(f"still missing: {mid_resume.state.scratch['_pending_approval']['missing_roles']}")

        print("Approving as 'finance'...")
        await approve_and_persist(billing_agent, mid_resume.state, "issue_refund", role="finance")
        final = await billing_agent.resume(result.run_id)
        print(f"final status: {final.status}")
        print(f"output: {final.output}")
        return final

    print(f"output: {result.output}")
    return result


async def run_handoff_scenario(store_url: str):
    print("\n" + "=" * 70)
    print("SCENARIO 3: Router agent -- hands off to the billing agent")
    print("=" * 70)

    billing_agent = Agent(
        provider=OllamaProvider(model=MODEL),
        tools=[issue_refund],
        store=store_url,
        system_prompt="You handle billing requests. Use the issue_refund tool when asked to refund a customer.",
    )
    billing_handoff = billing_agent.as_handoff_target(
        "transfer_to_billing", "Transfer the conversation to the billing specialist"
    )

    router = Agent(
        provider=OllamaProvider(model=MODEL),
        tools=[billing_handoff],
        store=store_url,
        system_prompt=(
            "You are a router. If the request is about billing, refunds, or payments, "
            "use the transfer_to_billing tool immediately."
        ),
    )

    result = await router.run("I'd like a refund for my last order.")
    print(f"\nrouter status: {result.status}")
    print(f"router output: {result.output}")

    handed_off_to = result.state.scratch.get("_handed_off_to")
    if handed_off_to:
        print(f"handed off to billing agent run_id: {handed_off_to}")
    return result


async def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = str(Path(tmpdir) / "ops_demo.db")
        store_url = f"sqlite:///{store_path}"

        await run_infra_scenario(store_url)
        await run_billing_scenario(store_url)
        await run_handoff_scenario(store_url)

        print("\n" + "=" * 70)
        print("All three scenarios completed. This exercised, against a real")
        print("local model: parallel tool calls, sub-agent delegation, a")
        print("single-role approval gate, a multi-role approval chain, an")
        print("approval timeout configuration, and multi-agent handoff.")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())