# Sub-Agents vs. Handoff

When building multi-agent systems, agents need a way to collaborate. Kestrion provides two distinct patterns for agent-to-agent collaboration: **Delegation (Sub-Agents)** and **Transfer (Handoff)**. Both patterns are designed around durable execution and correct approval gating.

---

## 1. Delegation (Sub-Agents)

In the delegation pattern, a parent agent delegates a sub-task to a specialist child agent and waits for an answer. The parent agent remains in control of the overall conversation.

### How it works under the hood
You wrap a specialist agent as a tool that the parent agent can call:
```python
specialist = Agent(provider=..., tools=[...], store=shared_store_url)
parent = Agent(
    provider=...,
    tools=[specialist.as_tool("ask_specialist", "Delegate a query to the specialist")],
    store=shared_store_url, # MUST share the same store
)
```

When the parent agent calls the `ask_specialist` tool:
1. The engine starts a new, independent run for the sub-agent using its own `run_id` in the shared store.
2. The sub-agent runs to completion (or pauses).
3. If the sub-agent completes normally, its final output is returned to the parent as a standard tool result.

### Gated approvals in sub-agents
If the sub-agent needs human approval for a gated tool:
* The sub-agent's run pauses with `WAITING_ON_HUMAN`.
* The `SubAgentTool` propagates this pause to the parent agent by raising `ApprovalRequired` with a synthetic role name indicating the child's run: `missing_roles=[f"sub_agent:{child_run_id}"]`.
* The parent run also pauses.
* To resume, the host application must approve the sub-agent's gated tool, resume the sub-agent's run, and then resume the parent's run.

---

## 2. Transfer (Handoff)

In the transfer pattern, the calling agent hands over the entire conversation to another agent and exits. The target agent takes over completely, and the original agent's run is terminated.

### How it works under the hood
You wrap the target agent as a handoff target tool:
```python
billing_agent = Agent(provider=..., tools=[...], store=shared_store_url)
router = Agent(
    provider=...,
    tools=[billing_agent.as_handoff_target("transfer_to_billing", "Transfer to billing specialist")],
    store=shared_store_url,
)
```

When the router decides to call `transfer_to_billing`:
1. The entire conversation message history is captured.
2. Under the hood, calling `HandoffTool` raises a `HandoffCompleted` exception.
3. The engine catches `HandoffCompleted` and terminates the caller's run immediately with `RunStatus.COMPLETED`.
4. The target agent starts a new run with `run_with_history(messages)` using a new `run_id`.
5. The calling run links to the target run by storing the target's run ID in its scratch memory under `_handed_off_to`.

---

## Key Differences

| Dimension | Delegation (Sub-Agents) | Transfer (Handoff) |
| :--- | :--- | :--- |
| **Control** | Parent agent retains control. | Target agent takes over completely. |
| **Caller Run Lifecycle** | Pauses/resumes along with the sub-agent; completes when final answer is found. | Terminates immediately with `RunStatus.COMPLETED` as soon as the handoff occurs. |
| **Conversation History** | Only the prompt is passed to the sub-agent; parent only receives the sub-agent's final answer. | The entire conversation history is transferred to the target agent. |
| **Run IDs** | Independent run IDs; parent tracks the sub-agent's run ID via `missing_roles` if paused. | Independent run IDs; caller links to target run ID in `scratch["_handed_off_to"]`. |
| **Use Case** | Orchestrating sub-tasks, querying databases, running isolated sub-routines. | Routing user to specialized departments (e.g. general router agent -> billing agent). |

## Safety and Scoping Design Decision

Both sub-agents and handoffs generate new, independent run IDs rather than continuing the same run ID. This is a deliberate security decision to ensure that **approval scopes are kept correct**. 

If the same run ID were reused across agents, a target agent could inherit `scratch["_approved_tools"]` permissions recorded for a different agent's tools. By keeping run IDs separate, tool permissions are cleanly isolated.
