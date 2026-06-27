# Approval Gates

A tool marked as requiring approval can't be invoked without it — enforced by the engine itself,
not by convention each tool author has to remember. This document covers how that gate works, and
the three extensions built on top of the base mechanism: multi-step chains, timeouts, and the
safety guarantee around parallel tool calls.

## The base mechanism

Every tool call, from anywhere, goes through one method: `Engine.call_tool`. There is no other way
to invoke a tool's `.call()` — which is what makes it impossible for a node author to accidentally
bypass the gate.

```python
def check_approval(self, state, tool_name, kwargs) -> None:
    tool = self.tools[tool_name]
    required_roles = tool.spec.required_roles()
    if not required_roles:
        return
    # ... check what's recorded in state.scratch["_approved_tools"] ...
    if missing:
        raise ApprovalRequired(tool_name, kwargs, missing_roles=missing)
```

`ApprovalRequired` is raised *before* anything else happens — before `TOOL_CALL_STARTED` is even
emitted. This matters: it means there is never a tool call that's "partially" executed because it
needed approval. Either the gate is satisfied and the call proceeds normally, or nothing happens at
all.

When `ApprovalRequired` escapes a node's `run()` method, `Engine._drive` catches it — not the node,
not your code — and:

1. Sets `state.status = RunStatus.WAITING_ON_HUMAN`
2. Records which tool and which roles are still missing in `scratch["_pending_approval"]`
3. Checkpoints
4. Returns control. No thread is blocked.

## `requires_approval` accepts four shapes

```python
ToolSpec(..., requires_approval=False)              # no approval needed (default)
ToolSpec(..., requires_approval=True)                # any single approval
ToolSpec(..., requires_approval="manager")           # approval from this specific role
ToolSpec(..., requires_approval=["engineer", "manager"])  # approval from ALL listed roles
```

The last form is a multi-step approval chain. `ToolSpec.required_roles()` normalizes all four into
a list — `True` becomes a single sentinel role (`"__any__"`), so the original boolean-style approval
is just a one-item, role-agnostic chain underneath.

## Recording an approval correctly

```python
Engine.record_approval(state, "deploy_to_prod", role="engineer")
Engine.record_approval(state, "deploy_to_prod", role="manager")
```

Use this, not direct dict manipulation. The reason: `record_approval` *adds* a role to whatever's
already recorded for that tool. Writing `scratch["_approved_tools"] = {"deploy_to_prod": True}`
directly would silently **destroy** a partially-satisfied chain — if "engineer" already approved
and you overwrite the whole dict to record "manager," the engineer's approval vanishes. This was a
real, deliberate design choice made specifically to prevent that footgun, not an accident of the
API surface.

A chain only unblocks once every required role appears in the recorded set. Approving with the
wrong role name — `record_approval(state, "deploy_to_prod", role="security_lead")` when the tool
needs `"manager"` — does not satisfy the requirement; `missing_roles` will still list `"manager"`.

## Timeouts

A tool can carry a deadline:

```python
ToolSpec(..., requires_approval=True, approval_timeout_seconds=3600.0)
```

The deadline is anchored to the **first** time approval was requested for that tool, not reset
every time someone resumes without approving. If a chain is partially satisfied (one role approved,
one still missing) and the run is resumed again before any further approval, the original deadline
still applies — it doesn't get pushed back.

If the deadline passes before all required roles approve, `resume()` transitions the run to a new
terminal state, `RunStatus.EXPIRED`, instead of indefinitely re-raising `ApprovalRequired`:

```python
result = await engine.resume(run_id)  # default: on_expired="status"
if result.status == RunStatus.EXPIRED:
    ...

# Or, for callers that want a hard failure instead of a status to check:
await engine.resume(run_id, on_expired="raise")  # raises RunExpiredError
```

Tools with no `approval_timeout_seconds` set (the default, `None`) behave exactly as they did
before this feature existed — they can wait indefinitely.

## The safety guarantee for parallel tool calls

`Agent`'s loop can dispatch several tool calls from one LLM turn concurrently via
`asyncio.gather`, rather than one at a time (see [the architecture
document](../architecture.md) for the full design). This raises an obvious question: what happens
if a batch contains both a gated tool and safe tools?

The guarantee: **a batch either fully runs or cleanly pauses with nothing partially executed.**
This is implemented as a two-phase process — every gated call in the batch is checked via
`Engine.check_approval` *before* any call, gated or not, is dispatched. If any one of them is
missing approval, `ApprovalRequired` is raised immediately, before `asyncio.gather` is ever called.
None of the batch's tools — not even the safe ones sitting alongside the gated one — execute.

This is directly tested:
`tests/unit/test_parallel_tool_calls.py::test_gated_call_in_a_batch_blocks_everything_before_any_call_executes`
confirms that neither tool in a mixed batch runs when one of them is gated and unapproved.

## What's still a known gap

`Agent.approve()` — a single-call convenience method for recording an approval — is currently a
stub that raises `NotImplementedError` with instructions for doing it manually (exactly the
`Engine.record_approval` + checkpoint-save pattern shown above and in
[Getting Started](../getting-started.md)). A real implementation needs a durable approval-
persistence layer of its own; see the project README's "Known gaps" for current status.

## Related

- [Event Sourcing](event-sourcing.md) — why a pause is just "stop folding, record one event, return"
- [Checkpointing](checkpointing.md) — how a paused run survives a process boundary