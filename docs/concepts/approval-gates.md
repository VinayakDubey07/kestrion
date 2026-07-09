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

## Sub-agent approval propagation

When using the sub-agent delegation pattern, a sub-agent's run might pause waiting on a gated tool. The engine propagates this pause to the parent run as a pending approval for a synthetic role named `sub_agent:<child_run_id>`. For a detailed discussion on how delegation and approvals work across multiple agents, see [Sub-Agents vs. Handoff](sub-agents-vs-handoff.md).

## Durable Approval APIs (`Agent.approve()` & `Engine.approve_pending_tool()`)

To make approvals extremely simple, Kestrion provides built-in methods at both the high-level `Agent` layer and the low-level `Engine` layer.

### 1. High-Level: `Agent.approve()`
`Agent.approve()` is a convenience method that takes care of the entire loading, validating, recording, checkpointing, and optional resuming flow in a single call.

```python
async def approve(
    self,
    run_id: str,
    tool: str | None = None,
    role: str = "__any__",
    and_resume: bool = True,
) -> RunResult | None:
```

* **Default behavior**: Calling `await agent.approve(run_id)` automatically resolves the currently pending tool from the checkpoint's state, records a generic approval, writes a durable `HUMAN_INTERVENTION` event (with an `approval_granted` payload), saves a new checkpoint, and calls `resume()` immediately.
* **Partial approvals**: When validating a multi-role chain, call `agent.approve(run_id, role="role_name", and_resume=False)`. This persists the approval fact safely to the store but keeps the run paused so other roles can approve later.

### 2. Low-Level: `Engine.approve_pending_tool()`
For raw graph/engine users, the `Engine` exposes `approve_pending_tool()`:

```python
async def approve_pending_tool(
    self,
    run_id: str,
    tool: str | None = None,
    role: str = "__any__",
) -> AgentState:
```

Unlike `Agent.approve()`, this method records and checkpoints the approval but does **not** automatically trigger resumption. The caller is responsible for calling `engine.resume(run_id)` separately:

```python
# Record the approval (persists a new checkpoint):
await engine.approve_pending_tool(run_id, tool="deploy_to_prod", role="engineer")

# Continue the run when ready:
result = await engine.resume(run_id)
```

## Input gates (human-in-the-loop text input)

Approval gates are binary: "may this tool run?" Sometimes the agent needs a **specific piece of
information** from the human — a 2FA code, a preferred name, a clarification. This is handled by a
separate, parallel mechanism that uses the same engine pause infrastructure but stores different
state.

### The built-in `ask_human` tool

```python
from kestrion.agent.tools import ask_human

agent = Agent(provider=..., tools=[ask_human], store=...)
result = await agent.run("Write a poem about my favorite color")
# result.status == RunStatus.WAITING_ON_HUMAN
# result.state.scratch["_pending_input"]["question"] == "What is your favorite color?"
```

When triggered, `ask_human` raises `InputRequired` (not `ApprovalRequired`). The engine catches
this in `_drive`, sets `state.status = WAITING_ON_HUMAN`, stores the question in
`scratch["_pending_input"]`, checkpoints, and returns — identical mechanics to an approval pause.

### Providing the answer

```python
# At the Agent level (recommended):
final = await agent.provide_input(run_id, text="Blue")

# Or at the Engine level:
await engine.provide_input(run_id, "Blue", tool="ask_human")
state = await engine.resume(run_id)
```

`provide_input` stores the human's text in `scratch["_human_inputs"][tool_name]`, clears
`_pending_input`, transitions back to `RUNNING`, and checkpoints. On resume, the `ask_human` tool
sees the stored answer and returns it as its output instead of raising again.

### Building your own input-requesting tool

You don't have to use `ask_human`. Any tool can raise `InputRequired` directly:

```python
from kestrion.core.errors import InputRequired

@tool
def get_2fa_code(service: str) -> str:
    """Prompt the user for a 2FA code."""
    raise InputRequired(tool_name="get_2fa_code", question=f"Enter the 2FA code for {service}")
```

The engine handles `InputRequired` identically regardless of which tool raises it.

### Approval vs. Input — when to use which

| Mechanism | Use when... | State key | Exception |
|---|---|---|---|
| Approval gate | You need permission to *run* a tool | `_pending_approval` | `ApprovalRequired` |
| Input gate | You need *data* from the human | `_pending_input` | `InputRequired` |

Attempting to `approve()` a run that's waiting for input (or `provide_input()` to a run waiting for
approval) raises a clear error — they are distinct mechanisms that share the same engine pause
infrastructure.

## Related

- [Event Sourcing](event-sourcing.md) — why a pause is just "stop folding, record one event, return"
- [Checkpointing](checkpointing.md) — how a paused run survives a process boundary
- [Sub-Agents vs. Handoff](sub-agents-vs-handoff.md) — delegation vs transfer patterns in multi-agent runs