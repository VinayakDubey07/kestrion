# Security & Compliance Architecture

Kestrion is built from the ground up for enterprise environments where agents must operate securely, traceably, and under strict governance.

This document details the three pillars of Kestrion's security posture: **Secret Management**, **Immutable Audit Logs**, and **Role-Based Approval Gates**.

## 1. Secret Management

AI agents frequently need credentials to access databases, clouds, and external APIs. Storing these credentials in an agent's working memory or the event store is a critical security vulnerability.

In Kestrion, **secrets are never stored in the `AgentState`**. Instead, they are injected exactly at the moment of execution via the `SecretProvider` interface.

### The `SecretProvider` Protocol

When an agent is instantiated, you provide a `SecretProvider`:

```python
from kestrion.core.secrets import EnvVarSecretProvider

engine = Engine(
    ...
    secrets=EnvVarSecretProvider() # Or your custom AWS Secrets Manager / Vault provider
)
```

Tools never receive secrets directly in their arguments from the LLM. Instead, the engine injects a `_secrets` reference into the tool kwargs when it invokes the Python function:

```python
from kestrion.agent.decorators import tool

@tool
def query_production_db(_secrets=None) -> str:
    # Fetch the secret dynamically at execution time
    db_password = _secrets.get("PROD_DB_PASSWORD")
    ...
```

Because `_secrets` is injected by the engine and stripped out before any events are recorded, the database password is never serialized to the Checkpoint Store.

## 2. Immutable Audit Log (Event Sourcing)

Most agent frameworks bolt observability on top of their execution loop. Kestrion uses **Event Sourcing**, meaning the event log *is* the execution loop.

Every single action is durably recorded as an immutable `Event`:
- `RUN_STARTED`
- `LLM_CALL_STARTED` & `LLM_CALL_COMPLETED` (including exact token counts and costs)
- `TOOL_CALL_STARTED` (including the exact arguments provided by the model)
- `TOOL_CALL_COMPLETED` (including the exact output returned by the tool)
- `HUMAN_INTERVENTION` (recording exactly who approved an action and when)

Because this log is immutable and append-only, it serves as a cryptographically verifiable audit trail of exactly what an agent did, what data it saw, and what decisions it made. If a rogue tool call drops a table, you have the complete lineage of how the agent arrived at that decision.

## 3. Role-Based Approval Gates (RBAC)

Agent autonomy is a spectrum. Some tools are safe to execute automatically (e.g., `check_weather`, `read_logs`). Other tools carry significant risk and require human oversight (e.g., `apply_kubernetes_manifest`, `execute_sql`).

Kestrion provides a centralized, engine-enforced approval gate.

### Multi-Step Approval Chains

A tool can be marked as requiring approval from specific roles:

```python
@tool(requires_approval=["engineer", "security_admin"])
def deploy_to_prod() -> dict:
    ...
```

When the LLM attempts to call `deploy_to_prod`, the Engine halts the run and transitions it to `WAITING_ON_HUMAN`. 

The tool **cannot** be executed until the `Engine.record_approval()` method has been called for all required roles. This is enforced centrally by the `Engine` — it is impossible for a bug in the LLM loop or the tool's code to bypass this check.

### Time-Boxed Approvals

Approvals can be time-boxed to prevent stale requests from lingering in the queue:

```python
@tool(requires_approval=True, approval_timeout_seconds=3600.0)
def restart_service() -> dict:
    ...
```

If the required approvals are not recorded within one hour, the engine automatically transitions the run to `EXPIRED`.

## Telemetry & OpenTelemetry (OTel)

For real-time security monitoring, Kestrion supports exporting its immutable event stream directly into OpenTelemetry (OTel) compatible backends (Datadog, Splunk, Jaeger). This allows enterprise SOC teams to monitor agent behavior and anomalies using their existing observability stack.

By instantiating an `OpenTelemetryProvider` and passing it to the `Agent` or the `Engine` at startup, execution metrics and tracing metadata are automatically dispatched to the configured collector:

```python
from kestrion.telemetry.otel import OpenTelemetryProvider

# Connects to your standard OTel collector or vendor agent (e.g. Datadog agent)
otel_telemetry = OpenTelemetryProvider(service_name="kestrion-prod-agent")

agent = Agent(
    provider=provider,
    tools=tools,
    store="postgres://...",
    telemetry=otel_telemetry,
)
```

Every run event, tool execution, and inference query is traced under nested span contexts (`kestrion.run.*`, `kestrion.llm.*`, `kestrion.tool.*`), enabling deep observability and real-time security auditing.

## 4. Data Loss Prevention (PII Redaction)

Even when running on secure infrastructure, sending data to external LLM providers (like OpenAI or Anthropic) carries the risk of data leakage. 

Kestrion includes a built-in `PIIRedactionMiddleware` that acts as a Data Loss Prevention (DLP) layer directly in the provider pipeline.

### Two-Way Vault System

The middleware intercepts traffic *before* it leaves your network:

1. **Redaction**: It scans the agent's prompts and context history for sensitive data (SSNs, Credit Cards, Emails, etc.) and replaces them with secure tokens (e.g., `[REDACTED_SSN_0]`).
2. **Restoration**: If the LLM attempts to use that data in a tool call (e.g., `submit_background_check(ssn="[REDACTED_SSN_0]")`), the middleware intercepts the response and seamlessly swaps the token back to the real SSN before it reaches your local Python tool.

The external AI provider **never** sees the sensitive data, but your local agent and its tools continue to function normally.

```python
from kestrion.agent.agent import Agent
from kestrion.llm.openai_provider import OpenAIProvider
from kestrion.llm.middleware import PIIRedactionMiddleware

base_provider = OpenAIProvider(model="gpt-4o")
# Wrap the provider with the DLP middleware
secure_provider = PIIRedactionMiddleware(provider=base_provider)

agent = Agent(
    provider=secure_provider,
    tools=[...]
)
```
