# Security Policy

## Supported Versions

Currently, Kestrion is in active development. Security updates are applied to the `main` branch and the latest PyPI release. 

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3.0 | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in Kestrion, please report it privately. **Do not open a public GitHub issue.**

Please send an email to **vinayak@kestrion.in**. We will endeavor to respond to your report within 48 hours and work with you to verify and resolve the issue.

## Enterprise Security Architecture

Kestrion is designed for enterprise environments where security, auditability, and role-based access control are prerequisites. 

- **Immutable Audit Logs:** All agent actions, including LLM requests, prompts, and tool executions, are written to an immutable event log.
- **RBAC Approval Gates:** Mutating tools (e.g., executing SQL or applying Kubernetes manifests) can be gated behind multi-role approval chains.
- **Secret Management:** API keys are never stored in agent state. They are injected at runtime via the `SecretProvider` interface.

For a deep dive into Kestrion's security architecture, please see the [Security Documentation](docs/security.md).
