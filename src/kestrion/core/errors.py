"""
Core Exception hierarchy for Kestrion.

All framework-specific exceptions derive from KestrionError, making it
straightforward for calling code to catch framework-level errors broadly
or specifically.
"""

from __future__ import annotations

from typing import Any


class KestrionError(Exception):
    """Base exception for all framework-level errors in Kestrion."""


class CheckpointNotFoundError(KestrionError):
    """Raised when a requested checkpoint is not found in the store."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        super().__init__(f"No checkpoint found for run_id={run_id!r}")


class InvalidRunStatusError(KestrionError):
    """Raised when performing an operation on a run with an invalid/unexpected status."""

    def __init__(self, run_id: str, status: str, expected_statuses: list[str] | None = None):
        self.run_id = run_id
        self.status = status
        self.expected_statuses = expected_statuses
        expected_msg = f" (expected one of: {expected_statuses})" if expected_statuses else ""
        super().__init__(
            f"Run {run_id!r} has invalid status {status!r}{expected_msg}"
        )


class InvalidStoreURLError(KestrionError, ValueError):
    """Raised when the checkpoint store URL scheme is unsupported or malformed."""


class InvalidToolApprovalError(KestrionError):
    """Raised when trying to approve a tool call that is not currently pending or mismatching."""


class RunExpiredError(KestrionError):
    """
    Raised by resume(on_expired="raise") when a run's pending approval
    deadline has passed without all required roles approving.
    """

    def __init__(self, run_id: str, tool_name: str, expired_at: str):
        self.run_id = run_id
        self.tool_name = tool_name
        self.expired_at = expired_at
        super().__init__(
            f"Run {run_id}'s pending approval for tool {tool_name!r} expired at {expired_at}"
        )


class ApprovalRequired(KestrionError):
    """
    Raised when a node wants to call a tool whose required-approval
    roles are not all satisfied yet.
    """

    def __init__(self, tool_name: str, kwargs: dict[str, Any], missing_roles: list[str]):
        self.tool_name = tool_name
        self.kwargs = kwargs
        self.missing_roles = missing_roles
        super().__init__(
            f"Approval required for tool {tool_name!r} (missing roles: {missing_roles})"
        )


class HandoffCompleted(KestrionError):
    """
    Raised to signal that the conversation has been successfully
    transferred to another agent. This is a control-flow signal, not a failure.
    """

    def __init__(self, target_run_id: str, target_status: str, target_output: str | None):
        self.target_run_id = target_run_id
        self.target_status = target_status
        self.target_output = target_output
        super().__init__(f"Handoff completed to run {target_run_id!r}")


class InputRequired(KestrionError):
    """
    Raised when a node wants to call a tool that requires human input.
    """

    def __init__(self, tool_name: str, kwargs: dict[str, Any], question: str):
        self.tool_name = tool_name
        self.kwargs = kwargs
        self.question = question
        super().__init__(f"Input required for tool {tool_name!r}: {question}")


class InvalidToolInputError(KestrionError):
    """Raised when trying to provide input for a tool that is not currently waiting for it."""


class LLMConnectionError(KestrionError):
    """Raised when the engine cannot communicate with the configured LLM provider."""


class StructuredOutputError(KestrionError):
    """Raised when the model's final response cannot be parsed or validated against the requested output_schema."""

    def __init__(self, message: str, raw_output: str | None = None):
        self.raw_output = raw_output
        super().__init__(f"{message} (raw_output: {raw_output!r})")

