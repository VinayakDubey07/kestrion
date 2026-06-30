"""
@tool turns a plain Python function into something satisfying the
Tool/ToolSpec contract from core/types.py, by introspecting the
function signature into a JSON schema. This is the highest-risk piece
of Phase 2, flagged as such in the build plan — type hints have a lot
of edge cases (Optional, defaults, list[str], nested types). What's
implemented here covers the common cases; anything genuinely exotic
(Pydantic models, deeply nested generics) is a known gap, not silently
mishandled — see the NotImplementedError below.
"""

from __future__ import annotations

import functools
import inspect
import time
from typing import Any, Callable, Union, get_args, get_origin

from kestrion.core.types import Tool, ToolResult, ToolSpec

_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """
    Best-effort conversion of a type annotation to a JSON schema
    fragment. Handles: plain types, Optional[X] (-> X's schema, since
    JSON schema has no first-class None), list[X], and falls back to
    "string" for anything unrecognized rather than crashing — an LLM
    can usually work with an under-specified schema, but a hard crash
    at decoration time would break the whole tool registration.
    """
    origin = get_origin(annotation)

    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _python_type_to_json_schema(args[0])
        return {"type": "string"}  # multi-type unions: punt to string, documented gap

    if origin in (list, tuple):
        item_args = get_args(annotation)
        item_schema = _python_type_to_json_schema(item_args[0]) if item_args else {"type": "string"}
        return {"type": "array", "items": item_schema}

    if origin is dict:
        return {"type": "object"}

    if annotation in _TYPE_MAP:
        return {"type": _TYPE_MAP[annotation]}

    if annotation is inspect.Parameter.empty:
        # No type hint at all — still allow the tool to be registered,
        # but the schema is maximally permissive rather than guessed.
        return {"type": "string"}

    # Unrecognized annotation (e.g. a Pydantic model, a custom class).
    # Rather than silently producing a wrong/misleading schema, this is
    # the explicit gap mentioned above.
    raise NotImplementedError(
        f"@tool cannot auto-generate a JSON schema for annotation {annotation!r}. "
        f"Supported: str, int, float, bool, list[T], dict, and Optional[T] of these."
    )


def _build_parameters_schema(func: Callable) -> dict[str, Any]:
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        properties[name] = _python_type_to_json_schema(param.annotation)
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            properties[name]["default"] = param.default

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


class _FunctionTool(Tool):
    """Wraps a plain function (sync or async) to satisfy the Tool interface."""

    def __init__(self, func: Callable, spec: ToolSpec):
        self._func = func
        self._is_async = inspect.iscoroutinefunction(func)
        self.spec = spec

    async def call(self, **kwargs) -> ToolResult:
        start = time.monotonic()
        try:
            if self._is_async:
                output = await self._func(**kwargs)
            else:
                output = self._func(**kwargs)
            return ToolResult(
                tool_name=self.spec.name,
                output=output,
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.spec.name,
                output=None,
                error=str(exc),
                duration_ms=(time.monotonic() - start) * 1000,
            )


def tool(
    func: Callable | None = None,
    *,
    requires_approval: bool | str | list[str] = False,
    approval_timeout_seconds: float | None = None,
    name: str | None = None,
):
    """
    Decorator that turns a function into a Kestrion Tool.

    Usage:
        @tool
        def get_cluster_state() -> dict:
            '''Read current deployment replica counts.'''
            ...

        @tool(requires_approval=True)
        def apply_manifest(yaml: str) -> dict:
            '''kubectl apply a manifest against the cluster.'''
            ...

        @tool(requires_approval=["engineer", "manager"])
        def deploy_to_prod() -> dict:
            '''Deploys to production. Needs both an engineer and a manager.'''
            ...

        @tool(requires_approval=True, approval_timeout_seconds=3600.0)
        def restart_service() -> dict:
            '''Restarts a service. Must be approved within an hour.'''
            ...

    The docstring becomes the tool description the LLM sees — make it
    descriptive, since it's the only thing the model has to decide
    whether/how to call the tool.

    NOTE: requires_approval accepting a role string or list of roles
    (approval chains), and approval_timeout_seconds, were both added to
    ToolSpec when those features were built, but this decorator's
    signature was never updated to pass them through — meaning anyone
    using @tool(requires_approval=["a", "b"]) before this fix hit a
    TypeError, since the decorator only accepted a bare bool. Found
    while building an integration demo that actually exercised every
    feature together through @tool, rather than constructing ToolSpec
    directly the way every unit test for chains/timeouts had done.
    """

    def decorator(f: Callable) -> _FunctionTool:
        tool_name = name or f.__name__
        description = (f.__doc__ or "").strip() or f"Calls {tool_name}"
        parameters = _build_parameters_schema(f)

        spec = ToolSpec(
            name=tool_name,
            description=description,
            parameters=parameters,
            requires_approval=requires_approval,
            approval_timeout_seconds=approval_timeout_seconds,
        )

        wrapped = _FunctionTool(f, spec)
        functools.update_wrapper(wrapped, f)
        return wrapped

    if func is not None:
        # Bare @tool usage (no parens, no kwargs)
        return decorator(func)
    return decorator