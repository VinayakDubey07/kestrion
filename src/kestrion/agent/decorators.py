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
import re
import time
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

from kestrion.core.types import Tool, ToolResult, ToolSpec

_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _is_pydantic_model(cls: Any) -> bool:
    try:
        from pydantic import BaseModel
        return isinstance(cls, type) and issubclass(cls, BaseModel)
    except Exception:
        return False


def _pydantic_model_to_json_schema(model_cls: Any) -> dict[str, Any]:
    if hasattr(model_cls, "model_json_schema") and callable(model_cls.model_json_schema):
        return model_cls.model_json_schema()
    if hasattr(model_cls, "schema") and callable(model_cls.schema):
        return model_cls.schema()
    try:
        from pydantic import TypeAdapter
        return TypeAdapter(model_cls).json_schema()
    except Exception as exc:
        raise NotImplementedError(f"@tool cannot auto-generate a JSON schema for {model_cls!r}: {exc}") from exc


def _python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """
    Best-effort conversion of a type annotation to a JSON schema
    fragment. Handles: plain types, Optional[X] (-> X's schema, since
    JSON schema has no first-class None), list[X], Pydantic models,
    dataclasses, and falls back to "string" for anything unrecognized.
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

    if _is_pydantic_model(annotation):
        return _pydantic_model_to_json_schema(annotation)

    import dataclasses
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        return _pydantic_model_to_json_schema(annotation)

    if annotation in _TYPE_MAP:
        return {"type": _TYPE_MAP[annotation]}

    if annotation is inspect.Parameter.empty:
        # No type hint at all — still allow the tool to be registered,
        # but the schema is maximally permissive rather than guessed.
        return {"type": "string"}

    # Unrecognized annotation (e.g. custom classes).
    raise NotImplementedError(
        f"@tool cannot auto-generate a JSON schema for annotation {annotation!r}. "
        f"Supported: str, int, float, bool, list[T], dict, Pydantic models, and Optional[T] of these."
    )


def _parse_docstring_params(doc: str | None) -> dict[str, str]:
    if not doc:
        return {}

    params = {}
    lines = doc.split("\n")

    in_args_section = False
    current_param = None

    for line in lines:
        stripped = line.strip()

        if stripped in ("Args:", "Arguments:", "Parameters:"):
            in_args_section = True
            continue

        if in_args_section and stripped == "":
            continue
        if in_args_section and re.match(r'^[A-Z][a-zA-Z]+:', stripped):
            in_args_section = False

        sphinx_match = re.match(r':param\s+(?:[\w\[\]|]+\s+)?(\w+)\s*:\s*(.*)', stripped)
        if sphinx_match:
            current_param = sphinx_match.group(1)
            params[current_param] = sphinx_match.group(2).strip()
            continue

        if in_args_section:
            google_match = re.match(r'^(\w+)\s*(?:\([^)]+\))?\s*:\s*(.*)', stripped)
            if google_match:
                current_param = google_match.group(1)
                params[current_param] = google_match.group(2).strip()
                continue

        if (
            current_param
            and stripped
            and not sphinx_match
            and not re.match(r'^(\w+)\s*(?:\([^)]+\))?\s*:', stripped)
            and not stripped.startswith(":")
        ):
            params[current_param] += " " + stripped

    return params


def _build_parameters_schema(func: Callable, args_schema: type | None = None) -> dict[str, Any]:
    if args_schema is not None and _is_pydantic_model(args_schema):
        return _pydantic_model_to_json_schema(args_schema)

    sig = inspect.signature(func)
    try:
        type_hints = get_type_hints(func)
    except Exception:
        type_hints = {}
    param_docs = _parse_docstring_params(func.__doc__)
    properties: dict[str, Any] = {}
    required: list[str] = []

    non_internal = [
        (name, param, type_hints.get(name, param.annotation))
        for name, param in sig.parameters.items()
        if name != "self" and not name.startswith("_")
    ]

    # If exactly ONE parameter and it is a Pydantic model:
    # Flatten into the top-level tool parameters schema for LLMs
    if len(non_internal) == 1 and _is_pydantic_model(non_internal[0][2]):
        _, _, model_cls = non_internal[0]
        return _pydantic_model_to_json_schema(model_cls)

    for name, param, annotation in non_internal:
        if _is_pydantic_model(annotation):
            prop_schema = _pydantic_model_to_json_schema(annotation)
        else:
            prop_schema = _python_type_to_json_schema(annotation)

        if name in param_docs:
            prop_schema["description"] = param_docs[name]

        properties[name] = prop_schema
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

    def __init__(self, func: Callable, spec: ToolSpec, args_schema: type | None = None):
        self._func = func
        self._is_async = inspect.iscoroutinefunction(func) or inspect.iscoroutinefunction(getattr(func, '__func__', None))
        self.spec = spec
        self.args_schema = args_schema
        self._sig = inspect.signature(func)
        self._sig_params = set(self._sig.parameters.keys())
        self._has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in self._sig.parameters.values()
        )
        try:
            self._type_hints = get_type_hints(func)
        except Exception:
            self._type_hints = {}
        non_internal = [
            (name, self._type_hints.get(name, param.annotation))
            for name, param in self._sig.parameters.items()
            if name != "self" and not name.startswith("_")
        ]
        self._single_pydantic_param = (
            non_internal[0][0]
            if len(non_internal) == 1 and _is_pydantic_model(non_internal[0][1])
            else None
        )
        self._pydantic_params = {
            name: ann for name, ann in non_internal if _is_pydantic_model(ann)
        }

    def __get__(self, instance, owner=None) -> _FunctionTool:
        if instance is None:
            return self
        import types
        bound_func = types.MethodType(self._func, instance) if not isinstance(self._func, types.MethodType) else self._func
        bound_tool = _FunctionTool(bound_func, self.spec, args_schema=self.args_schema)
        functools.update_wrapper(bound_tool, bound_func)  # type: ignore[arg-type]
        return bound_tool

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    async def call(self, **kwargs) -> ToolResult:
        start = time.monotonic()
        try:
            call_kwargs = dict(kwargs)

            # 1. If explicit args_schema was passed to @tool(args_schema=...)
            if self.args_schema is not None and _is_pydantic_model(self.args_schema):
                clean_args = {k: v for k, v in call_kwargs.items() if not k.startswith("_")}
                validated = self.args_schema.model_validate(clean_args)
                if self._single_pydantic_param and self._single_pydantic_param in self._sig_params:
                    call_kwargs[self._single_pydantic_param] = validated
                else:
                    dump = validated.model_dump() if hasattr(validated, "model_dump") else validated.dict()
                    if self._has_var_keyword:
                        call_kwargs.update(dump)
                    else:
                        for k, v in dump.items():
                            if k in self._sig_params:
                                call_kwargs[k] = v

            # 2. If single Pydantic parameter (flattened top-level schema)
            elif self._single_pydantic_param:
                p_name = self._single_pydantic_param
                model_cls = self._pydantic_params.get(p_name)
                if model_cls and _is_pydantic_model(model_cls):
                    if p_name in call_kwargs:
                        val = call_kwargs[p_name]
                        if isinstance(val, dict):
                            call_kwargs[p_name] = model_cls.model_validate(val)
                        elif isinstance(val, model_cls):
                            call_kwargs[p_name] = val
                    else:
                        clean_args = {k: v for k, v in call_kwargs.items() if not k.startswith("_")}
                        call_kwargs[p_name] = model_cls.model_validate(clean_args)

            # 3. If multiple parameters include Pydantic models
            if self._pydantic_params:
                for p_name, model_cls in self._pydantic_params.items():
                    if p_name in call_kwargs and _is_pydantic_model(model_cls):
                        val = call_kwargs[p_name]
                        if isinstance(val, dict):
                            call_kwargs[p_name] = model_cls.model_validate(val)

            # Filter out injected kwargs (like _state) that this function doesn't expect
            if self._has_var_keyword:
                filtered_kwargs = {k: v for k, v in call_kwargs.items() if not k.startswith("_")}
            else:
                filtered_kwargs = {k: v for k, v in call_kwargs.items() if k in self._sig_params}

            if self._is_async:
                output = await self._func(**filtered_kwargs)
            else:
                output = self._func(**filtered_kwargs)
            return ToolResult(
                tool_name=self.spec.name,
                output=output,
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            # Re-raise control-flow exceptions that the engine relies on
            from kestrion.core.errors import InputRequired, HandoffCompleted
            if isinstance(exc, (InputRequired, HandoffCompleted)):
                raise
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
    args_schema: type | None = None,
):
    """
    Decorator that turns a function into a Kestrion Tool.
    Supports standard Python type annotations, Pydantic BaseModel parameters,
    and explicit args_schema models for robust validation and schema generation.

    Usage:
        @tool
        def get_cluster_state() -> dict:
            '''Read current deployment replica counts.'''
            ...

        @tool(args_schema=DeployRequest)
        def deploy_service(req: DeployRequest) -> dict:
            '''Deploys service according to DeployRequest schema.'''
            ...
    """

    def decorator(f: Callable) -> _FunctionTool:
        tool_name = name or f.__name__
        description = (f.__doc__ or "").strip() or f"Calls {tool_name}"
        parameters = _build_parameters_schema(f, args_schema=args_schema)

        spec = ToolSpec(
            name=tool_name,
            description=description,
            parameters=parameters,
            requires_approval=requires_approval,
            approval_timeout_seconds=approval_timeout_seconds,
        )

        wrapped = _FunctionTool(f, spec, args_schema=args_schema)
        functools.update_wrapper(wrapped, f)  # type: ignore[arg-type]
        return wrapped

    if func is not None:
        # Bare @tool usage (no parens, no kwargs)
        return decorator(func)
    return decorator