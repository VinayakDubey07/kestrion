"""
Structured output helpers for Kestrion.
Handles conversion of Pydantic models, dataclasses, and type annotations
into JSON schemas, and parsing/validating LLM text responses into structured objects.
"""

from __future__ import annotations

import json
import re
from typing import Any

from kestrion.core.errors import StructuredOutputError


def resolve_output_schema(schema: Any) -> dict[str, Any]:
    """
    Convert a schema specification (dict, Pydantic model class, or Python type)
    into a JSON Schema dictionary.
    """
    if schema is None:
        return {}
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_json_schema"):
        # Pydantic v2
        return schema.model_json_schema()
    if hasattr(schema, "schema") and callable(schema.schema):
        # Pydantic v1
        return schema.schema()
    
    from kestrion.agent.decorators import _python_type_to_json_schema
    try:
        return _python_type_to_json_schema(schema)
    except Exception:
        return {"type": "object", "description": str(schema)}


def parse_structured_output(text: str | None, schema: Any) -> Any:
    """
    Parse raw LLM output string against the required schema.
    Returns instantiated Pydantic object if schema is a Pydantic model class,
    otherwise returns a parsed Python dict/list/primitive.
    """
    if text is None:
        raise StructuredOutputError("LLM returned empty output when structured output was required.", raw_output=None)
    
    raw = text.strip()
    # Strip markdown code blocks if present (e.g. ```json ... ```)
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if match:
        raw = match.group(1).strip()
    elif raw.startswith("`") and raw.endswith("`"):
        raw = raw[1:-1].strip()

    try:
        data = json.loads(raw)
    except Exception as e:
        raise StructuredOutputError(f"Failed to parse JSON from output: {e}", raw_output=text)

    if schema is None or isinstance(schema, dict):
        return data

    if hasattr(schema, "model_validate"):
        # Pydantic v2
        try:
            return schema.model_validate(data)
        except Exception as e:
            raise StructuredOutputError(f"Pydantic validation error: {e}", raw_output=text)
    if hasattr(schema, "parse_obj"):
        # Pydantic v1
        try:
            return schema.parse_obj(data)
        except Exception as e:
            raise StructuredOutputError(f"Pydantic validation error: {e}", raw_output=text)

    return data
