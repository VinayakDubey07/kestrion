import pytest
from typing import Any, Optional, List, Dict

from kestrion.agent.decorators import _python_type_to_json_schema, _build_parameters_schema, tool

def test_type_to_json_schema():
    assert _python_type_to_json_schema(str) == {"type": "string"}
    assert _python_type_to_json_schema(int) == {"type": "integer"}
    assert _python_type_to_json_schema(float) == {"type": "number"}
    assert _python_type_to_json_schema(bool) == {"type": "boolean"}
    
    assert _python_type_to_json_schema(Optional[str]) == {"type": "string"}
    assert _python_type_to_json_schema(list[str]) == {"type": "array", "items": {"type": "string"}}
    assert _python_type_to_json_schema(List[int]) == {"type": "array", "items": {"type": "integer"}}
    assert _python_type_to_json_schema(dict) == {"type": "object"}
    assert _python_type_to_json_schema(Dict[str, Any]) == {"type": "object"}
    
    # Unrecognized types should fallback gracefully or raise as designed
    class CustomClass:
        pass
    with pytest.raises(NotImplementedError):
        _python_type_to_json_schema(CustomClass)

def test_build_parameters_schema():
    def sample_func(
        a: str, 
        b: int = 5, 
        c: Optional[list[str]] = None
    ):
        """
        My function.
        
        Args:
            a: Description of a
            b: Description of b
            c: Description of c
        """
        pass

    schema = _build_parameters_schema(sample_func)
    
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"a"}
    
    props = schema["properties"]
    assert props["a"] == {"type": "string", "description": "Description of a"}
    assert props["b"] == {"type": "integer", "description": "Description of b", "default": 5}
    assert props["c"] == {"type": "array", "items": {"type": "string"}, "description": "Description of c", "default": None}

def test_tool_decorator_extracts_schema():
    @tool(requires_approval=True)
    def my_tool(x: float):
        """
        Does some math.
        
        :param x: The value to calculate.
        """
        return x * 2.0
        
    assert my_tool.spec.name == "my_tool"
    assert my_tool.spec.description == "Does some math.\n        \n        :param x: The value to calculate."
    assert my_tool.spec.requires_approval is True
    
    props = my_tool.spec.parameters["properties"]
    assert props["x"] == {"type": "number", "description": "The value to calculate."}
