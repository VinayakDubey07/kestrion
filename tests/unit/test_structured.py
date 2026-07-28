import pytest
from pydantic import BaseModel, Field
from kestrion.core.errors import StructuredOutputError
from kestrion.llm.structured import resolve_output_schema, parse_structured_output


class SampleModel(BaseModel):
    name: str
    age: int = Field(..., gt=0)
    tags: list[str] = []


def test_resolve_output_schema():
    schema = resolve_output_schema(SampleModel)
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]

    assert resolve_output_schema({"type": "string"}) == {"type": "string"}
    assert resolve_output_schema(str) == {"type": "string"}
    assert resolve_output_schema(None) == {}


def test_parse_structured_output():
    # Clean JSON
    text = '{"name": "Alice", "age": 30, "tags": ["admin"]}'
    obj = parse_structured_output(text, SampleModel)
    assert isinstance(obj, SampleModel)
    assert obj.name == "Alice"
    assert obj.age == 30
    assert obj.tags == ["admin"]

    # Markdown wrapped JSON
    text_md = 'Here is the data:\n```json\n{"name": "Bob", "age": 25}\n```'
    obj2 = parse_structured_output(text_md, SampleModel)
    assert isinstance(obj2, SampleModel)
    assert obj2.name == "Bob"
    assert obj2.age == 25
    assert obj2.tags == []

    # Dict schema fallback
    dict_obj = parse_structured_output('{"foo": "bar"}', {"type": "object"})
    assert dict_obj == {"foo": "bar"}


def test_parse_structured_output_errors():
    with pytest.raises(StructuredOutputError, match="Failed to parse JSON"):
        parse_structured_output("not json", SampleModel)

    with pytest.raises(StructuredOutputError, match="Pydantic validation error"):
        parse_structured_output('{"name": "Invalid Age", "age": -5}', SampleModel)


@pytest.mark.asyncio
async def test_agent_structured_output_integration(tmp_path):
    from kestrion.agent.agent import Agent
    from kestrion.llm.base import LLMResponse

    class FakeStructuredProvider:
        async def complete(self, messages, tools, system=None, output_schema=None):
            return LLMResponse(
                text='{"name": "Alice", "age": 28, "tags": ["engineer"]}',
                tool_calls=[],
                stop_reason="stop",
            )

    store_url = f"sqlite:///{tmp_path}/structured.db"
    agent = Agent(provider=FakeStructuredProvider(), store=store_url, output_schema=SampleModel)
    result = await agent.run("Get employee Alice")
    
    assert isinstance(result.output, SampleModel)
    assert result.output.name == "Alice"
    assert result.output.age == 28
    assert result.output.tags == ["engineer"]

