import pytest
from pydantic import BaseModel, Field

from kestrion.agent.decorators import tool


class SearchQuery(BaseModel):
    query: str = Field(description="Search keyword")
    limit: int = Field(default=5, ge=1, le=50, description="Max results")


class AdvancedFilter(BaseModel):
    tags: list[str] = Field(default_factory=list)
    case_sensitive: bool = False


# 1. Single Pydantic parameter (auto-flattened schema)
@tool
def search_database(q: SearchQuery) -> str:
    """Search the database using structured query."""
    return f"results for {q.query} (limit {q.limit})"


# 2. Tool with explicit args_schema
class DeployRequest(BaseModel):
    service: str = Field(description="Target service name")
    replicas: int = Field(default=3, ge=1)
    env: str = Field(default="staging")


@tool(args_schema=DeployRequest)
def deploy_service(**kwargs) -> str:
    """Deploy service."""
    return f"deployed {kwargs.get('service')} with {kwargs.get('replicas')} replicas to {kwargs.get('env')}"


# 3. Tool with multiple parameters including a Pydantic model
@tool
def advanced_search(keyword: str, filter: AdvancedFilter, page: int = 1) -> str:
    """Advanced search."""
    return f"keyword={keyword}, tags={filter.tags}, case={filter.case_sensitive}, page={page}"


# 4. Async tool with Pydantic parameter
@tool
async def async_fetch(q: SearchQuery) -> str:
    """Async search."""
    return f"async results for {q.query}"


def test_pydantic_tool_schema_generation():
    # Schema should have flattened properties from SearchQuery
    spec = search_database.spec
    assert spec.name == "search_database"
    assert "query" in spec.parameters["properties"]
    assert "limit" in spec.parameters["properties"]
    assert spec.parameters["properties"]["query"]["type"] == "string"
    assert spec.parameters["properties"]["limit"]["type"] == "integer"
    assert "query" in spec.parameters["required"]


def test_pydantic_args_schema_parameter():
    spec = deploy_service.spec
    assert spec.name == "deploy_service"
    assert "service" in spec.parameters["properties"]
    assert "replicas" in spec.parameters["properties"]
    assert "env" in spec.parameters["properties"]


@pytest.mark.asyncio
async def test_pydantic_tool_execution_flattened_args():
    # LLM passes top-level fields
    res = await search_database.call(query="agent architecture", limit=10)
    assert res.output == "results for agent architecture (limit 10)"
    assert res.error is None


@pytest.mark.asyncio
async def test_pydantic_tool_execution_nested_dict():
    # LLM passes parameter wrapped in dict
    res = await search_database.call(q={"query": "state machine", "limit": 3})
    assert res.output == "results for state machine (limit 3)"
    assert res.error is None


@pytest.mark.asyncio
async def test_pydantic_tool_validation_error():
    # Limit exceeds le=50
    res = await search_database.call(query="test", limit=999)
    assert res.output is None
    assert res.error is not None
    assert "less than or equal to 50" in res.error


@pytest.mark.asyncio
async def test_pydantic_args_schema_call():
    res = await deploy_service.call(service="kestrion-core", replicas=5, env="prod")
    assert res.output == "deployed kestrion-core with 5 replicas to prod"
    assert res.error is None


@pytest.mark.asyncio
async def test_pydantic_multiple_params():
    res = await advanced_search.call(
        keyword="metrics",
        filter={"tags": ["fast", "reliable"], "case_sensitive": True},
        page=2,
    )
    assert res.output == "keyword=metrics, tags=['fast', 'reliable'], case=True, page=2"
    assert res.error is None


@pytest.mark.asyncio
async def test_pydantic_async_tool():
    res = await async_fetch.call(query="async test")
    assert res.output == "async results for async test"
    assert res.error is None
