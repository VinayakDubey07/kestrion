import pytest
from kestrion.rag.base import Document
from kestrion.rag.memory_store import MemoryVectorStore
from kestrion.rag.chroma_store import ChromaVectorStore
from kestrion.tools.rag_toolkit import RAGToolkit
from kestrion.agent.agent import Agent
from kestrion.llm.base import LLMResponse, Message
from kestrion.core.types import RunStatus


def test_memory_vector_store():
    store = MemoryVectorStore()
    store.add_documents([
        Document(id="1", page_content="The capital of France is Paris.", metadata={"source": "wiki"}),
        Document(id="2", page_content="Python is a programming language.", metadata={"source": "book"}),
        Document(id="3", page_content="The Eiffel Tower is in Paris.", metadata={"source": "wiki"}),
    ])
    
    # Keyword overlap should match 1 and 3 more than 2
    results = store.similarity_search("Tell me about Paris", k=2)
    assert len(results) == 2
    # Ensure Paris documents are retrieved
    assert "Paris" in results[0].page_content
    assert "Paris" in results[1].page_content


@pytest.mark.asyncio
async def test_chroma_vector_store():
    # Uses EphemeralClient internally if no persist_directory
    store = ChromaVectorStore(collection_name="test_collection")
    store.add_documents([
        Document(id="1", page_content="The quick brown fox jumps over the lazy dog.", metadata={"animal": "fox"}),
        Document(id="2", page_content="A journey of a thousand miles begins with a single step.", metadata={"type": "quote"}),
    ])
    
    results = store.similarity_search("fox", k=1)
    assert len(results) == 1
    assert results[0].id == "1"
    assert results[0].metadata["animal"] == "fox"


def test_rag_toolkit_ingest():
    store = MemoryVectorStore()
    toolkit = RAGToolkit(store=store)
    
    # Test chunking
    text = "A" * 1200
    toolkit.ingest_text(text, document_id="doc1", chunk_size=1000, chunk_overlap=200)
    
    assert len(store._documents) == 2
    assert store._documents[0].id == "doc1_0"
    assert len(store._documents[0].page_content) == 1000
    assert store._documents[1].id == "doc1_1"
    # overlapping by 200, so remaining is 400 (from 800 to 1200)
    assert len(store._documents[1].page_content) == 400


class MockLLMProvider:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.call_count = 0

    async def complete(self, messages: list[Message], tools: list[dict], system: str | None = None, **kwargs) -> LLMResponse:
        self.call_count += 1
        # On first turn, model decides to search knowledge base
        if self.call_count == 1:
            from kestrion.llm.base import ToolCallRequest
            return LLMResponse(
                stop_reason="tool_calls",
                text="",
                tool_calls=[ToolCallRequest(id="call_123", name="search_knowledge_base", arguments={"query": "Mars"})],
                tokens_in=10,
                tokens_out=10,
                cost_usd=0.0
            )
        # On second turn, model gives final answer
        return LLMResponse(
            stop_reason="stop",
            text=self.response_text,
            tool_calls=[],
            tokens_in=10,
            tokens_out=10,
            cost_usd=0.0
        )


@pytest.mark.asyncio
async def test_rag_agent_integration():
    store = MemoryVectorStore()
    toolkit = RAGToolkit(store=store)
    toolkit.ingest_text("Mars is the fourth planet from the Sun.", "doc1")
    
    provider = MockLLMProvider(response_text="Mars is the red planet.")
    agent = Agent(
        provider=provider, # type: ignore
        tools=toolkit.get_tools()
    )
    
    result = await agent.run("Tell me about Mars.")
    assert result.status == RunStatus.COMPLETED
    assert result.output == "Mars is the red planet."
    
    # Verify the tool was actually called and injected context
    # We can inspect the state to see if search_knowledge_base was added to scratch
    messages = result.state.scratch["_messages"]
    tool_results = [m for m in messages if m["role"] == "tool"]
    assert len(tool_results) == 1
    assert "Mars is the fourth planet" in tool_results[0]["content"]
