import asyncio
from kestrion.agent import Agent
from kestrion.llm import OllamaProvider
from kestrion.tools import RAGToolkit
from kestrion.rag import ChromaVectorStore

# Mock document simulating an internal company policy or documentation
INTERNAL_WIKI = """
# Project Phoenix Deployment Guidelines

1. All deployments must occur during the maintenance window: 2:00 AM - 4:00 AM UTC.
2. The primary database for Project Phoenix is hosted on AWS RDS (Postgres).
3. If a rollback is required, the on-call engineer must execute the `rollback_phoenix.sh` script and notify the #ops channel.
4. The secret key for the staging environment is stored in AWS Secrets Manager under 'phoenix/staging/api_key'.
"""

async def main():
    print("Initializing ChromaDB (this might take a few seconds on first run to download the embedding model)...")
    
    # We use EphemeralClient (in-memory) for this demo.
    # To save to disk, use: ChromaVectorStore(persist_directory="./knowledge_base")
    store = ChromaVectorStore(collection_name="demo_collection")
    toolkit = RAGToolkit(store)

    print("Ingesting internal wiki into vector store...")
    toolkit.ingest_text(
        text=INTERNAL_WIKI,
        document_id="doc_phoenix_guidelines",
        chunk_size=500,
        chunk_overlap=50
    )

    print("Initializing Agent equipped with RAG Toolkit...")
    agent = Agent(
        provider=OllamaProvider(model="llama3.2"), # Using Ollama for local execution
        tools=toolkit.get_tools()
    )

    query = "What database does Project Phoenix use, and when should we deploy it?"
    print(f"\nUser: {query}")
    print("\nAgent is thinking (and searching the knowledge base)...")
    
    result = await agent.run(query)
    
    print(f"\nAgent: {result.output}")
    print("\nCheck the Agent's scratch state to see the tool calls:")
    messages = result.state.scratch["_messages"]
    for msg in messages:
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            print(f" -> Tool called: {msg['tool_calls'][0]['name']} with args {msg['tool_calls'][0]['arguments']}")
        elif msg["role"] == "tool":
            print(f" -> Tool returned knowledge base results!")

if __name__ == "__main__":
    asyncio.run(main())
