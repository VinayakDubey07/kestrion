import json
from kestrion.agent.decorators import tool, Tool
from kestrion.rag.base import Document, VectorStore


class RAGToolkit:
    """
    A toolkit that exposes vector database search capabilities to an agent.
    """

    def __init__(self, store: VectorStore):
        self.store = store

    def ingest_text(
        self, 
        text: str, 
        document_id: str, 
        metadata: dict | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> None:
        """
        Helper method to chunk a large text and ingest it into the vector store.
        (Not an agent tool; called by the developer before running the agent).
        """
        documents = []
        
        # Simple chunking logic
        if len(text) <= chunk_size:
            documents.append(Document(id=f"{document_id}_0", page_content=text, metadata=metadata))
        else:
            start = 0
            chunk_idx = 0
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                documents.append(
                    Document(id=f"{document_id}_{chunk_idx}", page_content=chunk, metadata=metadata)
                )
                start += (chunk_size - chunk_overlap)
                chunk_idx += 1

        self.store.add_documents(documents)

    def get_tools(self) -> list[Tool]:
        """
        Return the tools to be passed to the Agent.
        """
        
        @tool
        def search_knowledge_base(query: str, n_results: int = 3) -> str:
            """
            Search the knowledge base for documents semantically related to the query.
            Use this tool when you need external facts, context, or documentation 
            to answer the user's question.
            """
            results = self.store.similarity_search(query, k=n_results)
            if not results:
                return "No relevant documents found in the knowledge base."
                
            formatted = []
            for i, doc in enumerate(results):
                meta_str = f" (Metadata: {json.dumps(doc.metadata)})" if doc.metadata else ""
                formatted.append(f"--- Document {i+1}{meta_str} ---\n{doc.page_content}")
                
            return "\n\n".join(formatted)

        return [search_knowledge_base]
