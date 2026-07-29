from kestrion.rag.base import Document, VectorStore

try:
    import chromadb
except ImportError:
    chromadb = None  # type: ignore


class ChromaVectorStore(VectorStore):
    """
    A VectorStore implementation using ChromaDB.
    Requires `pip install chromadb`.
    """

    def __init__(self, collection_name: str = "kestrion_knowledge_base", persist_directory: str | None = None):
        if chromadb is None:
            raise ImportError("ChromaDB is not installed. Please install it using `pip install chromadb` or `pip install kestrion[chroma]`.")
        
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()
            
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return

        ids = [doc.id for doc in documents]
        texts = [doc.page_content for doc in documents]
        
        # ChromaDB requires either a list of non-empty dicts, or None for the entire metadatas arg
        # If no documents have metadata, we pass None. Otherwise we must pass a list where items are dicts or None.
        has_metadata = any(doc.metadata for doc in documents)
        metadatas = [doc.metadata if doc.metadata else None for doc in documents] if has_metadata else None

        self._collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        results = self._collection.query(
            query_texts=[query],
            n_results=k
        )
        
        documents = []
        if not results or not results.get("documents") or not results["documents"][0]:
            return documents
            
        # results["documents"] is a list of lists of strings
        # results["metadatas"] is a list of lists of dicts
        # results["ids"] is a list of lists of strings
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            text = results["documents"][0][i]
            meta = results["metadatas"][0][i] if results.get("metadatas") else None
            
            documents.append(Document(
                id=doc_id,
                page_content=text,
                metadata=meta
            ))
            
        return documents
