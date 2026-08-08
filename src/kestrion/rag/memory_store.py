import re
from kestrion.rag.base import Document, VectorStore


class MemoryVectorStore(VectorStore):
    """
    A zero-dependency fallback store that uses simple keyword overlap (bag of words)
    instead of real embeddings. Extremely fast for unit testing and simple exact-match
    scenarios where you don't want to download a real embedding model.
    """

    def __init__(self) -> None:
        self._documents: list[Document] = []

    def add_documents(self, documents: list[Document]) -> None:
        self._documents.extend(documents)

    def _tokenize(self, text: str) -> set[str]:
        # Simple lowercase alphanumeric tokenization
        return set(re.findall(r'\w+', text.lower()))

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        if not self._documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return self._documents[:k]

        scored_docs = []
        for doc in self._documents:
            doc_tokens = self._tokenize(doc.page_content)
            # Simple Jaccard similarity-esque score
            intersection = len(query_tokens.intersection(doc_tokens))
            union = len(query_tokens.union(doc_tokens))
            score = intersection / union if union > 0 else 0
            scored_docs.append((score, doc))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:k]]
