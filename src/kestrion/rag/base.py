import abc
from dataclasses import dataclass
from typing import Any

@dataclass
class Document:
    """
    A chunk of text along with its metadata and unique ID.
    """
    id: str
    page_content: str
    metadata: dict[str, Any] | None = None


class VectorStore(abc.ABC):
    """
    Protocol for a vector database that can store and retrieve Documents.
    """
    @abc.abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        """
        Add a list of Documents to the vector store.
        """
        pass

    @abc.abstractmethod
    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        """
        Search the vector store for the top k documents most similar to the query.
        """
        pass
