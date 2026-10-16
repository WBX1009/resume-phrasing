from abc import ABC, abstractmethod
from typing import List


class BaseEmbedding(ABC):
    """Port for text embedding adapters."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed one query or short text."""
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents or chunks."""
        raise NotImplementedError
