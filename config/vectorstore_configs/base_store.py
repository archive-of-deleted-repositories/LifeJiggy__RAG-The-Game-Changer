"""
Vector Store Base Classes - RAG-The-Game-Changer

Abstract base classes for vector storage implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreResult:
    """Result from vector store operations."""

    success: bool
    message: str
    ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""

    ids: List[str]
    scores: List[float]
    metadata: List[Dict[str, Any]]
    total_results: int
    search_time_ms: float


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the vector store connection."""
        pass

    @abstractmethod
    async def add_vectors(
        self,
        vectors: Union[np.ndarray, List[np.ndarray]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> VectorStoreResult:
        """Add vectors to the store."""
        pass

    @abstractmethod
    async def search(
        self, query_vector: np.ndarray, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> VectorSearchResult:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def delete(self, ids: List[str]) -> VectorStoreResult:
        """Delete vectors by IDs."""
        pass

    @abstractmethod
    async def update(
        self,
        vectors: Union[np.ndarray, List[np.ndarray]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> VectorStoreResult:
        """Update vectors by IDs."""
        pass

    @abstractmethod
    async def clear(self) -> VectorStoreResult:
        """Clear all vectors from the store."""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the vector store."""
        try:
            if not self._initialized:
                await self.initialize()

            stats = await self.get_stats()
            return {"status": "healthy", "initialized": self._initialized, "stats": stats}
        except Exception as e:
            return {"status": "unhealthy", "initialized": self._initialized, "error": str(e)}
