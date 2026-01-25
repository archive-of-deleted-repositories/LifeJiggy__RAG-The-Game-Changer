from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import numpy as np


class RetrievalStrategy(Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with relevance information."""
    content: str
    document_id: str
    chunk_id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    title: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    query: str
    chunks: List[RetrievedChunk]
    strategy: RetrievalStrategy
    total_chunks: int
    retrieval_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """Abstract base class for retrieval strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    async def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a query."""
        pass
    
    @abstractmethod
    async def add_documents(
        self, 
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> bool:
        """Add documents to the index."""
        pass
    
    @abstractmethod
    async def delete_documents(
        self, 
        document_ids: List[str]
    ) -> bool:
        """Delete documents from the index."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all documents from the index."""
        pass
