from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""

    embeddings: np.ndarray
    dimensions: int
    model: str
    metadata: Optional[Dict[str, Any]] = None


class BaseEmbeddingService(ABC):
    """Abstract base class for embedding services."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config is not None else {}

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Embed a list of texts."""
        pass

    @abstractmethod
    async def embed_query(self, query: str) -> EmbeddingResult:
        """Embed a single query."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Get the dimension of the embeddings."""
        pass


class OpenAIEmbeddingService(BaseEmbeddingService):
    """OpenAI embedding service."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = self.config.get("model", "text-embedding-3-small")
        self._dimensions = self.config.get("dimensions", 1536)
        self._client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI

            self._client = OpenAI()
        except ImportError:
            logger.error("OpenAI library not installed. Install with: pip install openai")
            raise

    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Embed a list of texts using OpenAI."""
        if not self._client:
            self._initialize_client()

        try:
            import asyncio

            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._client.embeddings.create(model=self.model, input=texts)
            )

            embeddings = np.array([data.embedding for data in response.data])

            return EmbeddingResult(
                embeddings=embeddings,
                dimensions=len(embeddings[0]),
                model=self.model,
                metadata={"usage": response.usage.model_dump() if response.usage else None},
            )

        except Exception as e:
            logger.error(f"Error embedding texts with OpenAI: {e}")
            raise

    async def embed_query(self, query: str) -> EmbeddingResult:
        """Embed a single query using OpenAI."""
        result = await self.embed_texts([query])
        return EmbeddingResult(
            embeddings=result.embeddings[0],
            dimensions=result.dimensions,
            model=result.model,
            metadata=result.metadata,
        )

    @property
    def dimensions(self) -> int:
        """Get the embedding dimension."""
        return self._dimensions


class SentenceTransformerEmbeddingService(BaseEmbeddingService):
    """Sentence Transformers embedding service."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model_name = self.config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.device = self.config.get("device", "cpu")
        self._model = None
        self._dimensions = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Sentence Transformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dimensions = self._model.get_sentence_embedding_dimension()
        except ImportError:
            logger.error(
                "sentence-transformers library not installed. Install with: pip install sentence-transformers"
            )
            raise

    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Embed a list of texts using Sentence Transformers."""
        if not self._model:
            self._initialize_model()

        try:
            import asyncio

            if not self._model:
                self._initialize_model()

            embeddings = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._model.encode(texts, convert_to_numpy=True)
            )

            return EmbeddingResult(
                embeddings=embeddings,
                dimensions=embeddings.shape[1],
                model=self.model_name,
                metadata={"device": self.device},
            )

            return EmbeddingResult(
                embeddings=embeddings,
                dimensions=embeddings.shape[1],
                model=self.model_name,
                metadata={"device": self.device},
            )

        except Exception as e:
            logger.error(f"Error embedding texts with Sentence Transformers: {e}")
            raise

    async def embed_query(self, query: str) -> EmbeddingResult:
        """Embed a single query using Sentence Transformers."""
        result = await self.embed_texts([query])
        return EmbeddingResult(
            embeddings=result.embeddings[0],
            dimensions=result.dimensions,
            model=result.model,
            metadata=result.metadata,
        )

    @property
    def dimensions(self) -> int:
        """Get the embedding dimension."""
        if self._dimensions is not None:
            return self._dimensions
        # Default dimension for MiniLM
        return 384


class MockEmbeddingService(BaseEmbeddingService):
    """Mock embedding service for testing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._dimensions = self.config.get("dimensions", 384)

    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Generate mock embeddings."""
        import random

        embeddings = np.random.rand(len(texts), self._dimensions).astype(np.float32)

        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=self._dimensions,
            model="mock",
            metadata={"mock": True},
        )

    async def embed_query(self, query: str) -> EmbeddingResult:
        """Generate mock embedding for query."""
        result = await self.embed_texts([query])
        return EmbeddingResult(
            embeddings=result.embeddings[0],
            dimensions=result.dimensions,
            model=result.model,
            metadata=result.metadata,
        )

    @property
    def dimensions(self) -> int:
        """Get the embedding dimension."""
        return self._dimensions


def create_embedding_service(
    provider: str, config: Optional[Dict[str, Any]] = None
) -> BaseEmbeddingService:
    """Create an embedding service based on provider."""
    if provider == "openai":
        return OpenAIEmbeddingService(config)
    elif provider == "sentence-transformers":
        return SentenceTransformerEmbeddingService(config)
    elif provider == "mock":
        return MockEmbeddingService(config)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
