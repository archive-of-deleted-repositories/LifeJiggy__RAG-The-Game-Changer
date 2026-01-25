"""
Dense Retriever - RAG-The-Game-Changer

Dense retrieval using embedding-based similarity search.
"""

import asyncio
import numpy as np
from typing import Any, Dict, List, Optional
import logging

from .base import BaseRetriever, RetrievalResult, RetrievedChunk, RetrievalStrategy
from config.embedding_configs.embedding_service import create_embedding_service

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """Dense retriever using vector similarity search."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.embedding_provider = self.config.get("embedding_provider", "openai")
        self.top_k = self.config.get("top_k", 5)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)

        # Initialize embedding service
        embedding_config = self.config.get("embedding", {})
        self.embedder = create_embedding_service(self.embedding_provider, embedding_config)

        # Mock vector store (will be replaced by proper vector store implementation)
        self._document_embeddings: List[np.ndarray] = []
        self._documents: List[Dict[str, Any]] = []

    async def retrieve(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Retrieve documents using dense similarity search."""
        start_time = asyncio.get_event_loop().time()

        try:
            # Embed the query
            query_result = await self.embedder.embed_query(query)
            query_embedding = query_result.embeddings

            if not self._document_embeddings:
                return RetrievalResult(
                    query=query,
                    chunks=[],
                    strategy=RetrievalStrategy.DENSE,
                    total_chunks=0,
                    retrieval_time_ms=0,
                )

            # Calculate similarities
            similarities = self._calculate_similarities(query_embedding)

            # Get top-k results
            top_k = min(top_k or self.top_k, len(similarities))
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            chunks = []
            for idx in top_indices:
                if similarities[idx] >= self.similarity_threshold:
                    doc = self._documents[idx]
                    chunk = RetrievedChunk(
                        content=doc.get("content", ""),
                        document_id=doc.get("document_id", f"doc_{idx}"),
                        chunk_id=doc.get("chunk_id", f"chunk_{idx}"),
                        score=float(similarities[idx]),
                        metadata=doc.get("metadata", {}),
                        source=doc.get("source"),
                        title=doc.get("title"),
                    )
                    chunks.append(chunk)

            retrieval_time = (asyncio.get_event_loop().time() - start_time) * 1000

            return RetrievalResult(
                query=query,
                chunks=chunks,
                strategy=RetrievalStrategy.DENSE,
                total_chunks=len(chunks),
                retrieval_time_ms=retrieval_time,
            )

        except Exception as e:
            logger.error(f"Error in dense retrieval: {e}")
            return RetrievalResult(
                query=query,
                chunks=[],
                strategy=RetrievalStrategy.DENSE,
                total_chunks=0,
                retrieval_time_ms=0,
            )

    def _calculate_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """Calculate cosine similarities between query and documents."""
        if not self._document_embeddings:
            return np.array([])

        # Convert document embeddings to numpy array if needed
        doc_embeddings = np.array(self._document_embeddings)

        # Calculate cosine similarity
        dot_product = np.dot(doc_embeddings, query_embedding)
        norm_query = np.linalg.norm(query_embedding)
        norm_docs = np.linalg.norm(doc_embeddings, axis=1)

        similarities = dot_product / (norm_query * norm_docs + 1e-8)
        return similarities

    async def add_documents(self, documents: List[Dict[str, Any]], **kwargs) -> bool:
        """Add documents to the dense retriever."""
        try:
            for doc in documents:
                content = doc.get("content", "")
                if content:
                    # Embed the document
                    embed_result = await self.embedder.embed_texts([content])
                    embedding = embed_result.embeddings[0]

                    # Store embedding and document
                    self._document_embeddings.append(embedding)
                    self._documents.append(doc)

            logger.info(f"Added {len(documents)} documents to dense retriever")
            return True

        except Exception as e:
            logger.error(f"Error adding documents to dense retriever: {e}")
            return False

    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the dense retriever."""
        try:
            indices_to_remove = []
            for i, doc in enumerate(self._documents):
                if doc.get("document_id") in document_ids:
                    indices_to_remove.append(i)

            # Remove in reverse order to maintain indices
            for idx in reversed(indices_to_remove):
                del self._documents[idx]
                del self._document_embeddings[idx]

            logger.info(f"Deleted {len(indices_to_remove)} documents from dense retriever")
            return True

        except Exception as e:
            logger.error(f"Error deleting documents from dense retriever: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all documents from the dense retriever."""
        try:
            self._document_embeddings.clear()
            self._documents.clear()
            logger.info("Cleared all documents from dense retriever")
            return True

        except Exception as e:
            logger.error(f"Error clearing dense retriever: {e}")
            return False
