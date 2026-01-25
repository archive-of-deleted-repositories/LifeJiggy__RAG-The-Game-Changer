"""
Hybrid Retriever - RAG-The-Game-Changer

Hybrid retrieval combining dense and sparse search methods.
"""

import asyncio
from typing import Any, Dict, List, Optional
import logging

from .dense_retriever import DenseRetriever
from .sparse_retriever import SparseRetriever
from .base import BaseRetriever, RetrievalResult, RetrievedChunk, RetrievalStrategy

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining dense and sparse retrieval."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.top_k = self.config.get("top_k", 5)
        self.dense_weight = self.config.get("dense_weight", 0.7)
        self.sparse_weight = self.config.get("sparse_weight", 0.3)

        # Initialize sub-retrievers
        self.dense_retriever = DenseRetriever(config.get("dense", {}))
        self.sparse_retriever = SparseRetriever(config.get("sparse", {}))

    async def retrieve(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Retrieve documents using hybrid search."""
        start_time = asyncio.get_event_loop().time()

        try:
            # Retrieve from both methods
            dense_result, sparse_result = await asyncio.gather(
                self.dense_retriever.retrieve(query, top_k * 2, filters),
                self.sparse_retriever.retrieve(query, top_k * 2, filters),
            )

            # Combine and rerank results
            combined_chunks = self._combine_results(
                dense_result.chunks, sparse_result.chunks, query
            )

            # Get top-k combined results
            top_k = min(top_k or self.top_k, len(combined_chunks))
            top_chunks = combined_chunks[:top_k]

            retrieval_time = (asyncio.get_event_loop().time() - start_time) * 1000

            return RetrievalResult(
                query=query,
                chunks=top_chunks,
                strategy=RetrievalStrategy.HYBRID,
                total_chunks=len(top_chunks),
                retrieval_time_ms=retrieval_time,
                metadata={
                    "dense_chunks": len(dense_result.chunks),
                    "sparse_chunks": len(sparse_result.chunks),
                    "dense_weight": self.dense_weight,
                    "sparse_weight": self.sparse_weight,
                },
            )

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return RetrievalResult(
                query=query,
                chunks=[],
                strategy=RetrievalStrategy.HYBRID,
                total_chunks=0,
                retrieval_time_ms=0,
            )

    def _combine_results(
        self, dense_chunks: List[RetrievedChunk], sparse_chunks: List[RetrievedChunk], query: str
    ) -> List[RetrievedChunk]:
        """Combine results from dense and sparse retrieval."""
        # Create a map to avoid duplicates
        combined_map = {}

        # Add dense results
        for chunk in dense_chunks:
            doc_id = chunk.document_id
            combined_map[doc_id] = {"chunk": chunk, "dense_score": chunk.score, "sparse_score": 0.0}

        # Add sparse results
        for chunk in sparse_chunks:
            doc_id = chunk.document_id
            if doc_id in combined_map:
                combined_map[doc_id]["sparse_score"] = chunk.score
            else:
                combined_map[doc_id] = {
                    "chunk": chunk,
                    "dense_score": 0.0,
                    "sparse_score": chunk.score,
                }

        # Calculate combined scores and sort
        combined_chunks = []
        for doc_id, data in combined_map.items():
            combined_score = (
                self.dense_weight * data["dense_score"] + self.sparse_weight * data["sparse_score"]
            )

            chunk = RetrievedChunk(
                content=data["chunk"].content,
                document_id=data["chunk"].document_id,
                chunk_id=data["chunk"].chunk_id,
                score=combined_score,
                metadata=data["chunk"].metadata,
                source=data["chunk"].source,
                title=data["chunk"].title,
            )
            combined_chunks.append(chunk)

        # Sort by combined score
        combined_chunks.sort(key=lambda x: x.score, reverse=True)
        return combined_chunks

    async def add_documents(self, documents: List[Dict[str, Any]], **kwargs) -> bool:
        """Add documents to both retrievers."""
        try:
            # Add to both retrievers
            dense_success, sparse_success = await asyncio.gather(
                self.dense_retriever.add_documents(documents, **kwargs),
                self.sparse_retriever.add_documents(documents, **kwargs),
            )

            success = dense_success and sparse_success
            logger.info(f"Added {len(documents)} documents to hybrid retriever: {success}")
            return success

        except Exception as e:
            logger.error(f"Error adding documents to hybrid retriever: {e}")
            return False

    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from both retrievers."""
        try:
            # Delete from both retrievers
            dense_success, sparse_success = await asyncio.gather(
                self.dense_retriever.delete_documents(document_ids),
                self.sparse_retriever.delete_documents(document_ids),
            )

            success = dense_success and sparse_success
            logger.info(f"Deleted {len(document_ids)} documents from hybrid retriever: {success}")
            return success

        except Exception as e:
            logger.error(f"Error deleting documents from hybrid retriever: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all documents from both retrievers."""
        try:
            # Clear both retrievers
            dense_success, sparse_success = await asyncio.gather(
                self.dense_retriever.clear(), self.sparse_retriever.clear()
            )

            success = dense_success and sparse_success
            logger.info(f"Cleared hybrid retriever: {success}")
            return success

        except Exception as e:
            logger.error(f"Error clearing hybrid retriever: {e}")
            return False
