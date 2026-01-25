"""
Sparse Retriever - RAG-The-Game-Changer

Sparse retrieval using keyword-based search (BM25).
"""

import asyncio
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional
import logging
import re

from .base import BaseRetriever, RetrievalResult, RetrievedChunk, RetrievalStrategy

logger = logging.getLogger(__name__)


class SparseRetriever(BaseRetriever):
    """Sparse retriever using BM25 algorithm."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.top_k = self.config.get("top_k", 5)
        self.k1 = self.config.get("k1", 1.2)  # BM25 parameter
        self.b = self.config.get("b", 0.75)  # BM25 parameter

        # Document storage
        self._documents: List[Dict[str, Any]] = []
        self._doc_lengths: List[float] = []
        self._vocab: Dict[str, int] = {}
        self._doc_freqs: List[Dict[str, int]] = []
        self._avg_doc_length: float = 0.0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    async def retrieve(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Retrieve documents using sparse BM25 search."""
        start_time = asyncio.get_event_loop().time()

        try:
            if not self._documents:
                return RetrievalResult(
                    query=query,
                    chunks=[],
                    strategy=RetrievalStrategy.SPARSE,
                    total_chunks=0,
                    retrieval_time_ms=0,
                )

            # Tokenize query
            query_tokens = self._tokenize(query)
            if not query_tokens:
                return RetrievalResult(
                    query=query,
                    chunks=[],
                    strategy=RetrievalStrategy.SPARSE,
                    total_chunks=0,
                    retrieval_time_ms=0,
                )

            # Calculate BM25 scores
            scores = self._calculate_bm25_scores(query_tokens)

            # Get top-k results
            top_k = min(top_k or self.top_k, len(scores))
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

            chunks = []
            for idx in top_indices:
                if scores[idx] > 0:
                    doc = self._documents[idx]
                    chunk = RetrievedChunk(
                        content=doc.get("content", ""),
                        document_id=doc.get("document_id", f"doc_{idx}"),
                        chunk_id=doc.get("chunk_id", f"chunk_{idx}"),
                        score=scores[idx],
                        metadata=doc.get("metadata", {}),
                        source=doc.get("source"),
                        title=doc.get("title"),
                    )
                    chunks.append(chunk)

            retrieval_time = (asyncio.get_event_loop().time() - start_time) * 1000

            return RetrievalResult(
                query=query,
                chunks=chunks,
                strategy=RetrievalStrategy.SPARSE,
                total_chunks=len(chunks),
                retrieval_time_ms=retrieval_time,
            )

        except Exception as e:
            logger.error(f"Error in sparse retrieval: {e}")
            return RetrievalResult(
                query=query,
                chunks=[],
                strategy=RetrievalStrategy.SPARSE,
                total_chunks=0,
                retrieval_time_ms=0,
            )

    def _calculate_bm25_scores(self, query_tokens: List[str]) -> List[float]:
        """Calculate BM25 scores for all documents."""
        scores = []

        for doc_idx in range(len(self._documents)):
            doc_score = 0.0
            doc_freqs = self._doc_freqs[doc_idx]
            doc_length = self._doc_lengths[doc_idx]

            for token in query_tokens:
                if token in doc_freqs and token in self._vocab:
                    # BM25 formula
                    df = self._vocab[token]  # Document frequency
                    tf = doc_freqs[token]  # Term frequency in this doc

                    idf = math.log((len(self._documents) - df + 0.5) / (df + 0.5))

                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (
                        1 - self.b + self.b * (doc_length / self._avg_doc_length)
                    )

                    doc_score += idf * (numerator / denominator)

            scores.append(doc_score)

        return scores

    async def add_documents(self, documents: List[Dict[str, Any]], **kwargs) -> bool:
        """Add documents to the sparse retriever."""
        try:
            new_doc_freqs = []
            new_doc_lengths = []

            for doc in documents:
                content = doc.get("content", "")
                tokens = self._tokenize(content)

                # Count term frequencies
                term_counts = Counter(tokens)
                new_doc_freqs.append(dict(term_counts))
                new_doc_lengths.append(len(tokens))

                # Update vocabulary
                for token in term_counts:
                    self._vocab[token] = self._vocab.get(token, 0) + 1

                # Store document
                self._documents.append(doc)
                self._doc_freqs.append(dict(term_counts))
                self._doc_lengths.append(len(tokens))

            # Update average document length
            if self._doc_lengths:
                self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths)

            logger.info(f"Added {len(documents)} documents to sparse retriever")
            return True

        except Exception as e:
            logger.error(f"Error adding documents to sparse retriever: {e}")
            return False

    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the sparse retriever."""
        try:
            indices_to_remove = []
            for i, doc in enumerate(self._documents):
                if doc.get("document_id") in document_ids:
                    indices_to_remove.append(i)

            # Remove in reverse order to maintain indices
            for idx in reversed(indices_to_remove):
                # Update vocabulary
                doc_freqs = self._doc_freqs[idx]
                for token in doc_freqs:
                    self._vocab[token] -= 1
                    if self._vocab[token] <= 0:
                        del self._vocab[token]

                del self._documents[idx]
                del self._doc_freqs[idx]
                del self._doc_lengths[idx]

            # Update average document length
            if self._doc_lengths:
                self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths)

            logger.info(f"Deleted {len(indices_to_remove)} documents from sparse retriever")
            return True

        except Exception as e:
            logger.error(f"Error deleting documents from sparse retriever: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all documents from the sparse retriever."""
        try:
            self._documents.clear()
            self._doc_lengths.clear()
            self._vocab.clear()
            self._doc_freqs.clear()
            self._avg_doc_length = 0.0
            logger.info("Cleared all documents from sparse retriever")
            return True

        except Exception as e:
            logger.error(f"Error clearing sparse retriever: {e}")
            return False
