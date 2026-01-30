"""
Pinecone Vector Store - RAG-The-Game-Changer

Production-grade Pinecone vector store implementation.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .base_store import BaseVectorStore, VectorStoreResult, VectorSearchResult

logger = logging.getLogger(__name__)


class PineconeStore(BaseVectorStore):
    """Pinecone vector store implementation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api_key = self.config.get("api_key")
        self.environment = self.config.get("environment", "us-east1-gcp")
        self.index_name = self.config.get("index_name", "rag-index")
        self.namespace = self.config.get("namespace", "")
        self.client = None
        self.index = None

    async def initialize(self) -> bool:
        """Initialize Pinecone connection."""
        try:
            import pinecone

            pinecone.init(api_key=self.api_key, environment=self.environment)

            # Check if index exists
            if self.index_name not in pinecone.list_indexes():
                logger.error(f"Index {self.index_name} does not exist")
                return False

            self.index = pinecone.Index(self.index_name)
            self._initialized = True
            logger.info(f"Pinecone initialized: {self.index_name}")
            return True

        except ImportError:
            logger.error("pinecone-client not installed. Install with: pip install pinecone-client")
            return False
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            return False

    async def add_vectors(
        self,
        vectors: Union[np.ndarray, List[np.ndarray]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> VectorStoreResult:
        """Add vectors to Pinecone."""
        try:
            if not self._initialized:
                await self.initialize()

            # Prepare vectors for upload
            if isinstance(vectors, np.ndarray):
                vectors = vectors.tolist()
            else:
                vectors = [v.tolist() if isinstance(v, np.ndarray) else v for v in vectors]

            # Create tuples (id, vector, metadata)
            to_upsert = []
            for i, (id, vec) in enumerate(zip(ids, vectors)):
                meta = metadata[i] if metadata and i < len(metadata) else {}
                to_upsert.append((id, vec, meta))

            # Upsert in batches
            batch_size = 100
            for i in range(0, len(to_upsert), batch_size):
                batch = to_upsert[i : i + batch_size]
                self.index.upsert(vectors=batch, namespace=self.namespace)

            logger.info(f"Added {len(ids)} vectors to Pinecone")
            return VectorStoreResult(success=True, message=f"Added {len(ids)} vectors", ids=ids)

        except Exception as e:
            logger.error(f"Error adding vectors to Pinecone: {e}")
            return VectorStoreResult(success=False, message=str(e))

    async def search(
        self, query_vector: np.ndarray, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> VectorSearchResult:
        """Search for similar vectors."""
        try:
            if not self._initialized:
                await self.initialize()

            start_time = asyncio.get_event_loop().time()

            # Convert query vector
            query_vector = (
                query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
            )

            # Build filter
            filter_dict = self._build_filter(filters) if filters else None

            # Query
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=self.namespace,
                filter=filter_dict,
                include_metadata=True,
            )

            search_time = (asyncio.get_event_loop().time() - start_time) * 1000

            # Extract results
            matches = results.get("matches", [])
            ids = [match["id"] for match in matches]
            scores = [match["score"] for match in matches]
            metadatas = [match.get("metadata", {}) for match in matches]

            return VectorSearchResult(
                ids=ids,
                scores=scores,
                metadata=metadatas,
                total_results=len(ids),
                search_time_ms=search_time,
            )

        except Exception as e:
            logger.error(f"Error searching Pinecone: {e}")
            return VectorSearchResult(
                ids=[], scores=[], metadata=[], total_results=0, search_time_ms=0
            )

    def _build_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build Pinecone filter from filters."""
        pinecone_filter = {}
        for key, value in filters.items():
            if isinstance(value, dict):
                for op, op_value in value.items():
                    if op == "eq":
                        pinecone_filter[key] = {"$eq": op_value}
                    elif op == "in":
                        pinecone_filter[key] = {"$in": op_value}
            else:
                pinecone_filter[key] = value
        return pinecone_filter

    async def delete(self, ids: List[str]) -> VectorStoreResult:
        """Delete vectors by IDs."""
        try:
            if not self._initialized:
                await self.initialize()

            self.index.delete(ids=ids, namespace=self.namespace)
            logger.info(f"Deleted {len(ids)} vectors from Pinecone")
            return VectorStoreResult(success=True, message=f"Deleted {len(ids)} vectors", ids=ids)

        except Exception as e:
            logger.error(f"Error deleting from Pinecone: {e}")
            return VectorStoreResult(success=False, message=str(e))

    async def update(
        self,
        vectors: Union[np.ndarray, List[np.ndarray]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> VectorStoreResult:
        """Update vectors by IDs."""
        try:
            # Pinecone upsert handles updates
            return await self.add_vectors(vectors, ids, metadata)
        except Exception as e:
            logger.error(f"Error updating Pinecone vectors: {e}")
            return VectorStoreResult(success=False, message=str(e))

    async def clear(self) -> VectorStoreResult:
        """Clear all vectors in namespace."""
        try:
            if not self._initialized:
                await self.initialize()

            self.index.delete(delete_all=True, namespace=self.namespace)
            logger.info("Cleared Pinecone namespace")
            return VectorStoreResult(success=True, message="Namespace cleared")

        except Exception as e:
            logger.error(f"Error clearing Pinecone: {e}")
            return VectorStoreResult(success=False, message=str(e))

    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            if not self._initialized:
                await self.initialize()

            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", 0),
                "index_name": self.index_name,
                "namespace": self.namespace,
                "initialized": self._initialized,
            }
        except Exception as e:
            logger.error(f"Error getting Pinecone stats: {e}")
            return {"total_vectors": 0, "error": str(e)}
