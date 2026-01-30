"""
ChromaDB Vector Store - RAG-The-Game-Changer

Production-grade ChromaDB vector store implementation.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .base_store import BaseVectorStore, VectorStoreResult, VectorSearchResult

logger = logging.getLogger(__name__)


class ChromaDBStore(BaseVectorStore):
    """ChromaDB vector store implementation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.host = self.config.get("host", "localhost")
        self.port = self.config.get("port", 8000)
        self.collection_name = self.config.get("collection_name", "rag_documents")
        self.client = None
        self.collection = None

    async def initialize(self) -> bool:
        """Initialize ChromaDB connection."""
        try:
            import chromadb
            from chromadb.config import Settings

            chroma_settings = Settings(
                chroma_server_host=self.host, chroma_server_http_port=self.port
            )

            self.client = chromadb.Client(chroma_settings)

            try:
                self.collection = self.client.get_collection(name=self.collection_name)
            except Exception:
                self.collection = self.client.create_collection(name=self.collection_name)

            self._initialized = True
            logger.info(f"ChromaDB initialized: {self.collection_name}")
            return True

        except ImportError:
            logger.error("chromadb not installed. Install with: pip install chromadb")
            return False
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            return False

    async def add_vectors(
        self,
        vectors: Union[np.ndarray, List[np.ndarray]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> VectorStoreResult:
        """Add vectors to ChromaDB."""
        try:
            if not self._initialized:
                await self.initialize()

            if isinstance(vectors, np.ndarray):
                vectors = vectors.tolist()
            else:
                vectors = [v.tolist() if isinstance(v, np.ndarray) else v for v in vectors]

            self.collection.add(
                embeddings=vectors, ids=ids, metadatas=metadata or [{} for _ in ids]
            )

            logger.info(f"Added {len(ids)} vectors to ChromaDB")
            return VectorStoreResult(success=True, message=f"Added {len(ids)} vectors", ids=ids)

        except Exception as e:
            logger.error(f"Error adding vectors to ChromaDB: {e}")
            return VectorStoreResult(success=False, message=str(e))

    async def search(
        self, query_vector: np.ndarray, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> VectorSearchResult:
        """Search for similar vectors."""
        try:
            if not self._initialized:
                await self.initialize()

            start_time = asyncio.get_event_loop().time()

            query_embedding = (
                query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
            )

            where_clause = None
            if filters:
                where_clause = self._build_where_clause(filters)

            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=top_k, where=where_clause
            )

            search_time = (asyncio.get_event_loop().time() - start_time) * 1000

            ids = results["ids"][0] if results["ids"] else []
            distances = results["distances"][0] if results["distances"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []

            scores = [1.0 - float(d) for d in distances]

            return VectorSearchResult(
                ids=ids,
                scores=scores,
                metadata=metadatas,
                total_results=len(ids),
                search_time_ms=search_time,
            )

        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return VectorSearchResult(
                ids=[], scores=[], metadata=[], total_results=0, search_time_ms=0
            )

    def _build_where_clause(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build ChromaDB where clause from filters."""
        where_clause = {}
        for key, value in filters.items():
            if isinstance(value, dict):
                for op, op_value in value.items():
                    if op == "eq":
                        where_clause[key] = op_value
                    elif op == "in":
                        where_clause[key] = {"$in": op_value}
            else:
                where_clause[key] = value
        return where_clause if where_clause else None

    async def delete(self, ids: List[str]) -> VectorStoreResult:
        """Delete vectors by IDs."""
        try:
            if not self._initialized:
                await self.initialize()

            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} vectors from ChromaDB")
            return VectorStoreResult(success=True, message=f"Deleted {len(ids)} vectors", ids=ids)

        except Exception as e:
            logger.error(f"Error deleting from ChromaDB: {e}")
            return VectorStoreResult(success=False, message=str(e))

    async def update(
        self,
        vectors: Union[np.ndarray, List[np.ndarray]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> VectorStoreResult:
        """Update vectors by IDs."""
        try:
            await self.delete(ids)
            return await self.add_vectors(vectors, ids, metadata)
        except Exception as e:
            logger.error(f"Error updating ChromaDB vectors: {e}")
            return VectorStoreResult(success=False, message=str(e))

    async def clear(self) -> VectorStoreResult:
        """Clear all vectors."""
        try:
            if not self._initialized:
                await self.initialize()

            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)

            logger.info("Cleared ChromaDB collection")
            return VectorStoreResult(success=True, message="Collection cleared")

        except Exception as e:
            logger.error(f"Error clearing ChromaDB: {e}")
            return VectorStoreResult(success=False, message=str(e))

    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            if not self._initialized:
                await self.initialize()

            count = self.collection.count()
            return {
                "total_vectors": count,
                "collection_name": self.collection_name,
                "initialized": self._initialized,
                "host": self.host,
                "port": self.port,
            }
        except Exception as e:
            logger.error(f"Error getting ChromaDB stats: {e}")
            return {"total_vectors": 0, "error": str(e)}
