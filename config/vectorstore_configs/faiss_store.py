"""
FAISS Vector Store - RAG-The-Game-Changer

FAISS-based vector storage implementation for local development.
"""

import asyncio
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import logging

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from .base_store import BaseVectorStore, VectorStoreResult, VectorSearchResult

logger = logging.getLogger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store for local development."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not installed. Install with: pip install faiss-cpu")

        self.index_path = self.config.get("index_path", "./data/faiss/index.faiss")
        self.metadata_path = self.config.get("metadata_path", "./data/faiss/metadata.pkl")
        self.dimension = self.config.get("dimension", 384)

        self._index = None
        self._id_to_index = {}
        self._index_to_id = {}
        self._metadata = {}
        self._next_index = 0

    async def initialize(self) -> bool:
        """Initialize FAISS index."""
        try:
            # Create directory if it doesn't exist
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)

            # Try to load existing index
            if Path(self.index_path).exists() and Path(self.metadata_path).exists():
                await self._load_index()
                logger.info(f"Loaded existing FAISS index from {self.index_path}")
            else:
                # Create new index
                self._index = faiss.IndexFlatL2(self.dimension)
                self._initialized = True
                logger.info("Created new FAISS index")

            return True

        except Exception as e:
            logger.error(f"Error initializing FAISS vector store: {e}")
            return False

    async def add_vectors(
        self,
        vectors: Union[np.ndarray, List[np.ndarray]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> VectorStoreResult:
        """Add vectors to FAISS index."""
        try:
            if not self._initialized:
                await self.initialize()

            # Convert to numpy array
            if isinstance(vectors, list):
                vectors = np.array(vectors)

            if len(vectors) != len(ids):
                return VectorStoreResult(
                    success=False, message="Number of vectors and IDs must match"
                )

            # Add to FAISS index
            self._index.add(vectors.astype(np.float32))

            # Store IDs and metadata
            for i, (vec_id, vec) in enumerate(zip(ids, vectors)):
                idx = self._next_index
                self._id_to_index[vec_id] = idx
                self._index_to_id[idx] = vec_id

                if metadata and i < len(metadata):
                    self._metadata[vec_id] = metadata[i]
                else:
                    self._metadata[vec_id] = {}

                self._next_index += 1

            # Save to disk
            await self._save_index()

            return VectorStoreResult(
                success=True, message=f"Added {len(ids)} vectors to FAISS index", ids=ids
            )

        except Exception as e:
            logger.error(f"Error adding vectors to FAISS: {e}")
            return VectorStoreResult(success=False, message=f"Error adding vectors: {str(e)}")

    async def search(
        self, query_vector: np.ndarray, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> VectorSearchResult:
        """Search for similar vectors."""
        try:
            if not self._initialized:
                await self.initialize()

            if self._index.ntotal == 0:
                return VectorSearchResult(
                    ids=[], scores=[], metadata=[], total_results=0, search_time_ms=0.0
                )

            # Ensure query_vector is 2D
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)

            # Search
            import time

            start_time = time.time()
            scores, indices = self._index.search(
                query_vector.astype(np.float32), min(top_k, self._index.ntotal)
            )
            search_time = (time.time() - start_time) * 1000

            # Convert results
            result_ids = []
            result_scores = []
            result_metadata = []

            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx in self._index_to_id:
                    vec_id = self._index_to_id[idx]

                    # Apply filters if provided
                    if filters:
                        meta = self._metadata.get(vec_id, {})
                        if not self._match_filters(meta, filters):
                            continue

                    result_ids.append(vec_id)
                    result_scores.append(float(score))
                    result_metadata.append(self._metadata.get(vec_id, {}))

            return VectorSearchResult(
                ids=result_ids,
                scores=result_scores,
                metadata=result_metadata,
                total_results=len(result_ids),
                search_time_ms=search_time,
            )

        except Exception as e:
            logger.error(f"Error searching FAISS index: {e}")
            return VectorSearchResult(
                ids=[], scores=[], metadata=[], total_results=0, search_time_ms=0.0
            )

    async def delete(self, ids: List[str]) -> VectorStoreResult:
        """Delete vectors from FAISS index."""
        try:
            if not self._initialized:
                await self.initialize()

            # FAISS doesn't support deletion, so we need to rebuild
            indices_to_remove = []
            for vec_id in ids:
                if vec_id in self._id_to_index:
                    indices_to_remove.append(self._id_to_index[vec_id])

            if not indices_to_remove:
                return VectorStoreResult(success=True, message="No vectors found to delete", ids=[])

            # Get all vectors except those to remove
            all_vectors = []
            all_ids = []
            all_metadata = []

            for idx in range(self._index.ntotal):
                if idx not in indices_to_remove and idx in self._index_to_id:
                    vec_id = self._index_to_id[idx]
                    if vec_id not in ids:  # Double check
                        # Retrieve vector (this is inefficient in FAISS)
                        vector = self._index.reconstruct(idx)
                        all_vectors.append(vector)
                        all_ids.append(vec_id)
                        all_metadata.append(self._metadata.get(vec_id, {}))

            # Rebuild index
            self._index = faiss.IndexFlatL2(self.dimension)
            self._id_to_index.clear()
            self._index_to_id.clear()
            self._metadata.clear()
            self._next_index = 0

            if all_vectors:
                await self.add_vectors(all_vectors, all_ids, all_metadata)

            return VectorStoreResult(
                success=True, message=f"Deleted {len(ids)} vectors from FAISS index", ids=ids
            )

        except Exception as e:
            logger.error(f"Error deleting vectors from FAISS: {e}")
            return VectorStoreResult(success=False, message=f"Error deleting vectors: {str(e)}")

    async def update(
        self,
        vectors: Union[np.ndarray, List[np.ndarray]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> VectorStoreResult:
        """Update vectors in FAISS index."""
        # Delete old vectors and add new ones
        await self.delete(ids)
        return await self.add_vectors(vectors, ids, metadata)

    async def clear(self) -> VectorStoreResult:
        """Clear all vectors from FAISS index."""
        try:
            self._index = faiss.IndexFlatL2(self.dimension)
            self._id_to_index.clear()
            self._index_to_id.clear()
            self._metadata.clear()
            self._next_index = 0

            # Delete files
            if Path(self.index_path).exists():
                os.remove(self.index_path)
            if Path(self.metadata_path).exists():
                os.remove(self.metadata_path)

            return VectorStoreResult(success=True, message="Cleared FAISS index")

        except Exception as e:
            logger.error(f"Error clearing FAISS index: {e}")
            return VectorStoreResult(success=False, message=f"Error clearing index: {str(e)}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get FAISS index statistics."""
        if not self._initialized:
            await self.initialize()

        return {
            "total_vectors": self._index.ntotal if self._index else 0,
            "dimension": self.dimension,
            "index_type": "IndexFlatL2",
            "storage_path": self.index_path,
        }

    async def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self._index, self.index_path)

            # Save metadata
            metadata_data = {
                "id_to_index": self._id_to_index,
                "index_to_id": self._index_to_id,
                "metadata": self._metadata,
                "next_index": self._next_index,
            }

            with open(self.metadata_path, "wb") as f:
                pickle.dump(metadata_data, f)

        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")

    async def _load_index(self):
        """Load FAISS index and metadata from disk."""
        try:
            # Load FAISS index
            self._index = faiss.read_index(self.index_path)

            # Load metadata
            with open(self.metadata_path, "rb") as f:
                metadata_data = pickle.load(f)

            self._id_to_index = metadata_data.get("id_to_index", {})
            self._index_to_id = metadata_data.get("index_to_id", {})
            self._metadata = metadata_data.get("metadata", {})
            self._next_index = metadata_data.get("next_index", 0)

            self._initialized = True

        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            raise

    def _match_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
