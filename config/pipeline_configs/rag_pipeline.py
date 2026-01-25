from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""

    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    query: str
    response_id: str
    timestamp: str
    generation_time_ms: float
    retrieval_time_ms: float
    total_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "sources": self.sources,
            "retrieved_chunks": self.retrieved_chunks,
            "query": self.query,
            "response_id": self.response_id,
            "timestamp": self.timestamp,
            "generation_time_ms": self.generation_time_ms,
            "retrieval_time_ms": self.retrieval_time_ms,
            "total_time_ms": self.total_time_ms,
            "metadata": self.metadata,
        }


class RAGPipeline:
    """Main RAG Pipeline for Retrieval-Augmented Generation."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        retrieval_strategy: str = "hybrid",
        embedding_provider: str = "openai",
        llm_provider: str = "openai",
        vector_db: str = "pinecone",
    ):
        self.config = config or {}
        self.retrieval_strategy = retrieval_strategy
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.vector_db = vector_db

        self._initialize_components()

    def _initialize_components(self):
        """Initialize the RAG pipeline components."""
        try:
            self._initialize_retriever()
            self._initialize_generator()
            self._initialize_embedder()
            logger.info("RAG Pipeline components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG Pipeline: {e}")
            raise

    def _initialize_retriever(self):
        """Initialize the retriever component."""
        if self.retrieval_strategy == "dense":
            from retrieval_systems.dense_retriever import DenseRetriever

            self.retriever = DenseRetriever(self.config.get("retrieval", {}))
        elif self.retrieval_strategy == "sparse":
            from retrieval_systems.sparse_retriever import SparseRetriever

            self.retriever = SparseRetriever(self.config.get("retrieval", {}))
        else:
            from retrieval_systems.hybrid_retriever import HybridRetriever

            self.retriever = HybridRetriever(self.config.get("retrieval", {}))

    def _initialize_generator(self):
        """Initialize the generator component."""
        from generation_components import GroundedGenerator

        self.generator = GroundedGenerator(self.config.get("generation", {}))

    def _initialize_embedder(self):
        """Initialize the embedding component."""
        if self.embedding_provider == "openai":
            try:
                from openai import OpenAI

                self.embedder = OpenAI()
            except ImportError:
                logger.warning("OpenAI not available, using fallback embedder")
                self.embedder = None
        else:
            try:
                from sentence_transformers import SentenceTransformer

                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                logger.warning("Sentence transformers not available")
                self.embedder = None

    async def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        include_sources: bool = True,
        include_confidence: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RAGResponse:
        """Process a query through the RAG pipeline."""
        start_time = time.time()
        response_id = str(uuid.uuid4())[:8]

        logger.info(f"Processing query: {query[:100]}...")

        try:
            retrieval_start = time.time()
            retrieval_result = await self.retriever.retrieve(
                query=query,
                top_k=top_k or 5,
                filters=filters,
            )
            retrieval_time = (time.time() - retrieval_start) * 1000

            generation_start = time.time()
            response = await self.generator.generate(
                query=query,
                retrieved_chunks=retrieval_result.chunks,
            )
            generation_time = (time.time() - generation_start) * 1000

            total_time = (time.time() - start_time) * 1000

            rag_response = RAGResponse(
                answer=response.answer,
                confidence=response.confidence if include_confidence else 0.0,
                sources=response.sources if include_sources else [],
                retrieved_chunks=[
                    {
                        "content": chunk.content,
                        "score": chunk.score,
                        "metadata": chunk.metadata,
                    }
                    for chunk in retrieval_result.chunks
                ],
                query=query,
                response_id=response_id,
                timestamp=datetime.utcnow().isoformat(),
                generation_time_ms=generation_time,
                retrieval_time_ms=retrieval_time,
                total_time_ms=total_time,
                metadata={
                    "retrieval_strategy": self.retrieval_strategy,
                    "chunks_retrieved": len(retrieval_result.chunks),
                },
            )

            logger.info(f"Query processed in {total_time:.2f}ms")
            return rag_response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    async def ingest(
        self, documents: List[Dict[str, Any]], chunk_strategy: str = "semantic", **kwargs
    ) -> Dict[str, Any]:
        """Ingest documents into the RAG pipeline."""
        logger.info(f"Ingesting {len(documents)} documents")

        try:
            results = {
                "total_documents": len(documents),
                "successful": 0,
                "failed": 0,
                "total_chunks": 0,
            }

            for doc in documents:
                try:
                    chunks = await self._chunk_document(doc, chunk_strategy)
                    await self._index_chunks(chunks)
                    results["successful"] += 1
                    results["total_chunks"] += len(chunks)
                except Exception as e:
                    logger.error(f"Error ingesting document: {e}")
                    results["failed"] += 1

            logger.info(
                f"Ingestion complete: {results['successful']}/{results['total_documents']} documents"
            )
            return results

        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            raise

    async def _chunk_document(
        self, document: Dict[str, Any], strategy: str
    ) -> List[Dict[str, Any]]:
        """Chunk a document into smaller pieces."""
        from data_ingestion.chunkers.document_chunker import create_chunker

        content = document.get("content", "")
        metadata = document.get("metadata", {})
        document_id = document.get("document_id", "unknown")

        chunker = create_chunker(strategy)
        chunks = await chunker.chunk(content, metadata, document_id)

        # Convert chunks to dict format
        return [
            {
                "content": chunk.content,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "metadata": chunk.metadata,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]

    async def _index_chunks(self, chunks: List[Dict[str, Any]]):
        """Index chunks in the vector database."""
        pass

    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the index."""
        try:
            await self.retriever.delete_documents(document_ids)
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    async def clear_index(self) -> bool:
        """Clear all documents from the index."""
        try:
            await self.retriever.clear()
            return True
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "retrieval_strategy": self.retrieval_strategy,
            "embedding_provider": self.embedding_provider,
            "llm_provider": self.llm_provider,
            "vector_db": self.vector_db,
            "components_initialized": True,
        }
