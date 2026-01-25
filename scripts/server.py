"""
RAG FastAPI Server - RAG-The-Game-Changer

REST API server for the RAG system.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
import time

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG-The-Game-Changer API",
    description="Production-Ready Retrieval-Augmented Generation System",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
rag_pipeline = None


# Request/Response Models
class DocumentRequest(BaseModel):
    content: str = Field(..., description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")


class IngestRequest(BaseModel):
    documents: List[DocumentRequest] = Field(..., description="Documents to ingest")
    chunk_strategy: str = Field(default="semantic", description="Chunking strategy")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Query string")
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    include_sources: bool = Field(default=True, description="Include source information")
    include_confidence: bool = Field(default=True, description="Include confidence scores")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Query filters")


class HealthResponse(BaseModel):
    status: str
    pipeline_initialized: bool
    components: Dict[str, str]


class StatsResponse(BaseModel):
    pipeline_stats: Dict[str, Any]
    health_check: Dict[str, Any]


# Initialize pipeline on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on server startup."""
    global rag_pipeline
    try:
        from config import RAGPipeline

        rag_pipeline = RAGPipeline()
        await rag_pipeline.initialize()
        logger.info("RAG Pipeline initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {e}")
        # Continue with pipeline as None, will handle in endpoints


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "RAG-The-Game-Changer API", "version": "0.1.0", "status": "running"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        if rag_pipeline:
            health = await rag_pipeline.health_check()
            return HealthResponse(
                status=health.get("status", "unknown"),
                pipeline_initialized=True,
                components=health.get("components", {}),
            )
        else:
            return HealthResponse(
                status="degraded",
                pipeline_initialized=False,
                components={"pipeline": "not_initialized"},
            )

    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get pipeline statistics."""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        stats = await rag_pipeline.get_stats()
        health = await rag_pipeline.health_check()

        return StatsResponse(pipeline_stats=stats, health_check=health)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get stats")


@app.post("/api/v1/ingest")
async def ingest_documents(request: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest documents into the RAG system."""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        # Convert request format
        documents = []
        for doc in request.documents:
            documents.append(
                {
                    "content": doc.content,
                    "metadata": doc.metadata or {},
                    "document_id": f"doc_{int(time.time() * 1000)}_{len(documents)}",
                }
            )

        # Ingest documents
        result = await rag_pipeline.ingest(
            documents=documents, chunk_strategy=request.chunk_strategy
        )

        return {"status": "success", "message": "Documents ingested successfully", "result": result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/v1/query")
async def query_documents(request: QueryRequest):
    """Query the RAG system."""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        # Execute query
        response = await rag_pipeline.query(
            query=request.query,
            top_k=request.top_k,
            include_sources=request.include_sources,
            include_confidence=request.include_confidence,
            filters=request.filters,
        )

        return {
            "status": "success",
            "query": response.query,
            "answer": response.answer,
            "confidence": response.confidence,
            "sources": response.sources,
            "metadata": response.metadata,
            "timing": {
                "total_time_ms": response.total_time_ms,
                "retrieval_time_ms": response.retrieval_time_ms,
                "generation_time_ms": response.generation_time_ms,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.delete("/api/v1/documents")
async def delete_documents(document_ids: List[str]):
    """Delete documents from the RAG system."""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        success = await rag_pipeline.delete_documents(document_ids)

        if success:
            return {
                "status": "success",
                "message": f"Deleted {len(document_ids)} documents",
                "deleted_ids": document_ids,
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete documents")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@app.delete("/api/v1/clear")
async def clear_index():
    """Clear all documents from the RAG system."""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        success = await rag_pipeline.clear_index()

        if success:
            return {"status": "success", "message": "Index cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear index")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")


@app.get("/api/v1/config")
async def get_config():
    """Get current configuration."""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        stats = await rag_pipeline.get_stats()

        return {
            "status": "success",
            "config": {
                "retrieval_strategy": stats.get("retrieval_strategy"),
                "embedding_provider": stats.get("embedding_provider"),
                "llm_provider": stats.get("llm_provider"),
                "vector_db": stats.get("vector_db"),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Config error: {e}")
        raise HTTPException(status_code=500, detail=f"Config retrieval failed: {str(e)}")


# Run server
def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run("scripts.server:app", host=host, port=port, reload=reload, log_level="info")


if __name__ == "__main__":
    run_server()
