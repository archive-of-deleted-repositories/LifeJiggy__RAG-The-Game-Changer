from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from .settings import (
    VectorStoreConfig, EmbeddingConfig, LLMConfig, 
    RetrievalConfig, ChunkingConfig, GenerationConfig
)


class PipelineType(Enum):
    INGESTION = "ingestion"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    FULL = "full"


@dataclass
class PipelineConfig:
    name: str = "main_pipeline"
    version: str = "1.0.0"
    pipeline_type: PipelineType = PipelineType.FULL
    
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    enabled: bool = True
    batch_size: int = 32
    max_concurrent: int = 4
    timeout: int = 300
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "pipeline_type": self.pipeline_type.value,
            "vector_store": {
                "provider": self.vector_store.provider,
                "pinecone_api_key": "***" if self.vector_store.pinecone_api_key else None,
                "pinecone_environment": self.vector_store.pinecone_environment,
                "pinecone_index": self.vector_store.pinecone_index,
            },
            "embedding": {
                "provider": self.embedding.provider,
                "openai_model": self.embedding.openai_model,
                "openai_dimensions": self.embedding.openai_dimensions,
            },
            "llm": {
                "provider": self.llm.provider,
                "openai_model": self.llm.openai_model,
                "openai_temperature": self.llm.openai_temperature,
            },
            "retrieval": {
                "default_strategy": self.retrieval.default_strategy,
                "top_k": self.retrieval.top_k,
                "rerank_enabled": self.retrieval.rerank_enabled,
            },
            "chunking": {
                "strategy": self.chunking.strategy,
                "chunk_size": self.chunking.chunk_size,
                "chunk_overlap": self.chunking.chunk_overlap,
            },
            "generation": {
                "max_context_tokens": self.generation.max_context_tokens,
                "min_confidence": self.generation.min_confidence,
                "citation_enabled": self.generation.citation_enabled,
            },
            "enabled": self.enabled,
            "batch_size": self.batch_size,
        }


@dataclass
class RAGConfig:
    name: str = "RAG-The-Game-Changer"
    version: str = "0.1.0"
    environment: str = "development"
    
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    cache_enabled: bool = True
    cache_ttl: int = 3600
    
    def __post_init__(self):
        if self.environment == "production":
            self.pipeline.retrieval.top_k = min(self.pipeline.retrieval.top_k, 10)
            self.pipeline.generation.min_confidence = max(
                self.pipeline.generation.min_confidence, 0.8
            )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "environment": self.environment,
            "pipeline": self.pipeline.to_dict(),
            "metrics_enabled": self.metrics_enabled,
            "tracing_enabled": self.tracing_enabled,
            "cache_enabled": self.cache_enabled,
            "cache_ttl": self.cache_ttl,
        }
