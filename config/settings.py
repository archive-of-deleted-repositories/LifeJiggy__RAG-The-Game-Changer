from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class VectorStoreConfig:
    provider: str = "pinecone"
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index: str = "rag-index"
    weaviate_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    faiss_index_path: str = "./data/faiss/index.faiss"


@dataclass
class EmbeddingConfig:
    provider: str = "openai"
    openai_api_key: Optional[str] = None
    openai_model: str = "text-embedding-3-small"
    openai_dimensions: int = 1536
    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sentence_transformer_device: str = "cpu"


@dataclass
class LLMConfig:
    provider: str = "openai"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4-turbo-preview"
    openai_temperature: float = 0.1
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-sonnet-20240229"


@dataclass
class RetrievalConfig:
    default_strategy: str = "hybrid"
    top_k: int = 5
    max_top_k: int = 20
    rerank_enabled: bool = True
    rerank_model: str = "ms-marco-MiniLM-l12-h384-uncased"
    dense_weight: float = 0.7
    sparse_weight: float = 0.3


@dataclass
class ChunkingConfig:
    strategy: str = "semantic"
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class GenerationConfig:
    max_context_tokens: int = 8000
    min_confidence: float = 0.7
    citation_enabled: bool = True
    citation_style: str = "apa"


@dataclass
class Settings:
    app_name: str = "RAG-The-Game-Changer"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    cache_enabled: bool = True
    cache_ttl: int = 3600
    
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        settings = cls()
        
        if "app_name" in data:
            settings.app_name = data["app_name"]
        if "app_version" in data:
            settings.app_version = data["app_version"]
        if "environment" in data:
            settings.environment = data["environment"]
        if "debug" in data:
            settings.debug = data["debug"]
        if "log_level" in data:
            settings.log_level = data["log_level"]
        
        if "api" in data:
            api = data["api"]
            if "host" in api:
                settings.api_host = api["host"]
            if "port" in api:
                settings.api_port = api["port"]
        
        if "vector_store" in data:
            vs = data["vector_store"]
            settings.vector_store = VectorStoreConfig(
                provider=vs.get("provider", "pinecone"),
                pinecone_api_key=vs.get("pinecone_api_key") or os.getenv("PINECONE_API_KEY"),
                pinecone_environment=vs.get("pinecone_environment") or os.getenv("PINECONE_ENVIRONMENT"),
                pinecone_index=vs.get("pinecone_index", "rag-index"),
                weaviate_url=vs.get("weaviate_url") or os.getenv("WEAVIATE_URL"),
                chroma_host=vs.get("chroma_host", "localhost"),
                chroma_port=vs.get("chroma_port", 8000),
                qdrant_url=vs.get("qdrant_url") or os.getenv("QDRANT_URL"),
            )
        
        if "embedding" in data:
            emb = data["embedding"]
            settings.embedding = EmbeddingConfig(
                provider=emb.get("provider", "openai"),
                openai_api_key=emb.get("openai_api_key") or os.getenv("OPENAI_API_KEY"),
                openai_model=emb.get("openai_model", "text-embedding-3-small"),
                openai_dimensions=emb.get("openai_dimensions", 1536),
                sentence_transformer_model=emb.get("sentence_transformer_model", "sentence-transformers/all-MiniLM-L6-v2"),
                sentence_transformer_device=emb.get("sentence_transformer_device", "cpu"),
            )
        
        if "llm" in data:
            llm = data["llm"]
            settings.llm = LLMConfig(
                provider=llm.get("provider", "openai"),
                openai_api_key=llm.get("openai_api_key") or os.getenv("OPENAI_API_KEY"),
                openai_model=llm.get("openai_model", "gpt-4-turbo-preview"),
                openai_temperature=llm.get("openai_temperature", 0.1),
                anthropic_api_key=llm.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY"),
                anthropic_model=llm.get("anthropic_model", "claude-3-sonnet-20240229"),
            )
        
        if "retrieval" in data:
            ret = data["retrieval"]
            settings.retrieval = RetrievalConfig(
                default_strategy=ret.get("default_strategy", "hybrid"),
                top_k=ret.get("top_k", 5),
                max_top_k=ret.get("max_top_k", 20),
                rerank_enabled=ret.get("rerank_enabled", True),
                rerank_model=ret.get("rerank_model", "ms-marco-MiniLM-l12-h384-uncased"),
                dense_weight=ret.get("dense_weight", 0.7),
                sparse_weight=ret.get("sparse_weight", 0.3),
            )
        
        if "chunking" in data:
            chunk = data["chunking"]
            settings.chunking = ChunkingConfig(
                strategy=chunk.get("strategy", "semantic"),
                chunk_size=chunk.get("chunk_size", 1000),
                chunk_overlap=chunk.get("chunk_overlap", 200),
            )
        
        if "generation" in data:
            gen = data["generation"]
            settings.generation = GenerationConfig(
                max_context_tokens=gen.get("max_context_tokens", 8000),
                min_confidence=gen.get("min_confidence", 0.7),
                citation_enabled=gen.get("citation_enabled", True),
                citation_style=gen.get("citation_style", "apa"),
            )
        
        return settings


def load_config(config_path: Optional[str] = None) -> Settings:
    if config_path is None:
        config_path = os.getenv("RAG_CONFIG_PATH", "config/pipeline_configs/main_pipeline.yaml")
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        return Settings()
    
    with open(config_file, "r") as f:
        data = yaml.safe_load(f)
    
    if data is None:
        return Settings()
    
    return Settings.from_dict(data)
