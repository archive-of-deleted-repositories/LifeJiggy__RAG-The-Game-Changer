"""Data Ingestion Module - RAG-The-Game-Changer

Production-grade data ingestion pipeline with loaders, preprocessors, and chunkers.
"""

# Import from loaders
from .loaders.base_classes import DocumentLoader, DocumentMetadata, LoadedDocument, LoaderError
from .loaders.pdf_loader import PDFLoader
from .loaders.web_loader import WebLoader
from .loaders.code_loader import CodeLoader
from .loaders.text_loader import TextLoader
from .loaders.database_loader import DatabaseLoader
from .loaders.api_loader import APILoader

# Import from preprocessors
from .preprocessors import (
    TextCleaner,
    MetadataExtractor,
    LanguageDetector,
    DuplicateDetector,
    QualityFilter,
    PreprocessingResult,
    BasePreprocessor,
)

# Import from chunkers
from .chunkers.document_chunker import (
    BaseChunker,
    TokenChunker,
    SemanticChunker,
    FixedSizeChunker,
    DocumentChunk,
    create_chunker,
)

__all__ = [
    # Loaders
    "DocumentLoader",
    "DocumentMetadata",
    "LoadedDocument",
    "LoaderError",
    "PDFLoader",
    "WebLoader",
    "CodeLoader",
    "DatabaseLoader",
    "APILoader",
    "TextLoader",
    # Preprocessors
    "TextCleaner",
    "MetadataExtractor",
    "LanguageDetector",
    "DuplicateDetector",
    "QualityFilter",
    "PreprocessingResult",
    "BasePreprocessor",
    # Chunkers
    "BaseChunker",
    "TokenChunker",
    "SemanticChunker",
    "FixedSizeChunker",
    "DocumentChunk",
    "create_chunker",
]
