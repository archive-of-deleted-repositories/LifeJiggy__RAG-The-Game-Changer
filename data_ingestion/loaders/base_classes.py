"""
Document Loader Base Classes - RAG-The-Game-Changer

Base classes and data structures for document loading.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import hashlib
import logging

logger = logging.getLogger(__name__)


class LoaderError(Exception):
    """Exception raised by document loaders."""

    def __init__(self, message: str, source: Optional[str] = None, details: Optional[Dict] = None):
        super().__init__(message)
        self.source = source
        self.details = details or {}


@dataclass
class DocumentMetadata:
    """Metadata for loaded documents."""

    source: str
    source_type: str
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[str] = None
    modified_date: Optional[str] = None
    file_size: Optional[int] = None
    file_extension: Optional[str] = None
    language: Optional[str] = None
    checksum: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "source_type": self.source_type,
            "title": self.title,
            "author": self.author,
            "created_date": self.created_date,
            "modified_date": self.modified_date,
            "file_size": self.file_size,
            "file_extension": self.file_extension,
            "language": self.language,
            "checksum": self.checksum,
            **self.extra,
        }


@dataclass
class LoadedDocument:
    """A loaded document with content and metadata."""

    content: str
    metadata: DocumentMetadata
    document_id: str
    chunks: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "chunks": len(self.chunks),
        }


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def can_load(self, source: Union[str, Path, Dict]) -> bool:
        """Check if this loader can handle the source."""
        pass

    @abstractmethod
    async def load(self, source: Union[str, Path, Dict]) -> List[LoadedDocument]:
        """Load documents from the source."""
        pass

    def _generate_document_id(self, content: str, source: str) -> str:
        """Generate a unique document ID."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        return f"doc_{content_hash}_{source_hash}"

    def _calculate_checksum(self, content: str) -> str:
        """Calculate checksum for content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _detect_language(self, content: str) -> str:
        """Simple language detection."""
        # Basic language detection based on character patterns
        if not content:
            return "unknown"

        # Check for common English indicators
        english_words = ["the", "and", "is", "in", "to", "of", "a", "that", "it", "with"]
        words = content.lower().split()[:100]  # Check first 100 words

        english_count = sum(1 for word in words if word in english_words)

        if len(words) > 0 and english_count / len(words) > 0.1:
            return "en"

        return "unknown"
