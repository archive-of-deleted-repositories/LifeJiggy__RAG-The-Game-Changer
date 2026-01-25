"""
Document Chunking - RAG-The-Game-Changer

Text chunking strategies for document processing.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A chunk of a document."""

    content: str
    chunk_id: str
    document_id: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_char: Optional[int] = None
    end_char: Optional[int] = None


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    async def chunk(
        self, content: str, metadata: Dict[str, Any], document_id: Optional[str] = None
    ) -> List[DocumentChunk]:
        """Chunk content into smaller pieces."""
        pass


class TokenChunker(BaseChunker):
    """Token-based chunker that splits by token count."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 200)
        self.encoding_name = self.config.get("encoding", "cl100k_base")  # tiktoken encoding

    async def chunk(
        self, content: str, metadata: Dict[str, Any], document_id: Optional[str] = None
    ) -> List[DocumentChunk]:
        """Chunk content by token count."""
        try:
            # Simple tokenization (will be enhanced with tiktoken if available)
            tokens = self._tokenize(content)

            if len(tokens) <= self.chunk_size:
                # Single chunk
                return [
                    DocumentChunk(
                        content=content,
                        chunk_id=f"{document_id}_chunk_0",
                        document_id=document_id or "unknown",
                        chunk_index=0,
                        metadata=metadata.copy(),
                    )
                ]

            chunks = []
            start_idx = 0

            while start_idx < len(tokens):
                end_idx = min(start_idx + self.chunk_size, len(tokens))

                # Get tokens for this chunk
                chunk_tokens = tokens[start_idx:end_idx]
                chunk_text = self._tokens_to_text(chunk_tokens)

                chunk = DocumentChunk(
                    content=chunk_text,
                    chunk_id=f"{document_id}_chunk_{len(chunks)}",
                    document_id=document_id or "unknown",
                    chunk_index=len(chunks),
                    metadata=metadata.copy(),
                    start_char=start_idx,
                    end_char=end_idx,
                )
                chunks.append(chunk)

                # Move start index with overlap
                start_idx = max(start_idx + 1, end_idx - self.chunk_overlap)

            return chunks

        except Exception as e:
            logger.error(f"Error in token chunking: {e}")
            return [
                DocumentChunk(
                    content=content,
                    chunk_id=f"{document_id}_chunk_0",
                    document_id=document_id or "unknown",
                    chunk_index=0,
                    metadata=metadata.copy(),
                )
            ]

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        try:
            # Try to use tiktoken if available
            import tiktoken

            encoding = tiktoken.get_encoding(self.encoding_name)
            return encoding.encode(text)
        except ImportError:
            # Fallback to simple whitespace tokenization
            return text.split()

    def _tokens_to_text(self, tokens: List[str]) -> str:
        """Convert tokens back to text."""
        try:
            # Try to use tiktoken if available
            import tiktoken

            encoding = tiktoken.get_encoding(self.encoding_name)
            return encoding.decode(tokens)
        except ImportError:
            # Fallback - join tokens with space
            return " ".join(tokens)


class SemanticChunker(BaseChunker):
    """Semantic chunker that splits on semantic boundaries."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_chunk_size = self.config.get("max_chunk_size", 1000)
        self.min_chunk_size = self.config.get("min_chunk_size", 200)
        self.separator_patterns = self.config.get(
            "separators",
            [
                r"\n\n\n",  # Triple newlines
                r"\n\n",  # Double newlines
                r"\n",  # Single newlines
                r"\. ",  # Sentence end
                r"! ",  # Exclamation
                r"\? ",  # Question
            ],
        )

    async def chunk(
        self, content: str, metadata: Dict[str, Any], document_id: Optional[str] = None
    ) -> List[DocumentChunk]:
        """Chunk content by semantic boundaries."""
        try:
            if len(content) <= self.max_chunk_size:
                return [
                    DocumentChunk(
                        content=content,
                        chunk_id=f"{document_id}_chunk_0",
                        document_id=document_id or "unknown",
                        chunk_index=0,
                        metadata=metadata.copy(),
                    )
                ]

            chunks = []
            remaining_content = content
            chunk_index = 0

            while remaining_content:
                # Find the best split point
                split_point = self._find_split_point(remaining_content)

                if split_point == 0:  # No good split found
                    # Force split at max size
                    split_point = min(self.max_chunk_size, len(remaining_content))

                chunk_content = remaining_content[:split_point].strip()

                if chunk_content:
                    chunk = DocumentChunk(
                        content=chunk_content,
                        chunk_id=f"{document_id}_chunk_{chunk_index}",
                        document_id=document_id or "unknown",
                        chunk_index=chunk_index,
                        metadata=metadata.copy(),
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                remaining_content = remaining_content[split_point:].strip()

            return chunks

        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            return [
                DocumentChunk(
                    content=content,
                    chunk_id=f"{document_id}_chunk_0",
                    document_id=document_id or "unknown",
                    chunk_index=0,
                    metadata=metadata.copy(),
                )
            ]

    def _find_split_point(self, content: str) -> int:
        """Find the best semantic split point."""
        if len(content) <= self.max_chunk_size:
            return len(content)

        # Try each separator pattern
        for pattern in self.separator_patterns:
            matches = list(re.finditer(pattern, content))

            # Find the split point closest to max_chunk_size
            best_split = 0
            for match in matches:
                split_pos = match.end()
                if split_pos <= self.max_chunk_size and split_pos > best_split:
                    best_split = split_pos

            if best_split >= self.min_chunk_size:
                return best_split

        # No semantic split found, return 0 to indicate force split
        return 0


class FixedSizeChunker(BaseChunker):
    """Fixed-size chunker that splits by character count."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 200)

    async def chunk(
        self, content: str, metadata: Dict[str, Any], document_id: Optional[str] = None
    ) -> List[DocumentChunk]:
        """Chunk content by character count."""
        try:
            if len(content) <= self.chunk_size:
                return [
                    DocumentChunk(
                        content=content,
                        chunk_id=f"{document_id}_chunk_0",
                        document_id=document_id or "unknown",
                        chunk_index=0,
                        metadata=metadata.copy(),
                    )
                ]

            chunks = []
            start_idx = 0

            while start_idx < len(content):
                end_idx = min(start_idx + self.chunk_size, len(content))

                chunk_content = content[start_idx:end_idx]

                chunk = DocumentChunk(
                    content=chunk_content,
                    chunk_id=f"{document_id}_chunk_{len(chunks)}",
                    document_id=document_id or "unknown",
                    chunk_index=len(chunks),
                    metadata=metadata.copy(),
                    start_char=start_idx,
                    end_char=end_idx,
                )
                chunks.append(chunk)

                # Move start index with overlap
                start_idx = max(start_idx + 1, end_idx - self.chunk_overlap)

            return chunks

        except Exception as e:
            logger.error(f"Error in fixed-size chunking: {e}")
            return [
                DocumentChunk(
                    content=content,
                    chunk_id=f"{document_id}_chunk_0",
                    document_id=document_id or "unknown",
                    chunk_index=0,
                    metadata=metadata.copy(),
                )
            ]


def create_chunker(strategy: str, config: Optional[Dict[str, Any]] = None) -> BaseChunker:
    """Create a chunker based on strategy."""
    if strategy == "semantic":
        return SemanticChunker(config)
    elif strategy == "token":
        return TokenChunker(config)
    elif strategy == "fixed":
        return FixedSizeChunker(config)
    else:
        logger.warning(f"Unknown chunking strategy: {strategy}, using semantic")
        return SemanticChunker(config)
