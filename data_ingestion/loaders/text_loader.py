"""
Text Document Loader - RAG-The-Game-Changer

Production-grade text file loader with encoding detection
and line-aware processing.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import chardet

from . import DocumentLoader, DocumentMetadata, LoadedDocument, LoaderError

logger = logging.getLogger(__name__)


class TextLoader(DocumentLoader):
    """
    Loader for plain text files with robust encoding handling.
    
    Features:
    - Automatic encoding detection
    - Line-based processing
    - Metadata extraction from file
    - Support for multiple text formats
    """
    
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.csv', '.json', '.xml', '.yaml', '.yml', '.rst', '.log'}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.supported_types = list(self.SUPPORTED_EXTENSIONS)
        self.max_file_size = self.config.get("max_file_size", 10 * 1024 * 1024)
        self.detect_encoding = self.config.get("detect_encoding", True)
    
    def can_load(self, source: Union[str, Path, Dict]) -> bool:
        if isinstance(source, dict):
            return source.get("type") == "text"
        
        if isinstance(source, str):
            source = Path(source)
        
        return isinstance(source, Path) and source.suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    async def load(self, source: Union[str, Path, Dict]) -> List[LoadedDocument]:
        if isinstance(source, dict):
            return await self._load_from_dict(source)
        else:
            return await self._load_from_file(source)
    
    async def _load_from_file(self, file_path: Union[str, Path]) -> List[LoadedDocument]:
        path = Path(file_path)
        
        if not path.exists():
            raise LoaderError(f"Text file not found: {path}", source=str(path))
        
        file_size = path.stat().st_size
        if file_size > self.max_file_size:
            raise LoaderError(f"File too large: {file_size} > {self.max_file_size}", source=str(path))
        
        encoding = self._detect_encoding(path) if self.detect_encoding else 'utf-8'
        
        with open(path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
        
        metadata = DocumentMetadata(
            source=str(path),
            source_type="text",
            title=path.stem,
            file_size=file_size,
            file_extension=path.suffix,
            language=self._detect_language(content),
            checksum=self._calculate_checksum(content),
            extra={
                "encoding": encoding,
                "line_count": len(content.splitlines()),
            }
        )
        
        document = LoadedDocument(
            content=content,
            metadata=metadata,
            document_id=self._generate_document_id(content, str(path)),
        )
        
        logger.info(f"Loaded text file: {path}")
        return [document]
    
    async def _load_from_dict(self, source: Dict) -> List[LoadedDocument]:
        content = source.get("content", "")
        metadata_dict = source.get("metadata", {})
        
        metadata = DocumentMetadata(
            source=metadata_dict.get("source", "text_input"),
            source_type="text",
            title=metadata_dict.get("title"),
            language=self._detect_language(content),
            checksum=self._calculate_checksum(content),
        )
        
        document = LoadedDocument(
            content=content,
            metadata=metadata,
            document_id=self._generate_document_id(content, metadata.source),
        )
        
        return [document]
    
    def _detect_encoding(self, path: Path) -> str:
        with open(path, 'rb') as f:
            raw_data = f.read(1024)
        
        result = chardet.detect(raw_data)
        return result.get('encoding', 'utf-8')
