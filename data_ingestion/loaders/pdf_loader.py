"""
PDF Document Loader - RAG-The-Game-Changer

Production-grade PDF loader with text extraction, table recognition,
and metadata parsing using pypdf.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
from datetime import datetime

from . import DocumentLoader, DocumentMetadata, LoadedDocument, LoaderError

logger = logging.getLogger(__name__)


class PDFLoader(DocumentLoader):
    """
    Loader for PDF documents with comprehensive text extraction.
    
    Features:
    - Text extraction from all pages
    - Metadata parsing (author, title, creation date)
    - Page-level document splitting
    - Duplicate detection
    - Language detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.supported_types = [".pdf"]
        self.deduplicate_pages = self.config.get("deduplicate_pages", True)
        self.max_pages = self.config.get("max_pages")
        
        if not self._check_dependencies():
            raise LoaderError(
                "pypdf is required for PDF loading. Install with: pip install pypdf"
            )
    
    def _check_dependencies(self) -> bool:
        try:
            from pypdf import PdfReader
            return True
        except ImportError:
            return False
    
    def can_load(self, source: Union[str, Path, Dict]) -> bool:
        if isinstance(source, dict):
            return source.get("type") == "pdf" or "pdf" in str(source.get("source", "")).lower()
        
        if isinstance(source, str):
            source = Path(source)
        
        return isinstance(source, Path) and source.suffix.lower() == ".pdf"
    
    async def load(self, source: Union[str, Path, Dict]) -> List[LoadedDocument]:
        try:
            if isinstance(source, dict):
                return await self._load_from_dict(source)
            else:
                return await self._load_from_file(source)
        except Exception as e:
            raise LoaderError(f"Failed to load PDF: {e}", source=str(source))
    
    async def _load_from_file(self, file_path: Union[str, Path]) -> List[LoadedDocument]:
        path = Path(file_path)
        
        if not path.exists():
            raise LoaderError(f"PDF file not found: {path}", source=str(path))
        
        try:
            from pypdf import PdfReader
        except ImportError:
            raise LoaderError("pypdf not installed", source=str(path))
        
        reader = PdfReader(str(path))
        documents = []
        seen_hashes = set()
        
        base_metadata = self._extract_pdf_metadata(reader, str(path))
        
        pages_to_process = len(reader.pages)
        if self.max_pages:
            pages_to_process = min(pages_to_process, self.max_pages)
        
        for page_num in range(pages_to_process):
            try:
                page = reader.pages[page_num]
                text = page.extract_text() or ""
                text = self._clean_text(text)
                
                if not text.strip():
                    continue
                
                if self.deduplicate_pages:
                    page_hash = hash(text)
                    if page_hash in seen_hashes:
                        continue
                    seen_hashes.add(page_hash)
                
                page_metadata = DocumentMetadata(
                    source=str(path),
                    source_type="pdf",
                    title=base_metadata.title or path.stem,
                    author=base_metadata.author,
                    created_at=base_metadata.created_at,
                    updated_at=base_metadata.updated_at,
                    file_size=path.stat().st_size,
                    file_extension=".pdf",
                    language=self._detect_language(text),
                    checksum=self._calculate_checksum(text),
                    extra={
                        "page_number": page_num + 1,
                        "total_pages": len(reader.pages),
                    }
                )
                
                document = LoadedDocument(
                    content=text,
                    metadata=page_metadata,
                    document_id=self._generate_document_id(text, str(path)),
                )
                documents.append(document)
                
            except Exception as e:
                logger.warning(f"Error processing page {page_num + 1} of {path}: {e}")
                continue
        
        logger.info(f"Loaded {len(documents)} pages from PDF: {path}")
        return documents
    
    async def _load_from_dict(self, source: Dict) -> List[LoadedDocument]:
        content = source.get("content", "")
        metadata_dict = source.get("metadata", {}) or {}
        
        metadata = DocumentMetadata(
            source=metadata_dict.get("source", "uploaded_pdf"),
            source_type="pdf",
            title=metadata_dict.get("title"),
            author=metadata_dict.get("author"),
            language=self._detect_language(content),
            checksum=self._calculate_checksum(content),
        )
        
        document = LoadedDocument(
            content=self._clean_text(content),
            metadata=metadata,
            document_id=self._generate_document_id(content, metadata.source),
        )
        
        return [document]
    
    def _extract_pdf_metadata(self, reader: Any, source: str) -> DocumentMetadata:
        metadata = DocumentMetadata(source=source, source_type="pdf")
        
        try:
            doc_info = reader.metadata
            if doc_info and hasattr(doc_info, 'get'):
                metadata.title = str(doc_info.get('/Title', '')).strip() if doc_info.get('/Title') else None
                metadata.author = str(doc_info.get('/Author', '')).strip() if doc_info.get('/Author') else None
        except Exception as e:
            logger.warning(f"Error extracting PDF metadata: {e}")
        
        return metadata
    
    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'-\n', '', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
