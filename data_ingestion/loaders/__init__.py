"""
Document Loaders - RAG-The-Game-Changer

Production-grade document loaders for various file formats and sources.
"""

from .base_classes import DocumentLoader, DocumentMetadata, LoadedDocument, LoaderError
from .pdf_loader import PDFLoader
from .web_loader import WebLoader
from .code_loader import CodeLoader
from .text_loader import TextLoader

__all__ = [
    "DocumentLoader",
    "DocumentMetadata",
    "LoadedDocument",
    "LoaderError",
    "PDFLoader",
    "WebLoader",
    "CodeLoader",
    "TextLoader",
]
