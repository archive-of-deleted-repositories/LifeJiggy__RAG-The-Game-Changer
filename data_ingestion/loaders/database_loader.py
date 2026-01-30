"""
Database Document Loader - RAG-The-Game-Changer

Production-grade database loader with SQL query support.
"""

import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import logging

from .base_classes import DocumentLoader, DocumentMetadata, LoadedDocument, LoaderError

logger = logging.getLogger(__name__)


class DatabaseLoader(DocumentLoader):
    """Loader for database content with SQL query support."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.db_type = self.config.get("db_type", "sqlite")
        self.connection_string = self.config.get("connection_string")
        self.host = self.config.get("host", "localhost")
        self.port = self.config.get("port")
        self.database = self.config.get("database")
        self.username = self.config.get("username")
        self.password = self.config.get("password")
        self.query = self.config.get("query")
        self.table = self.config.get("table")
        self.batch_size = self.config.get("batch_size", 1000)
        self.max_rows = self.config.get("max_rows", 10000)
    
    def can_load(self, source: Union[str, Dict]) -> bool:
        """Check if source is a database configuration."""
        if isinstance(source, dict):
            return source.get("type") == "database"
        return False
    
    async def load(self, source: Union[str, Dict]) -> List[LoadedDocument]:
        """Load documents from database."""
        try:
            if isinstance(source, dict):
                self._update_config_from_dict(source)
            
            conn = await self._get_connection()
            try:
                documents = await self._load_from_database(conn)
                return documents
            finally:
                await self._close_connection(conn)
        except Exception as e:
            logger.error(f"Error loading from database: {e}")
            raise LoaderError(f"Failed to load from database: {e}", source=str(source))
    
    def _update_config_from_dict(self, source: Dict):
        self.db_type = source.get("db_type", self.db_type)
        self.query = source.get("query", self.query)
        self.table = source.get("table", self.table)
        self.connection_string = source.get("connection_string", self.connection_string)
    
    async def _get_connection(self) -> Any:
        """Get database connection based on type."""
        if self.db_type == "sqlite":
            import sqlite3
            db_path = self.database or ":memory:"
            return sqlite3.connect(db_path)
        raise LoaderError(f"Unsupported database type: {self.db_type}")
    
    async def _close_connection(self, conn: Any):
        if hasattr(conn, "close"):
            conn.close()
    
    async def _load_from_database(self, conn: Any) -> List[LoadedDocument]:
        documents = []
        query = self._build_query()
        
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        
        for idx, row in enumerate(rows[:self.max_rows]):
            try:
                document = self._row_to_document(row, columns, idx)
                documents.append(document)
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
        
        return documents
    
    def _build_query(self) -> str:
        if self.query:
            return self.query
        if self.table:
            return f"SELECT * FROM {self.table} LIMIT {self.max_rows}"
        raise LoaderError("No query or table specified")
    
    def _row_to_document(self, row: tuple, columns: List[str], index: int) -> LoadedDocument:
        row_dict = {}
        for i, col in enumerate(columns):
            if i < len(row):
                row_dict[col] = row[i]
        
        content_parts = []
        for col, val in row_dict.items():
            if isinstance(val, str):
                content_parts.append(f"{col}: {val}")
        
        content = chr(10).join(content_parts) if content_parts else str(row_dict)
        
        metadata = DocumentMetadata(
            source=f"database:{self.database or self.table}",
            source_type="database",
            title=f"Row_{index}",
            extra={"row_data": row_dict}
        )
        
        return LoadedDocument(
            content=content,
            metadata=metadata,
            document_id=self._generate_document_id(content, str(index))
        )
