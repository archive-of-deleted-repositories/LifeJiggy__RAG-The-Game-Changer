"""
API Document Loader - RAG-The-Game-Changer

Production-grade API loader for REST endpoints.
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import logging

from .base_classes import DocumentLoader, DocumentMetadata, LoadedDocument, LoaderError

logger = logging.getLogger(__name__)


class APILoader(DocumentLoader):
    """Loader for API endpoints with HTTP support."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.endpoint = self.config.get("endpoint")
        self.method = self.config.get("method", "GET")
        self.headers = self.config.get("headers", {})
        self.params = self.config.get("params", {})
        self.body = self.config.get("body")
        self.auth_type = self.config.get("auth_type")
        self.auth_token = self.config.get("auth_token")
        self.timeout = self.config.get("timeout", 30)
        self.max_pages = self.config.get("max_pages", 1)
    
    def can_load(self, source: Union[str, Dict]) -> bool:
        """Check if source is an API endpoint."""
        if isinstance(source, dict):
            return source.get("type") == "api" or "endpoint" in source
        if isinstance(source, str):
            return source.startswith(("http://", "https://"))
        return False
    
    async def load(self, source: Union[str, Dict]) -> List[LoadedDocument]:
        """Load documents from API endpoint."""
        try:
            if isinstance(source, dict):
                self._update_config_from_dict(source)
            elif isinstance(source, str):
                self.endpoint = source
            
            return await self._fetch_from_api()
        except Exception as e:
            logger.error(f"Error loading from API: {e}")
            raise LoaderError(f"Failed to load from API: {e}", source=str(source))
    
    def _update_config_from_dict(self, source: Dict):
        self.endpoint = source.get("endpoint", self.endpoint)
        self.method = source.get("method", self.method)
        self.headers = source.get("headers", self.headers)
        self.params = source.get("params", self.params)
        self.body = source.get("body", self.body)
    
    async def _fetch_from_api(self) -> List[LoadedDocument]:
        try:
            import aiohttp
        except ImportError:
            raise LoaderError("aiohttp not installed. Install with: pip install aiohttp")
        
        documents = []
        
        async with aiohttp.ClientSession() as session:
            headers = self.headers.copy()
            
            if self.auth_type == "bearer" and self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            elif self.auth_type == "api_key" and self.auth_token:
                headers["X-API-Key"] = self.auth_token
            
            for page in range(self.max_pages):
                params = self.params.copy()
                if self.max_pages > 1:
                    params["page"] = page + 1
                
                try:
                    async with session.request(
                        method=self.method,
                        url=self.endpoint,
                        headers=headers,
                        params=params,
                        json=self.body if self.body else None,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        if response.status != 200:
                            logger.warning(f"API returned status {response.status}")
                            continue
                        
                        data = await response.json()
                        docs = self._parse_response(data)
                        documents.extend(docs)
                        
                except Exception as e:
                    logger.error(f"Error fetching page {page}: {e}")
                    break
        
        return documents
    
    def _parse_response(self, data: Any) -> List[LoadedDocument]:
        documents = []
        
        if isinstance(data, list):
            for idx, item in enumerate(data):
                doc = self._item_to_document(item, idx)
                documents.append(doc)
        elif isinstance(data, dict):
            if "results" in data and isinstance(data["results"], list):
                for idx, item in enumerate(data["results"]):
                    doc = self._item_to_document(item, idx)
                    documents.append(doc)
            elif "data" in data and isinstance(data["data"], list):
                for idx, item in enumerate(data["data"]):
                    doc = self._item_to_document(item, idx)
                    documents.append(doc)
            else:
                doc = self._item_to_document(data, 0)
                documents.append(doc)
        
        return documents
    
    def _item_to_document(self, item: Any, index: int) -> LoadedDocument:
        if isinstance(item, str):
            content = item
        elif isinstance(item, dict):
            content = json.dumps(item, indent=2)
        else:
            content = str(item)
        
        metadata = DocumentMetadata(
            source=self.endpoint or "api",
            source_type="api",
            title=f"API_Response_{index}",
            extra={"item_index": index, "content_type": type(item).__name__}
        )
        
        return LoadedDocument(
            content=content,
            metadata=metadata,
            document_id=self._generate_document_id(content, f"{self.endpoint}_{index}")
        )
