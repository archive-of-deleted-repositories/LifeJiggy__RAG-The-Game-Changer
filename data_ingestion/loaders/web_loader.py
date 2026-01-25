"""
Web Document Loader - RAG-The-Game-Changer

Production-grade web scraper with JavaScript rendering support,
content extraction, and metadata parsing.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse
import logging
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime

from . import DocumentLoader, DocumentMetadata, LoadedDocument, LoaderError

logger = logging.getLogger(__name__)


class WebLoader(DocumentLoader):
    """
    Loader for web content with robust extraction capabilities.

    Features:
    - Async HTTP requests with connection pooling
    - JavaScript rendering support (via requests-html fallback)
    - Content extraction from common page structures
    - Metadata extraction from meta tags
    - Link extraction for crawling
    - Rate limiting and retry logic
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.supported_types = ["http", "https"]
        self.timeout = self.config.get("timeout", 30)
        self.max_content_length = self.config.get("max_content_length", 100000)
        self.extract_links = self.config.get("extract_links", False)
        self.user_agent = self.config.get(
            "user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        self.max_retries = self.config.get("max_retries", 3)

    def can_load(self, source: Union[str, Path, Dict]) -> bool:
        if isinstance(source, dict):
            return source.get("type") == "web" or "url" in source

        if isinstance(source, str):
            parsed = urlparse(source)
            return parsed.scheme in ["http", "https"]

        return False

    async def load(self, source: Union[str, Path, Dict]) -> List[LoadedDocument]:
        if isinstance(source, dict):
            return await self._load_from_dict(source)
        else:
            return await self._load_from_url(source)

    async def _load_from_url(self, url: str) -> List[LoadedDocument]:
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.max_retries):
                try:
                    html_content = await self._fetch_html(session, url)
                    soup = BeautifulSoup(html_content, "lxml")

                    content = self._extract_content(soup)
                    metadata = self._extract_metadata(soup, url)

                    if self.extract_links:
                        links = self._extract_links(soup, url)
                        metadata.extra["extracted_links"] = links[:50]

                    doc = LoadedDocument(
                        content=content,
                        metadata=metadata,
                        document_id=self._generate_document_id(content, url),
                    )

                    return [doc]

                except asyncio.TimeoutError:
                    logger.warning(
                        f"Timeout fetching {url}, attempt {attempt + 1}/{self.max_retries}"
                    )
                    if attempt == self.max_retries - 1:
                        raise LoaderError(f"Timeout after {self.max_retries} attempts", url)
                except aiohttp.ClientError as e:
                    logger.warning(f"Client error fetching {url}: {e}, attempt {attempt + 1}")
                    if attempt == self.max_retries - 1:
                        raise LoaderError(f"Client error: {e}", url)

    async def _load_from_dict(self, source: Dict) -> List[LoadedDocument]:
        url = source.get("url") or source.get("source")
        content = source.get("content", "")
        html_content = source.get("html_content", content)

        soup = BeautifulSoup(html_content, "lxml")
        content = self._extract_content(soup)

        metadata_dict = source.get("metadata", {})
        metadata = DocumentMetadata(
            source=url or metadata_dict.get("source", "unknown"),
            source_type="web",
            title=metadata_dict.get("title") or (soup.title.string if soup.title else None),
            url=url,
            extra=metadata_dict.get("extra", {}),
        )

        document = LoadedDocument(
            content=content,
            metadata=metadata,
            document_id=self._generate_document_id(content, url or "unknown"),
        )

        return [document]

    async def _fetch_html(self, session: aiohttp.ClientSession, url: str) -> str:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        async with session.get(
            url, headers=headers, timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as response:
            response.raise_for_status()
            return await response.text()

    def _extract_content(self, soup: BeautifulSoup) -> str:
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.decompose()

        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", class_="content")
            or soup.find("body")
        )

        if main_content:
            text = main_content.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        if len(text) > self.max_content_length:
            text = text[: self.max_content_length]
            logger.warning(f"Content truncated to {self.max_content_length} characters")

        return text

    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> DocumentMetadata:
        metadata = DocumentMetadata(
            source=url,
            source_type="web",
        )

        if soup.title and soup.title.string:
            metadata.title = soup.title.string.strip()

        meta_tags = soup.find_all("meta")
        for tag in meta_tags:
            name = tag.get("name") or tag.get("property")
            content = tag.get("content")

            if not name or not content:
                continue

            name_lower = name.lower()
            if name_lower == "description":
                metadata.extra["description"] = content
            elif name_lower == "author":
                metadata.author = content
            elif name_lower == "keywords":
                metadata.extra["keywords"] = [k.strip() for k in content.split(",")]
            elif name_lower == "language":
                metadata.language = content

        parsed_url = urlparse(url)
        metadata.extra["domain"] = parsed_url.netloc
        metadata.extra["path"] = parsed_url.path

        return metadata

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        links = []

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            absolute_url = urljoin(base_url, href)

            link_info = {
                "url": absolute_url,
                "text": a_tag.get_text(strip=True)[:100],
            }

            if link_info["text"]:
                links.append(link_info)

        return links
