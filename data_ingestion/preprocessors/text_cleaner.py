"""
Text Preprocessor - RAG-The-Game-Changer

Basic text cleaning and preprocessing utilities.
"""

import re
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TextCleaner:
    """Text cleaning and preprocessing utilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.remove_extra_whitespace = self.config.get("remove_extra_whitespace", True)
        self.normalize_unicode = self.config.get("normalize_unicode", True)
        self.remove_special_chars = self.config.get("remove_special_chars", False)

    def clean(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""

        # Remove excessive whitespace
        if self.remove_extra_whitespace:
            text = re.sub(r"\s+", " ", text)
            text = text.strip()

        # Remove special characters if enabled
        if self.remove_special_chars:
            text = re.sub(r"[^\w\s\.,!?;:\-\'\"]", "", text)

        # Normalize Unicode if enabled
        if self.normalize_unicode:
            try:
                import unicodedata

                text = unicodedata.normalize("NFKC", text)
            except ImportError:
                logger.warning("unicodedata not available for Unicode normalization")

        return text

    def clean_batch(self, texts: list[str]) -> list[str]:
        """Clean multiple texts."""
        return [self.clean(text) for text in texts]
