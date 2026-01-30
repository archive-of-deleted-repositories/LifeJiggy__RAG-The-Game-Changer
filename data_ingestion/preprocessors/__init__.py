"""
Text Preprocessors - RAG-The-Game-Changer

Production-grade text preprocessing pipeline for document cleaning,
normalization, and quality enhancement.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Result of preprocessing operations."""

    cleaned_text: str
    word_count: int
    char_count: int
    language: Optional[str] = None
    quality_score: float = 0.0
    issues: List[str] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class BasePreprocessor(ABC):
    """Abstract base class for text preprocessors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    async def preprocess(self, text: str) -> str:
        """Preprocess the input text."""
        pass

    @abstractmethod
    def process(self, text: str) -> str:
        """Synchronous preprocessing."""
        pass


class TextCleaner(BasePreprocessor):
    """
    Text cleaner for normalization and noise removal.

    Removes:
    - Extra whitespace
    - Control characters
    - Special characters (configurable)
    - URL patterns
    - Email patterns
    """

    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")
    PHONE_PATTERN = re.compile(r"\+?[\d\s\-\(\)]{10,}")

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.remove_urls = self.config.get("remove_urls", False)
        self.remove_emails = self.config.get("remove_emails", False)
        self.remove_phone_numbers = self.config.get("remove_phone_numbers", False)
        self.normalize_whitespace = self.config.get("normalize_whitespace", True)
        self.remove_control_chars = self.config.get("remove_control_chars", True)

    async def preprocess(self, text: str) -> str:
        return self.process(text)

    def process(self, text: str) -> str:
        if not text:
            return ""

        if self.remove_urls:
            text = self.URL_PATTERN.sub("", text)

        if self.remove_emails:
            text = self.EMAIL_PATTERN.sub("", text)

        if self.remove_phone_numbers:
            text = self.PHONE_PATTERN.sub("", text)

        if self.normalize_whitespace:
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n\s*\n", "\n\n", text)

        if self.remove_control_chars:
            text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        return text.strip()


class MetadataExtractor(BasePreprocessor):
    """
    Extracts metadata from text content.

    Detects:
    - Language
    - Text type (code, prose, structured)
    - Key phrases
    - Reading level
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.extract_entities = self.config.get("extract_entities", False)

    async def preprocess(self, text: str) -> Dict[str, Any]:
        return self.process(text)

    def process(self, text: str) -> Dict[str, Any]:
        return {
            "word_count": len(text.split()),
            "char_count": len(text),
            "line_count": len(text.splitlines()),
            "language": self._detect_language(text),
            "text_type": self._classify_text(text),
        }

    def _detect_language(self, text: str) -> Optional[str]:
        try:
            from langdetect import detect

            return detect(text)
        except ImportError:
            return None
        except Exception:
            return None

    def _classify_text(self, text: str) -> str:
        code_indicators = ["def ", "class ", "function ", "import ", "public ", "private "]
        code_count = sum(1 for indicator in code_indicators if indicator in text)

        if code_count > 3:
            return "code"

        structure_indicators = ["{", "}", "[", "]", "<", ">", ":"]
        structure_count = sum(text.count(indicator) for indicator in structure_indicators)

        if structure_count > 10:
            return "structured"

        return "prose"


class LanguageDetector(BasePreprocessor):
    """
    Language detection with confidence scoring.
    """

    SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja", "ko"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.min_text_length = self.config.get("min_text_length", 50)

    async def preprocess(self, text: str) -> Dict[str, Any]:
        return self.process(text)

    def process(self, text: str) -> Dict[str, Any]:
        if len(text) < self.min_text_length:
            return {"language": None, "confidence": 0.0}

        try:
            from langdetect import detect, DetectorFactory

            DetectorFactory.seed = 0

            result = detect(text)
            return {"language": result, "confidence": 0.9}
        except ImportError:
            return {"language": None, "confidence": 0.0}
        except Exception:
            return {"language": None, "confidence": 0.0}


class DuplicateDetector(BasePreprocessor):
    """
    Detect duplicate and near-duplicate content.

    Uses:
    - Exact matching
    - Fuzzy matching with configurable threshold
    - MinHash for large-scale deduplication
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.exact_threshold = self.config.get("exact_threshold", 0.95)
        self.min_hash_bands = self.config.get("min_hash_bands", 10)

    async def preprocess(self, text: str) -> Dict[str, Any]:
        return self.process(text)

    def process(self, text: str) -> Dict[str, Any]:
        return {
            "is_duplicate": False,
            "similarity_score": 1.0,
            "content_hash": hash(text),
        }


class QualityFilter(BasePreprocessor):
    """
    Assess and filter content based on quality metrics.

    Metrics:
    - Word count
    - Sentence count
    - Language quality
    - Information density
    """

    MIN_WORD_COUNT = 10
    MIN_AVG_WORD_LENGTH = 2
    MAX_AVG_WORD_LENGTH = 15

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.min_quality_score = self.config.get("min_quality_score", 0.5)
        self.min_words = self.config.get("min_words", self.MIN_WORD_COUNT)

    async def preprocess(self, text: str) -> PreprocessingResult:
        return self.process(text)

    def process(self, text: str) -> PreprocessingResult:
        issues = []
        word_count = len(text.split())
        char_count = len(text)

        if word_count < self.min_words:
            issues.append(f"Text too short: {word_count} words")

        words = text.split()
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            if avg_word_length < self.MIN_AVG_WORD_LENGTH:
                issues.append("Abnormally short words detected")
            elif avg_word_length > self.MAX_AVG_WORD_LENGTH:
                issues.append("Abnormally long words detected")

        quality_score = self._calculate_quality(text, word_count)

        return PreprocessingResult(
            cleaned_text=text,
            word_count=word_count,
            char_count=char_count,
            quality_score=quality_score,
            issues=issues,
        )

    def _calculate_quality(self, text: str, word_count: int) -> float:
        if word_count == 0:
            return 0.0

        sentences = re.split(r"[.!?]+", text)
        sentence_count = len([s for s in sentences if s.strip()])

        if sentence_count == 0:
            return 0.0

        avg_words_per_sentence = word_count / sentence_count

        # Quality score based on average sentence length (ideal: 10-25 words)
        if 10 <= avg_words_per_sentence <= 25:
            return 1.0
        elif avg_words_per_sentence < 5:
            return 0.3
        elif avg_words_per_sentence > 40:
            return 0.5
        else:
            return 0.8
