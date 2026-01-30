"""
Code Document Loader - RAG-The-Game-Changer

Production-grade code loader with syntax parsing,
language detection, and structure extraction.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
import hashlib

from . import DocumentLoader, DocumentMetadata, LoadedDocument, LoaderError

logger = logging.getLogger(__name__)


class CodeLoader(DocumentLoader):
    """
    Loader for code files with syntax-aware processing.

    Features:
    - Multi-language support (Python, JavaScript, Java, C/C++, Go, Rust, etc.)
    - Comment extraction and filtering
    - Function/class structure extraction
    - Import/enumeration parsing
    - Language detection

    Supported extensions:
    .py, .js, .ts, .java, .c, .cpp, .h, .hpp, .go, .rs, .rb, .php, .swift, .kt, .scala
    """

    LANGUAGE_CONFIGS: Dict[str, Dict[str, Any]] = {
        "python": {
            "extensions": [".py", ".pyw"],
            "comment_patterns": [r"#.*$", r'"""[\s\S]*?"""', r"'''[\s\S]*?'''"],
            "string_patterns": [r'r?""".*?"""', r"r?'''.*?'''", r'"[^"]*"', r"'[^']*'"],
            "function_pattern": r"^def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[\w\[\]]+\s*)?:",
            "class_pattern": r"^class\s+(\w+)(?:\([^)]*\))?\s*:",
            "import_pattern": r"^(?:from|import)\s+([\w.]+)",
        },
        "javascript": {
            "extensions": [".js", ".jsx", ".mjs", ".cjs"],
            "comment_patterns": [r"//.*$", r"/\*[\s\S]*?\*/"],
            "string_patterns": [r"`[^`]*`", r'"[^"]*"', r"'[^']*'"],
            "function_pattern": r"(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?function)",
            "class_pattern": r"^class\s+(\w+)",
            "import_pattern": r"^(?:import|export(?:\s+\{?))\s+([\w.\s{},*]+)",
        },
        "typescript": {
            "extensions": [".ts", ".tsx"],
            "comment_patterns": [r"//.*$", r"/\*[\s\S]*?\*/"],
            "string_patterns": [r"`[^`]*`", r'"[^"]*"', r"'[^']*'"],
            "function_pattern": r"(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?function)",
            "class_pattern": r"^class\s+(\w+)",
            "import_pattern": r"^(?:import|export(?:\s+\{?))\s+([\w.\s{},*]+)",
        },
        "java": {
            "extensions": [".java"],
            "comment_patterns": [r"//.*$", r"/\*[\s\S]*?\*/", r"/\*\*[\s\S]*?\*/"],
            "string_patterns": [r'"[^"]*"'],
            "function_pattern": r"(?:public|private|protected|\s)*(?:static\s+)?(?:final\s+)?(?:[\w<>[\]]+\s+)+(\w+)\s*\([^)]*\)",
            "class_pattern": r"(?:public|private|protected|\s)*class\s+(\w+)",
            "import_pattern": r"^import\s+([\w.]+);",
        },
        "c": {
            "extensions": [".c", ".h"],
            "comment_patterns": [r"//.*$", r"/\*[\s\S]*?\*/"],
            "string_patterns": [r'"[^"]*"'],
            "function_pattern": r"(?:static\s+)?(?:inline\s+)?(?:[\w*]+\s+)+(\w+)\s*\([^)]*\)",
            "class_pattern": None,
            "import_pattern": r'^#include\s+[<"]([^>"]+)[">]',
        },
        "cpp": {
            "extensions": [".cpp", ".cc", ".cxx", ".hpp", ".hxx"],
            "comment_patterns": [r"//.*$", r"/\*[\s\S]*?\*/"],
            "string_patterns": [r'"[^"]*"', r'R"([^)]*)\((?:(?!\1).)*\1"'],
            "function_pattern": r"(?:static|constexpr|inline\s+)?(?:[\w*]+\s+)+(\w+)\s*\([^)]*\)",
            "class_pattern": r"(?:class|struct)\s+(\w+)",
            "import_pattern": r'^#include\s+[<"]([^>"]+)[">]',
        },
        "go": {
            "extensions": [".go"],
            "comment_patterns": [r"//.*$", r"/\*[\s\S]*?\*/"],
            "string_patterns": [r"`[^`]*`", r'"[^"]*"'],
            "function_pattern": r"func\s+(?:\([^)]+\)\s*)?(\w+)\s*\([^)]*\)",
            "class_pattern": r"type\s+(\w+)\s+struct",
            "import_pattern": r'^import\s*(?:\(\s*)?["\']([^"\']+)["\']',
        },
        "rust": {
            "extensions": [".rs"],
            "comment_patterns": [r"//.*$", r"/\*[\s\S]*?\*/", r"///.*$", r"/\*\*[\s\S]*?\*/"],
            "string_patterns": [r'"[^"]*"', r'r#".*"#', r'r#".*"#\d'],
            "function_pattern": r"(?:pub(?:\s+crate)?(?:\s+async)?\s+)?fn\s+(\w+)",
            "class_pattern": r"struct\s+(\w+)",
            "import_pattern": r"^use\s+([\w:]+)",
        },
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.supported_extensions: Set[str] = set()
        self.remove_comments = self.config.get("remove_comments", False)
        self.extract_structure = self.config.get("extract_structure", True)

        for lang_config in self.LANGUAGE_CONFIGS.values():
            self.supported_extensions.update(lang_config["extensions"])

        self.supported_types = list(self.supported_extensions)

    def can_load(self, source: Union[str, Path, Dict]) -> bool:
        if isinstance(source, dict):
            return source.get("type") == "code" or any(
                source.get("source", "").endswith(ext) for ext in self.supported_extensions
            )

        if isinstance(source, str):
            source = Path(source)

        return isinstance(source, Path) and source.suffix.lower() in self.supported_extensions

    async def load(self, source: Union[str, Path, Dict]) -> List[LoadedDocument]:
        if isinstance(source, dict):
            return await self._load_from_dict(source)
        else:
            return await self._load_from_file(source)

    async def _load_from_file(self, file_path: Union[str, Path]) -> List[LoadedDocument]:
        path = Path(file_path)

        if not path.exists():
            raise LoaderError(f"Code file not found: {path}", source=str(path))

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return [await self._process_code(content, str(path), path.suffix.lower())]

    async def _load_from_dict(self, source: Dict) -> List[LoadedDocument]:
        content = source.get("content", "")
        file_path = source.get("source", "unknown")
        extension = source.get("extension", "")

        return [await self._process_code(content, file_path, extension)]

    async def _process_code(self, content: str, source: str, extension: str) -> LoadedDocument:
        lang = self._detect_language(extension)
        config = self.LANGUAGE_CONFIGS.get(lang, {})

        if self.remove_comments:
            content = self._remove_comments(content, config)

        structure = {}
        if self.extract_structure:
            structure = self._extract_structure(content, config)

        metadata = DocumentMetadata(
            source=source,
            source_type="code",
            title=Path(source).stem,
            language=lang,
            file_size=len(content.encode("utf-8")),
            file_extension=extension,
            checksum=self._calculate_checksum(content),
            extra={
                "lines_of_code": len(content.splitlines()),
                "file_path": source,
                "structure": structure,
            },
        )

        return LoadedDocument(
            content=content,
            metadata=metadata,
            document_id=self._generate_document_id(content, source),
        )

    def _detect_language(self, extension: str) -> str:
        """Detect programming language from file extension."""
        ext = extension.lower()

        for lang, config in self.LANGUAGE_CONFIGS.items():
            if ext in config.get("extensions", []):
                return lang

        return "unknown"

    def _remove_comments(self, content: str, config: Dict) -> str:
        """Remove comments from code."""
        for pattern in config.get("comment_patterns", []):
            content = re.sub(pattern, "", content, flags=re.MULTILINE)
        return content

    def _extract_structure(self, content: str, config: Dict) -> Dict:
        """Extract code structure (functions, classes, imports)."""
        structure = {
            "functions": [],
            "classes": [],
            "imports": [],
        }

        # Extract functions
        func_pattern = config.get("function_pattern")
        if func_pattern:
            for match in re.finditer(func_pattern, content, re.MULTILINE):
                func_name = match.group(1) or match.group(2) if match.groups() else match.group(0)
                if func_name:
                    structure["functions"].append(func_name)

        # Extract classes
        class_pattern = config.get("class_pattern")
        if class_pattern:
            for match in re.finditer(class_pattern, content, re.MULTILINE):
                class_name = match.group(1)
                if class_name:
                    structure["classes"].append(class_name)

        # Extract imports
        import_pattern = config.get("import_pattern")
        if import_pattern:
            for match in re.finditer(import_pattern, content, re.MULTILINE):
                import_stmt = match.group(1)
                if import_stmt:
                    structure["imports"].append(import_stmt)

        return structure

    def _generate_document_id(self, content: str, source: str) -> str:
        """Generate unique document ID from content and source."""
        hash_input = f"{source}:{content[:1000]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def _calculate_checksum(self, content: str) -> str:
        """Calculate MD5 checksum of content."""
        return hashlib.md5(content.encode()).hexdigest()
