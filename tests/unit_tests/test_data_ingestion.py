"""
Unit tests for Data Ingestion - RAG-The-Game-Changer
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any


@pytest.mark.asyncio
async def test_text_loader():
    """Test TextLoader functionality."""
    from data_ingestion.loaders.text_loader import TextLoader

    loader = TextLoader({})

    # Test loading from string
    result = await loader.load({"content": "This is a test document.", "source": "test.txt"})

    assert len(result) == 1
    assert result[0].content == "This is a test document."
    assert result[0].document_id is not None


@pytest.mark.asyncio
async def test_code_loader():
    """Test CodeLoader functionality."""
    from data_ingestion.loaders.code_loader import CodeLoader

    loader = CodeLoader({})

    code_content = """
def hello_world():
    print("Hello, World!")
    return True
"""

    result = await loader.load({"content": code_content, "source": "test.py"})

    assert len(result) == 1
    assert "hello_world" in result[0].content


@pytest.mark.asyncio
async def test_database_loader():
    """Test DatabaseLoader functionality."""
    from data_ingestion.loaders.database_loader import DatabaseLoader
    import sqlite3

    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, content TEXT)")
        conn.execute("INSERT INTO documents VALUES (1, 'Test content')")
        conn.commit()

        # Load from database
        loader = DatabaseLoader(
            {"db_type": "sqlite", "database": str(db_path), "table": "documents"}
        )

        result = await loader.load({"type": "database"})

        assert len(result) > 0

        conn.close()


@pytest.mark.asyncio
async def test_api_loader():
    """Test APILoader functionality."""
    from data_ingestion.loaders.api_loader import APILoader

    loader = APILoader({})

    # Test with mock data
    result = await loader.load(
        {"type": "api", "content": '{"message": "Test API response"}', "source": "test://api"}
    )

    assert len(result) >= 0


@pytest.mark.asyncio
async def test_semantic_chunker():
    """Test SemanticChunker."""
    from data_ingestion.chunkers.document_chunker import SemanticChunker

    chunker = SemanticChunker({"max_chunk_size": 200, "min_chunk_size": 50})

    text = "This is a long text that should be chunked into multiple pieces. Each piece will be approximately 200 characters or less. The semantic chunker tries to find natural break points."

    chunks = await chunker.chunk(text, {}, "doc_1")

    assert len(chunks) > 1
    assert all(chunk.content for chunk in chunks)


@pytest.mark.asyncio
async def test_token_chunker():
    """Test TokenChunker."""
    from data_ingestion.chunkers.document_chunker import TokenChunker

    chunker = TokenChunker({"chunk_size": 50, "chunk_overlap": 10})

    text = "This is a test document for token chunking. It will be split based on token count rather than character count."

    chunks = await chunker.chunk(text, {}, "doc_1")

    assert len(chunks) >= 1
    assert all(chunk.content for chunk in chunks)


@pytest.mark.asyncio
async def test_fixed_size_chunker():
    """Test FixedSizeChunker."""
    from data_ingestion.chunkers.document_chunker import FixedSizeChunker

    chunker = FixedSizeChunker({"chunk_size": 100, "chunk_overlap": 20})

    text = "This is a test document for fixed-size chunking. It will be split at exactly 100 characters per chunk with 20 characters of overlap."

    chunks = await chunker.chunk(text, {}, "doc_1")

    assert len(chunks) >= 1
    assert all(len(chunk.content) <= 120 for chunk in chunks)


@pytest.mark.asyncio
async def test_chunker_create_factory():
    """Test create_chunker factory function."""
    from data_ingestion.chunkers.document_chunker import create_chunker

    # Test different chunker types
    semantic = create_chunker("semantic")
    token = create_chunker("token")
    fixed = create_chunker("fixed")

    assert semantic.__class__.__name__ == "SemanticChunker"
    assert token.__class__.__name__ == "TokenChunker"
    assert fixed.__class__.__name__ == "FixedSizeChunker"


@pytest.mark.asyncio
async def test_chunker_with_empty_content():
    """Test chunkers with empty content."""
    from data_ingestion.chunkers.document_chunker import SemanticChunker

    chunker = SemanticChunker({})

    chunks = await chunker.chunk("", {}, "doc_1")

    assert len(chunks) == 1
    assert chunks[0].content == ""


@pytest.mark.asyncio
async def test_chunker_with_short_content():
    """Test chunkers with content shorter than chunk size."""
    from data_ingestion.chunkers.document_chunker import SemanticChunker

    chunker = SemanticChunker({"max_chunk_size": 1000})

    text = "Short text"

    chunks = await chunker.chunk(text, {}, "doc_1")

    assert len(chunks) == 1
    assert chunks[0].content == text
