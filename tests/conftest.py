"""
Test Configuration - RAG-The-Game-Changer

Pytest fixtures and configuration for testing.
"""

import pytest
import asyncio
import numpy as np
from typing import List, Dict, Any


@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """Return sample documents for testing."""
    return [
        {
            "document_id": "doc_001",
            "content": "RAG systems combine retrieval and generation for better AI responses.",
            "metadata": {"source": "test", "title": "RAG Overview"},
        },
        {
            "document_id": "doc_002",
            "content": "Vector databases store embeddings for similarity search.",
            "metadata": {"source": "test", "title": "Vector DBs"},
        },
        {
            "document_id": "doc_003",
            "content": "Chunking strategies affect retrieval quality significantly.",
            "metadata": {"source": "test", "title": "Chunking"},
        },
    ]


@pytest.fixture
def sample_query() -> str:
    """Return sample query for testing."""
    return "How do RAG systems work?"


@pytest.fixture
def test_embeddings() -> np.ndarray:
    """Return sample embeddings for testing."""
    return np.random.randn(3, 384).astype(np.float32)


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Return mock configuration for testing."""
    return {
        "retrieval": {"top_k": 5, "similarity_threshold": 0.7},
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
    }
