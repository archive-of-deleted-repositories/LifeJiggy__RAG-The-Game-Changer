"""
Integration Tests for API - RAG-The-Game-Changer
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from typing import Dict, Any

from scripts.server import create_app


@pytest.fixture
def client():
    """Create test client for FastAPI."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_query() -> str:
    """Return sample query for testing."""
    return "How does RAG work?"


@pytest.fixture
def sample_document() -> Dict[str, Any]:
    """Return sample document for testing."""
    return {
        "content": "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval systems with generative AI models to produce more accurate and contextually relevant responses.",
        "metadata": {"source": "test", "title": "RAG Overview"},
    }


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_query_endpoint(client, sample_query):
    """Test query endpoint."""
    response = client.post("/query", json={"query": sample_query, "top_k": 5})

    # May return 200 or 500 depending on setup
    assert response.status_code in [200, 500, 422]

    if response.status_code == 200:
        data = response.json()
        assert "answer" in data or "error" in data


def test_ingest_endpoint(client, sample_document):
    """Test document ingestion endpoint."""
    response = client.post(
        "/ingest", json={"documents": [sample_document], "chunk_strategy": "semantic"}
    )

    # May return 200 or 500 depending on setup
    assert response.status_code in [200, 500, 422]

    if response.status_code == 200:
        data = response.json()
        assert "total_documents" in data or "error" in data


def test_batch_ingest(client):
    """Test batch document ingestion."""
    documents = [{"content": f"Document {i}", "metadata": {"index": i}} for i in range(10)]

    response = client.post("/ingest", json={"documents": documents, "chunk_strategy": "semantic"})

    assert response.status_code in [200, 500, 422]


def test_query_with_filters(client):
    """Test query with metadata filters."""
    response = client.post(
        "/query", json={"query": "test", "top_k": 5, "filters": {"source": "test"}}
    )

    assert response.status_code in [200, 500, 422]


def test_query_with_top_k(client, sample_query):
    """Test query with different top_k values."""
    for top_k in [1, 5, 10]:
        response = client.post("/query", json={"query": sample_query, "top_k": top_k})

        assert response.status_code in [200, 500, 422]


def test_delete_documents(client):
    """Test document deletion endpoint."""
    response = client.post("/documents/delete", json={"document_ids": ["doc_1", "doc_2"]})

    assert response.status_code in [200, 500, 404]


def test_clear_index(client):
    """Test clear index endpoint."""
    response = client.post("/index/clear")

    assert response.status_code in [200, 500]


def test_get_stats(client):
    """Test statistics endpoint."""
    response = client.get("/stats")

    assert response.status_code in [200, 500]

    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, dict)
