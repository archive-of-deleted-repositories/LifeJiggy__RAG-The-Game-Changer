"""
Unit tests for Retrieval Systems - RAG-The-Game-Changer
"""

import pytest
import asyncio
import numpy as np
from typing import Dict, Any

from retrieval_systems.dense_retriever import DenseRetriever
from retrieval_systems.sparse_retriever import SparseRetriever
from retrieval_systems.hybrid_retriever import HybridRetriever
from retrieval_systems.base import RetrievalStrategy


@pytest.mark.asyncio
async def test_dense_retriever_initialization():
    """Test DenseRetriever initialization."""
    config = {"embedding_provider": "openai", "top_k": 5}

    retriever = DenseRetriever(config)

    assert retriever.embedding_provider == "openai"
    assert retriever.top_k == 5


@pytest.mark.asyncio
async def test_sparse_retriever_initialization():
    """Test SparseRetriever initialization."""
    config = {"top_k": 10}

    retriever = SparseRetriever(config)

    assert retriever.top_k == 10


@pytest.mark.asyncio
async def test_hybrid_retriever_initialization():
    """Test HybridRetriever initialization."""
    config = {"dense_weight": 0.7, "sparse_weight": 0.3}

    retriever = HybridRetriever(config)

    assert retriever.dense_weight == 0.7
    assert retriever.sparse_weight == 0.3


@pytest.mark.asyncio
async def test_dense_retriever_add_documents():
    """Test adding documents to DenseRetriever."""
    config = {"top_k": 5}
    retriever = DenseRetriever(config)

    documents = [
        {
            "content": "RAG systems combine retrieval and generation.",
            "document_id": "doc_1",
            "metadata": {"source": "test"},
        },
        {
            "content": "Vector databases store embeddings.",
            "document_id": "doc_2",
            "metadata": {"source": "test"},
        },
    ]

    result = await retriever.add_documents(documents)

    assert result is True
    assert len(retriever._documents) == 2


@pytest.mark.asyncio
async def test_sparse_retriever_retrieve():
    """Test retrieval from SparseRetriever."""
    config = {"top_k": 5}
    retriever = SparseRetriever(config)

    documents = [
        {"content": "RAG is a powerful AI technique.", "document_id": "doc_1", "metadata": {}}
    ]

    await retriever.add_documents(documents)
    result = await retriever.retrieve("RAG AI", top_k=5)

    assert result.strategy == RetrievalStrategy.SPARSE
    assert len(result.chunks) >= 0


@pytest.mark.asyncio
async def test_hybrid_retriever_retrieve():
    """Test retrieval from HybridRetriever."""
    config = {"top_k": 5}
    retriever = HybridRetriever(config)

    documents = [
        {
            "content": "Hybrid search combines dense and sparse retrieval.",
            "document_id": "doc_1",
            "metadata": {},
        }
    ]

    await retriever.add_documents(documents)
    result = await retriever.retrieve("hybrid search", top_k=5)

    assert result.strategy == RetrievalStrategy.HYBRID


@pytest.mark.asyncio
async def test_dense_retriever_delete():
    """Test deleting documents from DenseRetriever."""
    config = {"top_k": 5}
    retriever = DenseRetriever(config)

    documents = [{"content": "Document to delete", "document_id": "doc_to_delete", "metadata": {}}]

    await retriever.add_documents(documents)
    result = await retriever.delete_documents(["doc_to_delete"])

    assert result is True


@pytest.mark.asyncio
async def test_retriever_clear():
    """Test clearing all documents."""
    config = {"top_k": 5}
    retriever = DenseRetriever(config)

    documents = [{"content": "Document to clear", "document_id": "doc_1", "metadata": {}}]

    await retriever.add_documents(documents)
    result = await retriever.clear()

    assert result is True
    assert len(retriever._documents) == 0
