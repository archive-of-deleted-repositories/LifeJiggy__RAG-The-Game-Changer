"""
Test RAG Pipeline - RAG-The-Game-Changer

Simple test to verify the implementation works.
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_functionality():
    """Test basic RAG functionality."""
    logger.info("Starting RAG pipeline test...")

    try:
        # Test imports
        logger.info("Testing imports...")
        from config import RAGPipeline, RAGConfig

        logger.info("✅ Imports successful")

        # Test pipeline initialization
        logger.info("Testing pipeline initialization...")
        pipeline = RAGPipeline()
        logger.info("✅ Pipeline created")

        # Test pipeline stats
        stats = await pipeline.get_stats()
        logger.info(f"✅ Pipeline stats: {stats}")

        # Test health check
        health = await pipeline.health_check()
        logger.info(f"✅ Health check: {health}")

        # Test document ingestion
        logger.info("Testing document ingestion...")
        test_docs = [
            {
                "content": "RAG stands for Retrieval-Augmented Generation. It combines retrieval systems with generative AI models to produce more accurate and grounded responses.",
                "metadata": {"title": "RAG Introduction", "source": "test_doc_1"},
            },
            {
                "content": "Vector databases store high-dimensional vectors and enable fast similarity search. Common examples include Pinecone, Weaviate, and FAISS.",
                "metadata": {"title": "Vector Databases", "source": "test_doc_2"},
            },
            {
                "content": "Embeddings are numerical representations of text that capture semantic meaning. They allow machines to understand text similarity and relationships.",
                "metadata": {"title": "Text Embeddings", "source": "test_doc_3"},
            },
        ]

        ingest_result = await pipeline.ingest(test_docs, chunk_strategy="semantic")
        logger.info(f"✅ Ingestion result: {ingest_result}")

        # Test querying
        logger.info("Testing query functionality...")
        test_queries = [
            "What is RAG?",
            "How do vector databases work?",
            "What are text embeddings?",
        ]

        for query in test_queries:
            response = await pipeline.query(
                query=query, top_k=3, include_sources=True, include_confidence=True
            )

            logger.info(f"✅ Query: {query}")
            logger.info(f"   Answer: {response.answer}")
            logger.info(f"   Confidence: {response.confidence}")
            logger.info(f"   Sources: {len(response.sources)}")
            logger.info(f"   Time: {response.total_time_ms:.2f}ms")
            logger.info("")

        logger.info("✅ All tests completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_components():
    """Test individual components."""
    logger.info("Testing individual components...")

    try:
        # Test embedding service
        logger.info("Testing embedding service...")
        from config.embedding_configs.embedding_service import create_embedding_service

        embedder = create_embedding_service("mock", {"dimensions": 384})
        result = await embedder.embed_texts(["Hello world", "RAG test"])
        logger.info(f"✅ Embedding service: {result.embeddings.shape}")

        # Test retriever
        logger.info("Testing retriever...")
        from retrieval_systems.dense_retriever import DenseRetriever

        retriever = DenseRetriever({"embedding_provider": "mock"})
        test_docs = [
            {"content": "This is a test document about AI", "document_id": "doc1"},
            {"content": "This is another test about machine learning", "document_id": "doc2"},
        ]
        await retriever.add_documents(test_docs)

        retrieval_result = await retriever.retrieve("What is AI?", top_k=2)
        logger.info(f"✅ Retrieval result: {len(retrieval_result.chunks)} chunks")

        # Test chunker
        logger.info("Testing chunker...")
        from data_ingestion.chunkers.document_chunker import create_chunker

        chunker = create_chunker("semantic", {"max_chunk_size": 200})
        test_content = "This is a long test document that should be split into multiple chunks. It contains several sentences and paragraphs to test the chunking functionality."
        chunks = await chunker.chunk(test_content, {"source": "test"}, "test_doc")
        logger.info(f"✅ Chunking result: {len(chunks)} chunks")

        logger.info("✅ Component tests completed!")
        return True

    except Exception as e:
        logger.error(f"❌ Component test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    logger.info("🚀 Starting RAG-The-Game-Changer Tests")
    logger.info("=" * 50)

    # Run tests
    test1 = await test_components()
    test2 = await test_basic_functionality()

    # Summary
    logger.info("=" * 50)
    if test1 and test2:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
