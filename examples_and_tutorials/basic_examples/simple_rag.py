"""
Simple RAG Example - RAG-The-Game-Changer

This example demonstrates the basic usage of the RAG pipeline.
"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main example function."""
    logger.info("Starting RAG-The-Game-Changer example...")

    try:
        from config import RAGPipeline, RAGConfig

        config = RAGConfig(name="RAG-The-Game-Changer", version="0.1.0", environment="development")

        pipeline = RAGPipeline(
            config=config.to_dict(),
            retrieval_strategy="hybrid",
            embedding_provider="openai",
            llm_provider="openai",
            vector_db="pinecone",
        )

        logger.info("RAG Pipeline initialized successfully")

        stats = pipeline.get_stats()
        logger.info(f"Pipeline stats: {stats}")

        sample_documents = [
            {
                "content": """
                Retrieval-Augmented Generation (RAG) is a technique that combines 
                retrieval-based methods with generative models. RAG helps reduce 
                hallucinations in Large Language Models by grounding responses 
                in retrieved evidence from knowledge bases.
                
                Key benefits of RAG include:
                - Improved factual accuracy
                - Access to up-to-date information
                - Source attribution and citations
                - Reduced hallucination rates
                """,
                "metadata": {"title": "Introduction to RAG", "source": "doc_001", "type": "text"},
            },
            {
                "content": """
                Hybrid retrieval combines dense and sparse retrieval methods.
                Dense retrieval uses embedding similarity to find semantically 
                similar documents, while sparse retrieval uses keyword-based 
                methods like BM25. Combining both approaches provides better 
                coverage and accuracy.
                
                Cross-encoder reranking can further improve results by 
                re-scoring retrieved documents using a more computationally 
                intensive but accurate cross-encoder model.
                """,
                "metadata": {
                    "title": "Hybrid Retrieval Methods",
                    "source": "doc_002",
                    "type": "text",
                },
            },
        ]

        logger.info(f"Ingesting {len(sample_documents)} documents...")
        ingest_result = await pipeline.ingest(documents=sample_documents, chunk_strategy="semantic")
        logger.info(f"Ingestion result: {ingest_result}")

        test_queries = [
            "What is RAG and how does it reduce hallucinations?",
            "What are the benefits of hybrid retrieval?",
        ]

        for query in test_queries:
            logger.info(f"\nProcessing query: {query}")

            response = await pipeline.query(
                query=query, top_k=3, include_sources=True, include_confidence=True
            )

            logger.info(f"\nQuery: {response.query}")
            logger.info(f"Answer: {response.answer}")
            logger.info(f"Confidence: {response.confidence}")
            logger.info(f"Sources: {len(response.sources)}")
            logger.info(f"Total time: {response.total_time_ms:.2f}ms")

            if response.sources:
                logger.info("\nSource details:")
                for i, source in enumerate(response.sources, 1):
                    logger.info(f"  {i}. {source.get('title', 'Unknown')}")

        logger.info("\nExample completed successfully!")

        return True

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Some components may not be available. Install dependencies with:")
        logger.info("pip install -r requirements.txt")
        return False
    except Exception as e:
        logger.error(f"Error during example execution: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
