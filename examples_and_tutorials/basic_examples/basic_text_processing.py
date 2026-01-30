"""
Basic Example - Text Processing

Simple example demonstrating basic RAG usage for text documents.
"""

import asyncio
from config.pipeline_configs.rag_pipeline import RAGPipeline


async def main():
    """Run basic text processing example."""
    print("Basic Text Processing Example")
    print("=" * 50)

    # Initialize RAG pipeline
    pipeline = RAGPipeline(
        {"retrieval": {"strategy": "hybrid", "top_k": 5}, "generation": {"min_confidence": 0.7}}
    )

    # Sample documents
    documents = [
        {
            "document_id": "doc_001",
            "content": "RAG (Retrieval-Augmented Generation) is a technique in natural language processing (NLP) that retrieves relevant information from external sources and uses it to generate more accurate and contextually appropriate responses.",
            "metadata": {"title": "RAG Overview", "source": "example", "topic": "NLP"},
        },
        {
            "document_id": "doc_002",
            "content": "Vector databases are specialized databases designed to store, manage, and query vector embeddings. They enable efficient similarity search and retrieval in high-dimensional vector spaces, which is crucial for RAG systems.",
            "metadata": {"title": "Vector Databases", "source": "example", "topic": "Databases"},
        },
        {
            "document_id": "doc_003",
            "content": "Embeddings are numerical representations of text that capture semantic meaning. Modern embedding models like BERT, GPT, and sentence-transformers produce dense vector representations that can be used for semantic similarity search.",
            "metadata": {"title": "Embeddings", "source": "example", "topic": "ML"},
        },
        {
            "document_id": "doc_004",
            "content": "Chunking strategies for documents include fixed-size chunking (splitting text into equal-sized pieces), semantic chunking (splitting at natural boundaries like sentences or paragraphs), and token-based chunking (splitting based on token count for language models).",
            "metadata": {
                "title": "Chunking Strategies",
                "source": "example",
                "topic": "Text Processing",
            },
        },
    ]

    # Ingest documents
    print("\n1. Ingesting documents...")
    ingestion_result = await pipeline.ingest(documents, chunk_strategy="semantic")
    print(
        f"   Ingested {ingestion_result['successful']}/{ingestion_result['total_documents']} documents"
    )
    print(f"   Created {ingestion_result['total_chunks']} chunks")

    # Sample queries
    queries = [
        "What is RAG?",
        "How do vector databases work?",
        "What are embeddings?",
        "What are chunking strategies?",
        "How does semantic chunking work?",
    ]

    # Process queries
    print("\n2. Processing queries...")
    for i, query in enumerate(queries, 1):
        print(f"   Query {i}: {query}")

        result = await pipeline.query(
            query=query, top_k=5, include_sources=True, include_confidence=True
        )

        print(f"   Answer: {result.answer[:100]}...")
        print(f"   Confidence: {result.confidence:.2%}")
        print(f"   Sources: {len(result.sources)} retrieved")
        print(f"   Time: {result.total_time_ms:.2f}ms")
        print()

    # Get pipeline stats
    print("\n3. Pipeline Statistics:")
    stats = pipeline.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 50)
    print("Example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
