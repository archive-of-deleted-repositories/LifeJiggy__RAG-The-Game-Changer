"""
Advanced Example - Multi-Hop RAG

Demonstrating multi-hop RAG for complex reasoning queries.
"""

import asyncio
from advanced_rag_patterns.multi_hop_rag import MultiHopRAG
from config.pipeline_configs.rag_pipeline import RAGPipeline


async def main():
    """Run multi-hop RAG example."""
    print("Multi-Hop RAG Example")
    print("=" * 50)

    # Initialize multi-hop RAG
    multi_hop = MultiHopRAG({"max_hops": 3, "hop_threshold": 0.7, "early_stop": True})

    # Sample documents with interconnected information
    documents = [
        {
            "document_id": "doc_001",
            "content": "Albert Einstein was born in Ulm, Germany. He developed the theory of relativity and received the Nobel Prize in Physics in 1921.",
            "metadata": {
                "title": "Albert Einstein",
                "source": "biography",
                "birth_year": 1879,
                "country": "Germany",
            },
        },
        {
            "document_id": "doc_002",
            "content": "Einstein moved to Switzerland in 1896. He worked at the Swiss Patent Office in Bern.",
            "metadata": {
                "title": "Einstein in Switzerland",
                "source": "biography",
                "year": 1896,
                "country": "Switzerland",
            },
        },
        {
            "document_id": "doc_003",
            "content": "In 1902, Einstein obtained a position at the Swiss Federal Institute of Intellectual Property in Bern.",
            "metadata": {
                "title": "Einstein at Swiss Institute",
                "source": "biography",
                "year": 1902,
                "country": "Switzerland",
            },
        },
        {
            "document_id": "doc_004",
            "content": "In 1914, Einstein moved to Berlin to join the Royal Prussian Academy of Sciences.",
            "metadata": {
                "title": "Einstein in Berlin",
                "source": "biography",
                "year": 1914,
                "country": "Germany",
            },
        },
        {
            "document_id": "doc_005",
            "content": "Einstein became a member of the Academy of Sciences in Berlin. He published over 300 scientific papers.",
            "metadata": {
                "title": "Einstein Publications",
                "source": "biography",
                "year": 1915,
                "country": "Germany",
            },
        },
    ]

    # Ingest documents
    print("\n1. Ingesting documents...")
    await multi_hop.initialize(documents)
    print(f"   Ingested {len(documents)} documents with multi-hop connections")

    # Complex queries requiring multiple hops
    queries = [
        "Where was Albert Einstein born?",
        "What countries did Einstein live in?",
        "What position did Einstein have in Switzerland?",
        "Where did Einstein move in 1914?",
        "What academy did Einstein join in 1914?",
        "How many scientific papers did Einstein publish?",
    ]

    # Process multi-hop queries
    print("\n2. Processing multi-hop queries...")
    for i, query in enumerate(queries, 1):
        print(f"\n   Query {i}: {query}")
        print(f"   {'-' * 50}")

        result = await multi_hop.query(query=query, top_k=5, max_hops=3)

        print(f"   Final Answer: {result.final_answer}")
        print(f"   Hops Used: {result.hop_count}")
        print(f"   Reasoning Trace:")

        for j, hop in enumerate(result.hops, 1):
            print(f"      Hop {j + 1}:")
            print(f"        Question: {hop.question}")
            print(f"        Answer: {hop.answer}")
            print(f"        Source: {hop.source_document}")
            print(f"        Confidence: {hop.confidence:.2%}")

        print(f"   Total Time: {result.total_time_ms:.2f}ms")
        print(f"   Confidence: {result.confidence:.2%}")
        print()

    # Simple query comparison
    print("\n3. Comparing single-hop vs multi-hop:")
    simple_query = "Where was Einstein born?"

    print("Single-hop query:")
    result = await multi_hop.query(query=simple_query, top_k=5, max_hops=1)
    print(f"  Answer: {result.final_answer}")
    print(f"  Time: {result.total_time_ms:.2f}ms")

    print("\n" + "=" * 50)
    print("Multi-hop RAG example completed!")
    print("\nKey Benefits:")
    print("  - Handles complex, multi-part questions")
    print("  - Chains reasoning across multiple documents")
    print("  - Maintains traceability of reasoning process")
    print("  - Improves accuracy for factual queries")


if __name__ == "__main__":
    asyncio.run(main())
