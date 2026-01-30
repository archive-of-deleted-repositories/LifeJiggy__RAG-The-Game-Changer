"""
Basic Example - API Client

Simple example showing how to use the RAG API programmatically.
"""

import aiohttp
import asyncio
from typing import Dict, Any


class RAGClient:
    """Simple client for RAG API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        """Async context manager for session."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up session on exit."""
        if self.session:
            await self.session.close()

    async def ingest_document(
        self, content: str, metadata: Dict[str, Any] = None, chunk_strategy: str = "semantic"
    ) -> Dict[str, Any]:
        """Ingest a document into RAG system."""
        url = f"{self.base_url}/ingest"

        payload = {
            "documents": [{"content": content, "metadata": metadata or {}}],
            "chunk_strategy": chunk_strategy,
        }

        async with self as client:
            async with client.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Ingestion failed: {response.status} - {error_text}")

    async def query(
        self,
        question: str,
        top_k: int = 5,
        include_sources: bool = True,
        include_confidence: bool = True,
    ) -> Dict[str, Any]:
        """Query the RAG system."""
        url = f"{self.base_url}/query"

        payload = {
            "query": question,
            "top_k": top_k,
            "include_sources": include_sources,
            "include_confidence": include_confidence,
        }

        async with self as client:
            async with client.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Query failed: {response.status} - {error_text}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        url = f"{self.base_url}/stats"

        async with self as client:
            async with client.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Stats request failed: {response.status}")


async def main():
    """Run API client example."""
    print("RAG API Client Example")
    print("=" * 50)

    client = RAGClient("http://localhost:8000")

    try:
        async with client:
            # 1. Check health
            print("\n1. Checking health...")
            health = await client.get_stats()
            print(f"   Status: {health.get('status', 'unknown')}")

            # 2. Ingest document
            print("\n2. Ingesting document...")
            doc_content = """
            The transformer architecture, introduced in the 2017 paper 'Attention Is All You Need' by Vaswani et al., revolutionized natural language processing. It uses self-attention mechanisms to weigh the importance of different words in a sequence.
            
            Key features include:
            - Parallel computation: All positions in the sequence can be processed simultaneously
            - Long-range dependencies: Unlike RNNs, transformers can learn long-range dependencies
            - Scalability: Can handle very long sequences
            - Transfer learning: Pre-trained models can be fine-tuned for specific tasks
            """

            result = await client.ingest_document(
                content=doc_content,
                metadata={"title": "Transformers", "source": "example"},
                chunk_strategy="semantic",
            )

            print(f"   Document ID: {result.get('document_ids', ['N/A'])[0]}")
            print(f"   Chunks created: {result.get('total_chunks', 0)}")

            # 3. Query
            print("\n3. Querying RAG system...")
            query_result = await client.query(
                question="What is the transformer architecture?", top_k=5
            )

            print(f"   Answer: {query_result.get('answer', '')[:100]}")
            print(f"   Confidence: {query_result.get('confidence', 0):.2f}")
            print(f"   Sources retrieved: {len(query_result.get('sources', []))}")
            print(f"   Response time: {query_result.get('total_time_ms', 0):.2f}ms")

            # 4. Get stats
            print("\n4. Getting statistics...")
            stats = await client.get_stats()
            for key, value in stats.items():
                print(f"   {key}: {value}")

        print("\n" + "=" * 50)
        print("API client example completed!")

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
