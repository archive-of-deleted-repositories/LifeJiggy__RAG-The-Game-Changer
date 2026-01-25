"""
RAG CLI - RAG-The-Game-Changer

Command-line interface for the RAG system.
"""

import asyncio
import argparse
import sys
from typing import List, Dict, Any
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def ingest_command(args):
    """Handle document ingestion."""
    try:
        from config import RAGPipeline

        # Initialize pipeline
        pipeline = RAGPipeline(
            retrieval_strategy=args.strategy,
            embedding_provider=args.embedding_provider,
            vector_db=args.vector_db,
        )

        # Load documents
        documents = []
        for file_path in args.files:
            try:
                if file_path.endswith(".json"):
                    # Load from JSON file
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_docs = json.load(f)
                        if isinstance(file_docs, list):
                            documents.extend(file_docs)
                        else:
                            documents.append(file_docs)
                else:
                    # Load as text file
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        documents.append({"content": content, "metadata": {"source": file_path}})
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
                continue

        if not documents:
            logger.error("No documents to ingest")
            return False

        # Ingest documents
        logger.info(f"Ingesting {len(documents)} documents...")
        result = await pipeline.ingest(documents, chunk_strategy=args.chunk_strategy)

        print(f"✅ Ingestion completed:")
        print(f"   Documents processed: {result['documents_processed']}")
        print(f"   Processing time: {result['processing_time_seconds']:.2f}s")

        return True

    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        return False


async def query_command(args):
    """Handle querying."""
    try:
        from config import RAGPipeline

        # Initialize pipeline
        pipeline = RAGPipeline(
            retrieval_strategy=args.strategy,
            embedding_provider=args.embedding_provider,
            vector_db=args.vector_db,
        )

        # Execute query
        logger.info(f"Processing query: {args.query}")
        response = await pipeline.query(
            query=args.query,
            top_k=args.top_k,
            include_sources=args.sources,
            include_confidence=True,
        )

        # Display results
        print(f"\n🔍 Query: {response.query}")
        print(f"\n💡 Answer: {response.answer}")
        print(f"\n📊 Confidence: {response.confidence:.2f}")
        print(f"⏱️  Total time: {response.total_time_ms:.2f}ms")
        print(f"🔎 Retrieval time: {response.retrieval_time_ms:.2f}ms")
        print(f"🤖 Generation time: {response.generation_time_ms:.2f}ms")

        if response.sources and args.sources:
            print(f"\n📚 Sources ({len(response.sources)}):")
            for i, source in enumerate(response.sources, 1):
                title = source.get("title", "Unknown")
                score = source.get("score", 0.0)
                print(f"   {i}. {title} (score: {score:.3f})")

        return True

    except Exception as e:
        logger.error(f"Error during query: {e}")
        return False


async def stats_command(args):
    """Handle stats command."""
    try:
        from config import RAGPipeline

        # Initialize pipeline
        pipeline = RAGPipeline(
            retrieval_strategy=args.strategy,
            embedding_provider=args.embedding_provider,
            vector_db=args.vector_db,
        )

        # Get stats
        stats = await pipeline.get_stats()
        health = await pipeline.health_check()

        print("📊 RAG Pipeline Statistics:")
        print(json.dumps(stats, indent=2))

        print("\n🏥 Health Check:")
        print(json.dumps(health, indent=2))

        return True

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return False


async def interactive_command(args):
    """Handle interactive mode."""
    try:
        from config import RAGPipeline

        # Initialize pipeline
        pipeline = RAGPipeline(
            retrieval_strategy=args.strategy,
            embedding_provider=args.embedding_provider,
            vector_db=args.vector_db,
        )

        print("🚀 RAG Interactive Mode")
        print("Type 'quit' or 'exit' to leave")
        print("-" * 50)

        while True:
            try:
                query = input("\n🔍 Enter your query: ").strip()

                if query.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                if not query:
                    continue

                # Process query
                response = await pipeline.query(
                    query=query, top_k=args.top_k, include_sources=True, include_confidence=True
                )

                print(f"\n💡 Answer: {response.answer}")
                print(f"📊 Confidence: {response.confidence:.2f}")
                print(f"⏱️  Time: {response.total_time_ms:.2f}ms")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")

        return True

    except Exception as e:
        logger.error(f"Error starting interactive mode: {e}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG-The-Game-Changer: Production-Ready RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest documents
  rag-cli ingest --files doc1.txt doc2.pdf --strategy hybrid
  
  # Query the system
  rag-cli query "What is RAG?" --top-k 5 --sources
  
  # Interactive mode
  rag-cli interactive --strategy hybrid --top-k 3
  
  # Get statistics
  rag-cli stats
        """,
    )

    # Global arguments
    parser.add_argument(
        "--strategy",
        choices=["dense", "sparse", "hybrid"],
        default="hybrid",
        help="Retrieval strategy",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["openai", "sentence-transformers"],
        default="openai",
        help="Embedding provider",
    )
    parser.add_argument(
        "--vector-db",
        choices=["faiss", "pinecone", "chroma"],
        default="faiss",
        help="Vector database",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("files", nargs="+", help="Document files to ingest")
    ingest_parser.add_argument(
        "--chunk-strategy",
        choices=["semantic", "token", "fixed"],
        default="semantic",
        help="Chunking strategy",
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("query", help="Query string")
    query_parser.add_argument(
        "--top-k", "-k", type=int, default=5, help="Number of documents to retrieve"
    )
    query_parser.add_argument(
        "--sources", "-s", action="store_true", help="Include source information"
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show system statistics")

    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive query mode")
    interactive_parser.add_argument(
        "--top-k", "-k", type=int, default=3, help="Number of documents to retrieve"
    )

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle commands
    if args.command == "ingest":
        success = asyncio.run(ingest_command(args))
    elif args.command == "query":
        success = asyncio.run(query_command(args))
    elif args.command == "stats":
        success = asyncio.run(stats_command(args))
    elif args.command == "interactive":
        success = asyncio.run(interactive_command(args))
    else:
        parser.print_help()
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
