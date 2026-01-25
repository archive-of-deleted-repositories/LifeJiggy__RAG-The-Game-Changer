"""
Retrieval Augmented Generation - Advanced RAG Pattern

Base class for advanced RAG implementations.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

from ..config.pipeline_configs.rag_pipeline import RAGPipeline, RAGResponse

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for advanced RAG patterns."""

    max_context_length: int = 4000
    min_relevance_score: float = 0.5
    enable_reranking: bool = True
    enable_filtering: bool = True


class RetrievalAugmentedGeneration:
    """Base class for advanced RAG patterns."""

    def __init__(self, base_pipeline: RAGPipeline, config: Optional[Dict[str, Any]] = None):
        self.pipeline = base_pipeline
        self.config = RAGConfig(**(config or {}))

        # Advanced settings
        self.enable_contextual_ranking = config.get("enable_contextual_ranking", True)
        self.enable_query_transformation = config.get("enable_query_transformation", True)
        self.enable_response_refinement = config.get("enable_response_refinement", True)

    async def query(
        self, query: str, context: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Base query method - can be overridden by subclasses."""

        # Transform query if enabled
        if self.enable_query_transformation:
            transformed_query = await self._transform_query(query, context)
        else:
            transformed_query = query

        # Execute standard RAG query
        response = await self.pipeline.query(
            query=transformed_query, top_k=top_k or 5, include_sources=True, include_confidence=True
        )

        # Refine response if enabled
        if self.enable_response_refinement:
            refined_answer = await self._refine_response(
                response.answer, query, response.sources, context
            )
            response.answer = refined_answer

        return {
            "query": query,
            "transformed_query": transformed_query,
            "answer": response.answer,
            "sources": response.sources,
            "confidence": response.confidence,
            "metadata": {
                "context": context or {},
                "transformation_applied": self.enable_query_transformation,
                "refinement_applied": self.enable_response_refinement,
            },
            "response_time_ms": response.total_time_ms,
        }

    async def _transform_query(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Transform query based on context."""
        # Basic transformation - can be enhanced with LLM
        if not context:
            return query

        # Add context hints
        context_hints = []
        if "domain" in context:
            context_hints.append(f"in the domain of {context['domain']}")
        if "recent_queries" in context:
            context_hints.append(f"related to: {', '.join(context['recent_queries'][-2:])}")

        if context_hints:
            return f"{query} (context: {'; '.join(context_hints)})"

        return query

    async def _refine_response(
        self,
        answer: str,
        query: str,
        sources: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Refine the generated response."""
        # Basic refinement - add citations
        if self.config.enable_reranking and sources:
            citations = self._generate_citations(sources)
            if citations:
                return f"{answer}\n\nReferences:\n{citations}"

        return answer

    def _generate_citations(self, sources: List[Dict[str, Any]]) -> str:
        """Generate citations from sources."""
        citations = []
        for i, source in enumerate(sources[:5], 1):  # Top 5 sources
            title = source.get("title", "Unknown Source")
            citations.append(f"[{i}] {title}")

        return "\n".join(citations)

    async def batch_query(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple queries."""
        results = []

        for query_data in queries:
            result = await self.query(
                query=query_data["query"],
                context=query_data.get("context"),
                top_k=query_data.get("top_k"),
            )
            results.append(result)

        return results

    async def evaluate_performance(self, test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate RAG performance."""
        results = await self.batch_query(test_queries)

        # Calculate performance metrics
        latencies = [r["response_time_ms"] for r in results]
        confidences = [r["confidence"] for r in results]

        return {
            "total_queries": len(results),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "avg_confidence": sum(confidences) / len(confidences),
            "success_rate": len([r for r in results if r["confidence"] > 0.5]) / len(results),
        }
