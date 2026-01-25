"""
Multi-Hop RAG - RAG-The-Game-Changer

Advanced RAG pattern for complex queries requiring multiple retrieval steps.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time

from ..config.pipeline_configs.rag_pipeline import RAGPipeline, RAGResponse

logger = logging.getLogger(__name__)


@dataclass
class HopResult:
    """Result from a single retrieval hop."""

    hop_number: int
    query: str
    retrieved_chunks: List[Any]
    answer: str
    confidence: float
    next_query: Optional[str] = None
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiHopResponse:
    """Complete multi-hop response."""

    original_query: str
    hops: List[HopResult]
    final_answer: str
    total_confidence: float
    reasoning_path: List[str]
    all_sources: List[Dict[str, Any]]
    total_time_ms: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiHopRAG:
    """Advanced RAG pattern for complex, multi-step reasoning queries."""

    def __init__(self, base_pipeline: RAGPipeline, config: Optional[Dict[str, Any]] = None):
        self.pipeline = base_pipeline
        self.config = config or {}

        # Multi-hop settings
        self.max_hops = self.config.get("max_hops", 3)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.hop_timeout = self.config.get("hop_timeout", 30)
        self.use_decomposition = self.config.get("use_decomposition", True)

        # Query transformation settings
        self.use_query_planning = self.config.get("use_query_planning", True)
        self.use_hallucination_detection = self.config.get("use_hallucination_detection", True)

        # Reasoning settings
        self.require_reasoning_path = self.config.get("require_reasoning_path", True)
        self.reasoning_model = self.config.get("reasoning_model", "gpt-4")

    async def query(
        self, query: str, max_hops: Optional[int] = None, require_reasoning: bool = True
    ) -> MultiHopResponse:
        """Process multi-hop query."""
        start_time = time.time()

        try:
            # Analyze query complexity and plan hops
            hop_plan = await self._plan_hops(query, max_hops or self.max_hops)

            # Execute hops sequentially
            hops = []
            current_query = query
            accumulated_context = []

            for hop_num in range(len(hop_plan)):
                logger.info(f"Executing hop {hop_num + 1}/{len(hop_plan)}: {hop_plan[hop_num]}")

                # Execute single hop
                hop_result = await self._execute_hop(
                    hop_plan[hop_num], hop_num + 1, accumulated_context, hops
                )

                hops.append(hop_result)
                accumulated_context.extend(hop_result.retrieved_chunks)

                # Check if we have enough information
                if hop_result.confidence >= self.confidence_threshold:
                    if hop_num < len(hop_plan) - 1:
                        # Continue to next hop
                        current_query = hop_result.next_query or hop_plan[hop_num + 1]
                    else:
                        # Final hop reached
                        break
                else:
                    logger.warning(f"Hop {hop_num + 1} confidence too low: {hop_result.confidence}")
                    break

            # Synthesize final answer
            final_answer, total_confidence = await self._synthesize_final_answer(
                query, hops, accumulated_context
            )

            # Generate reasoning path
            reasoning_path = [hop.reasoning for hop in hops if hop.reasoning]

            # Collect all sources
            all_sources = []
            for hop in hops:
                all_sources.extend(hop.metadata.get("sources", []))

            total_time = (time.time() - start_time) * 1000

            # Detect hallucinations if enabled
            success = True
            if self.use_hallucination_detection:
                success = await self._detect_hallucinations(query, final_answer, all_sources)

            return MultiHopResponse(
                original_query=query,
                hops=hops,
                final_answer=final_answer,
                total_confidence=total_confidence,
                reasoning_path=reasoning_path,
                all_sources=self._deduplicate_sources(all_sources),
                total_time_ms=total_time,
                success=success,
                metadata={
                    "planned_hops": len(hop_plan),
                    "executed_hops": len(hops),
                    "average_hop_confidence": sum(h.confidence for h in hops) / len(hops)
                    if hops
                    else 0,
                },
            )

        except Exception as e:
            logger.error(f"Error in multi-hop query: {e}")
            return MultiHopResponse(
                original_query=query,
                hops=[],
                final_answer=f"Error processing multi-hop query: {str(e)}",
                total_confidence=0.0,
                reasoning_path=[],
                all_sources=[],
                total_time_ms=(time.time() - start_time) * 1000,
                success=False,
            )

    async def _plan_hops(self, query: str, max_hops: int) -> List[str]:
        """Plan the sequence of queries for multi-hop retrieval."""
        if not self.use_query_planning:
            # Simple approach: use original query for all hops
            return [query] * min(2, max_hops)

        try:
            # Use LLM to decompose complex query
            planning_prompt = f"""Given the following complex query, break it down into a sequence of simpler queries that need to be answered to fully address the original query.

Original query: {query}

Please provide up to {max_hops} queries in order, each building on the previous ones. Focus on:
1. What information is needed first
2. What additional information is needed next
3. What final question ties everything together

Return only the queries, one per line:"""

            from openai import OpenAI

            client = OpenAI()

            response = client.chat.completions.create(
                model=self.reasoning_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at breaking down complex questions into sequential, simpler questions.",
                    },
                    {"role": "user", "content": planning_prompt},
                ],
                temperature=0.1,
                max_tokens=300,
            )

            planned_queries = [
                line.strip()
                for line in response.choices[0].message.content.split("\n")
                if line.strip()
            ]

            # Ensure we don't exceed max_hops
            return planned_queries[:max_hops]

        except Exception as e:
            logger.warning(f"Failed to plan hops: {e}")
            return [query] * min(2, max_hops)

    async def _execute_hop(
        self,
        query: str,
        hop_number: int,
        previous_context: List[Any],
        previous_hops: List[HopResult],
    ) -> HopResult:
        """Execute a single retrieval hop."""
        try:
            # Retrieve relevant information
            response = await self.pipeline.query(query=query, top_k=5, include_sources=True)

            # Generate reasoning for this hop
            reasoning = await self._generate_hop_reasoning(
                query, response.answer, response.sources, hop_number, previous_hops
            )

            # Plan next query if needed
            next_query = await self._plan_next_query(
                query, response.answer, hop_number, previous_hops
            )

            return HopResult(
                hop_number=hop_number,
                query=query,
                retrieved_chunks=response.metadata.get("retrieved_chunks", []),
                answer=response.answer,
                confidence=response.confidence,
                next_query=next_query,
                reasoning=reasoning,
                metadata={
                    "sources": response.sources,
                    "retrieval_time_ms": response.retrieval_time_ms,
                    "generation_time_ms": response.generation_time_ms,
                },
            )

        except Exception as e:
            logger.error(f"Error executing hop {hop_number}: {e}")
            return HopResult(
                hop_number=hop_number,
                query=query,
                retrieved_chunks=[],
                answer=f"Error in hop {hop_number}: {str(e)}",
                confidence=0.0,
                reasoning=f"Failed to execute hop: {str(e)}",
            )

    async def _generate_hop_reasoning(
        self,
        query: str,
        answer: str,
        sources: List[Dict[str, Any]],
        hop_number: int,
        previous_hops: List[HopResult],
    ) -> str:
        """Generate reasoning for a hop."""
        try:
            previous_reasoning = " | ".join([h.reasoning for h in previous_hops if h.reasoning])

            reasoning_prompt = f"""Explain the reasoning for answering this query in a multi-step process.

Hop {hop_number} Query: {query}

Previous reasoning: {previous_reasoning if previous_reasoning else "None"}

Found information: {answer}

Provide a brief explanation of how this information helps answer the overall question:"""

            from openai import OpenAI

            client = OpenAI()

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a reasoning assistant that explains step-by-step thinking.",
                    },
                    {"role": "user", "content": reasoning_prompt},
                ],
                temperature=0.1,
                max_tokens=150,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"Failed to generate reasoning for hop {hop_number}: {e}")
            return f"Retrieved information to answer: {query}"

    async def _plan_next_query(
        self,
        current_query: str,
        current_answer: str,
        hop_number: int,
        previous_hops: List[HopResult],
    ) -> Optional[str]:
        """Plan the next query in the sequence."""
        if hop_number >= self.max_hops:
            return None

        try:
            context = " | ".join(
                [
                    f"Q{i + 1}: {h.query} -> A{i + 1}: {h.answer}"
                    for i, h in enumerate(
                        previous_hops
                        + [type("", (), {"query": current_query, "answer": current_answer})()]
                    )
                ]
            )

            planning_prompt = f"""Given the current state of a multi-hop reasoning process, determine the next query needed.

Current context:
{context}

What is the next most important question to ask to reach the final answer?
If this is sufficient for the final answer, respond with "SUFFICIENT".
Otherwise, provide the next specific question:"""

            from openai import OpenAI

            client = OpenAI()

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a planning assistant for multi-step reasoning.",
                    },
                    {"role": "user", "content": planning_prompt},
                ],
                temperature=0.1,
                max_tokens=100,
            )

            result = response.choices[0].message.content.strip()

            if result.upper() == "SUFFICIENT":
                return None

            return result

        except Exception as e:
            logger.warning(f"Failed to plan next query: {e}")
            return None

    async def _synthesize_final_answer(
        self, original_query: str, hops: List[HopResult], context: List[Any]
    ) -> Tuple[str, float]:
        """Synthesize final answer from all hops."""
        if not hops:
            return "No information could be retrieved to answer the query.", 0.0

        try:
            # Build synthesis prompt
            hop_summaries = "\n".join(
                [
                    f"Step {i + 1}: {h.query} -> {h.answer} (confidence: {h.confidence:.2f})"
                    for i, h in enumerate(hops)
                ]
            )

            synthesis_prompt = f"""Based on the following multi-step reasoning process, provide a comprehensive answer to the original question.

Original Question: {original_query}

Multi-step Process:
{hop_summaries}

Synthesize a complete answer that addresses the original question using all the information gathered:"""

            from openai import OpenAI

            client = OpenAI()

            response = client.chat.completions.create(
                model=self.reasoning_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a synthesis expert that combines multi-step reasoning into comprehensive answers.",
                    },
                    {"role": "user", "content": synthesis_prompt},
                ],
                temperature=0.1,
                max_tokens=800,
            )

            final_answer = response.choices[0].message.content.strip()

            # Calculate total confidence
            if hops:
                total_confidence = sum(h.confidence for h in hops) / len(hops)
                # Boost confidence if multiple hops agree
                if len(hops) > 1:
                    avg_conf = total_confidence
                    min_conf = min(h.confidence for h in hops)
                    max_conf = max(h.confidence for h in hops)

                    # Reduce penalty for outlier low confidence
                    if max_conf - min_conf < 0.3:
                        total_confidence = avg_conf * 1.1
                    total_confidence = min(total_confidence, 1.0)
            else:
                total_confidence = 0.0

            return final_answer, total_confidence

        except Exception as e:
            logger.error(f"Failed to synthesize final answer: {e}")
            if hops:
                return hops[-1].answer, hops[-1].confidence
            return "Failed to synthesize answer", 0.0

    async def _detect_hallucinations(
        self, query: str, answer: str, sources: List[Dict[str, Any]]
    ) -> bool:
        """Detect potential hallucinations in the answer."""
        try:
            # Extract source content
            source_texts = [
                source.get("content", "")[:500]  # First 500 chars
                for source in sources[:5]  # Top 5 sources
            ]
            combined_sources = "\n".join(filter(None, source_texts))

            if not combined_sources.strip():
                # No sources, assume it might be hallucination
                return False

            detection_prompt = f"""Check if the following answer is supported by the provided sources.

Question: {query}

Answer: {answer}

Sources:
{combined_sources}

Respond with TRUE if the answer is well-supported by the sources, FALSE if it contains significant unsupported claims:"""

            from openai import OpenAI

            client = OpenAI()

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a fact-checker that determines if answers are supported by provided sources.",
                    },
                    {"role": "user", "content": detection_prompt},
                ],
                temperature=0.1,
                max_tokens=10,
            )

            result = response.choices[0].message.content.strip().upper()
            return result == "TRUE"

        except Exception as e:
            logger.warning(f"Failed hallucination detection: {e}")
            return True  # Assume it's okay if we can't check

    def _deduplicate_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate sources."""
        seen = set()
        deduplicated = []

        for source in sources:
            source_key = (source.get("title", ""), source.get("source", ""))

            if source_key not in seen:
                seen.add(source_key)
                deduplicated.append(source)

        return deduplicated
