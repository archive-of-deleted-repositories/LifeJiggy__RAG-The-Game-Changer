from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class GeneratedResponse:
    """Response from generation."""

    answer: str
    confidence: float
    sources: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseGenerator(ABC):
    """Abstract base class for answer generators."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    async def generate(
        self, query: str, retrieved_chunks: List[Any], **kwargs
    ) -> GeneratedResponse:
        """Generate an answer based on the query and retrieved context."""
        pass


class GroundedGenerator(BaseGenerator):
    """Grounded answer generator with evidence-based generation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.citation_enabled = self.config.get("citation_enabled", True)
        self.citation_style = self.config.get("citation_style", "apa")
        self.min_confidence = self.config.get("min_confidence", 0.7)

    async def generate(
        self, query: str, retrieved_chunks: List[Any], **kwargs
    ) -> GeneratedResponse:
        """Generate an answer grounded in the retrieved context."""
        if not retrieved_chunks:
            return GeneratedResponse(
                answer="I don't have enough information to answer your question.",
                confidence=0.0,
            )

        context = self._build_context(retrieved_chunks)
        answer = await self._generate_answer(query, context)

        sources = self._extract_sources(retrieved_chunks)
        citations = self._generate_citations(sources)

        confidence = self._calculate_confidence(retrieved_chunks, answer)

        return GeneratedResponse(
            answer=answer,
            confidence=confidence,
            sources=sources,
            citations=citations,
            metadata={
                "chunks_used": len(retrieved_chunks),
                "context_length": len(context),
            },
        )

    def _build_context(self, chunks: List[Any]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []

        for i, chunk in enumerate(chunks):
            source_info = f"[Source {i + 1}]"
            if chunk.metadata.get("title"):
                source_info += f" ({chunk.metadata['title']})"

            context_parts.append(f"{source_info}\n{chunk.content}")

        return "\n\n".join(context_parts)

    async def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM with context."""
        prompt = self._create_prompt(query, context)

        try:
            from openai import OpenAI

            client = OpenAI()

            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on the provided context. Always cite your sources when providing information.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            return response.choices[0].message.content or "I couldn't generate an answer."

        except ImportError:
            return self._fallback_answer(query, context)
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"

    def _create_prompt(self, query: str, context: str) -> str:
        """Create the generation prompt."""
        return f"""Based on the following context, answer the question. If the answer is not in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

    def _fallback_answer(self, query: str, context: str) -> str:
        """Fallback answer generation without LLM."""
        return f"Based on the retrieved information, here is what I found regarding '{query}':\n\n{context[:500]}..."

    def _extract_sources(self, chunks: List[Any]) -> List[Dict[str, Any]]:
        """Extract source information from chunks."""
        sources = []
        seen_ids = set()

        for chunk in chunks:
            source_id = chunk.metadata.get("source") or chunk.document_id
            if source_id not in seen_ids:
                seen_ids.add(source_id)
                sources.append(
                    {
                        "id": source_id,
                        "title": chunk.metadata.get("title", "Unknown"),
                        "score": chunk.score,
                        "metadata": chunk.metadata,
                    }
                )

        return sources

    def _generate_citations(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate citation information."""
        citations = []

        for i, source in enumerate(sources):
            citation = {
                "index": i + 1,
                "source_id": source["id"],
                "title": source["title"],
                "style": self.citation_style,
            }
            citations.append(citation)

        return citations

    def _calculate_confidence(self, chunks: List[Any], answer: str) -> float:
        """Calculate confidence score based on retrieved chunks."""
        if not chunks:
            return 0.0

        avg_score = sum(chunk.score for chunk in chunks) / len(chunks)

        score = min(avg_score * 1.2, 1.0)

        if score < self.min_confidence:
            return round(score, 2)

        return round(min(score, 1.0), 2)


class OpenAIGenerator(GroundedGenerator):
    """OpenAI-specific generator with additional features."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = (config or {}).get("model", "gpt-4-turbo-preview")
        self.temperature = (config or {}).get("temperature", 0.1)
        self.max_tokens = (config or {}).get("max_tokens", 1000)

    async def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using OpenAI."""
        prompt = self._create_prompt(query, context)

        try:
            from openai import OpenAI

            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on the provided context. Always cite your sources when providing information.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            return response.choices[0].message.content or "I couldn't generate an answer."

        except ImportError:
            return self._fallback_answer(query, context)
        except Exception as e:
            logging.error(f"OpenAI generation error: {e}")
            raise
