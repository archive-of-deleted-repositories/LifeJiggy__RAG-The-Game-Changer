"""
Output Formatting - RAG-The-Game-Changer

Advanced output formatting and structuring for RAG responses.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported output formats."""

    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    PLAIN_TEXT = "plain_text"
    XML = "xml"


@dataclass
class FormattedOutput:
    """Formatted response with structure and citations."""

    answer: str
    format_type: OutputFormat
    formatted_content: str
    citations: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_data: Any = None


class OutputFormatter:
    """Format and structure RAG outputs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.default_format = OutputFormat(self.config.get("default_format", "markdown"))
        self.include_citations = self.config.get("include_citations", True)
        self.include_confidence = self.config.get("include_confidence", False)
        self.include_sources = self.config.get("include_sources", True)

    def format_output(
        self,
        answer: str,
        retrieved_chunks: List[Dict],
        confidence: float,
        format_type: Optional[OutputFormat] = None,
    ) -> FormattedOutput:
        """Format output according to specified format."""
        fmt = format_type or self.default_format
        citations = self._extract_citations(retrieved_chunks)

        formatted_content = self._generate_formatted_content(
            answer=answer,
            format_type=fmt,
            citations=citations,
            confidence=confidence,
            sources=retrieved_chunks,
        )

        return FormattedOutput(
            answer=answer,
            format_type=fmt,
            formatted_content=formatted_content,
            citations=citations,
            metadata={
                "confidence": confidence if self.include_confidence else None,
                "chunk_count": len(retrieved_chunks),
            },
            raw_data={"answer": answer, "chunks": retrieved_chunks, "confidence": confidence},
        )

    def _extract_citations(self, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Extract citation information from retrieved chunks."""
        citations = []

        for i, chunk in enumerate(chunks):
            citation = {
                "index": i + 1,
                "id": chunk.get("chunk_id", f"chunk_{i}"),
                "document_id": chunk.get("document_id", ""),
                "title": chunk.get("title", "Unknown"),
                "source": chunk.get("source", "Unknown"),
                "content": chunk.get("content", "")[:100] + "...",
            }

            if chunk.get("metadata"):
                citation["metadata"] = chunk["metadata"]

            citations.append(citation)

        return citations

    def _generate_formatted_content(
        self,
        answer: str,
        format_type: OutputFormat,
        citations: List[Dict],
        confidence: float,
        sources: List[Dict],
    ) -> str:
        """Generate formatted content based on format type."""

        if format_type == OutputFormat.JSON:
            return self._format_json(answer, citations, confidence, sources)

        elif format_type == OutputFormat.MARKDOWN:
            return self._format_markdown(answer, citations, confidence, sources)

        elif format_type == OutputFormat.HTML:
            return self._format_html(answer, citations, confidence, sources)

        elif format_type == OutputFormat.XML:
            return self._format_xml(answer, citations, confidence, sources)

        else:  # PLAIN_TEXT
            return self._format_plain_text(answer, citations, sources)

    def _format_json(
        self, answer: str, citations: List[Dict], confidence: float, sources: List[Dict]
    ) -> str:
        """Format output as JSON."""
        output = {"answer": answer}

        if self.include_citations:
            output["citations"] = citations

        if self.include_confidence:
            output["confidence"] = confidence

        if self.include_sources:
            output["sources"] = [
                {"id": c.get("id"), "title": c.get("title"), "source": c.get("source")}
                for c in citations
            ]

        return json.dumps(output, indent=2)

    def _format_markdown(
        self, answer: str, citations: List[Dict], confidence: float, sources: List[Dict]
    ) -> str:
        """Format output as Markdown."""
        lines = [answer]

        if self.include_citations and citations:
            lines.append("\n\n**Sources:**")
            for citation in citations:
                lines.append(f"\n[{citation['index']}] {citation['title']}")
                lines.append(f"   - Source: {citation['source']}")
                lines.append(f"   - Document ID: {citation['document_id']}")

        if self.include_confidence:
            lines.append(f"\n\n**Confidence:** {confidence:.1%}")

        return "\n".join(lines)

    def _format_html(
        self, answer: str, citations: List[Dict], confidence: float, sources: List[Dict]
    ) -> str:
        """Format output as HTML."""
        html = f"<div class='rag-response'>\n"
        html += f"  <p class='answer'>{answer}</p>\n"

        if self.include_citations and citations:
            html += "  <div class='citations'>\n"
            html += "    <h3>Sources:</h3>\n"
            html += "    <ul>\n"
            for citation in citations:
                html += (
                    f"      <li data-id='{citation['id']}' data-source='{citation['source']}'>\n"
                )
                html += f"        <strong>{citation['index']}. {citation['title']}</strong>\n"
                html += f"        <br/><small>{citation['source']}</small>\n"
                html += "      </li>\n"
            html += "    </ul>\n"
            html += "  </div>\n"

        if self.include_confidence:
            html += f"  <div class='confidence'>Confidence: {confidence:.1%}</div>\n"

        html += "</div>"
        return html

    def _format_xml(
        self, answer: str, citations: List[Dict], confidence: float, sources: List[Dict]
    ) -> str:
        """Format output as XML."""
        xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xml += "<response>\n"
        xml += f"  <answer>{answer}</answer>\n"

        if self.include_citations and citations:
            xml += "  <citations>\n"
            for citation in citations:
                xml += f"    <citation id='{citation['id']}' source='{citation['source']}'>\n"
                xml += f"      <title>{citation['title']}</title>\n"
                xml += f"      <document_id>{citation['document_id']}</document_id>\n"
                xml += "    </citation>\n"
            xml += "  </citations>\n"

        if self.include_confidence:
            xml += f"  <confidence>{confidence:.4f}</confidence>\n"

        xml += "</response>"
        return xml

    def _format_plain_text(self, answer: str, citations: List[Dict], sources: List[Dict]) -> str:
        """Format output as plain text."""
        lines = [answer]

        if self.include_citations and citations:
            lines.append("\n\nSources:")
            for citation in citations:
                lines.append(f"\n[{citation['index']}] {citation['title']} ({citation['source']})")

        return "\n".join(lines)


class StructuredOutputGenerator:
    """Generate structured outputs for different use cases."""

    @staticmethod
    def generate_qa_pair(
        question: str,
        answer: str,
        context: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate Q&A pair structure."""
        qa_pair = {"question": question, "answer": answer}

        if context:
            qa_pair["context"] = context

        if confidence is not None:
            qa_pair["confidence"] = confidence

        return qa_pair

    @staticmethod
    def generate_search_result(
        query: str, results: List[Dict], total_results: int, search_time_ms: float
    ) -> Dict[str, Any]:
        """Generate search result structure."""
        return {
            "query": query,
            "results": results,
            "total_results": total_results,
            "search_time_ms": search_time_ms,
            "results_count": len(results),
        }

    @staticmethod
    def generate_conversation_turn(
        user_message: str,
        assistant_message: str,
        turn_id: str,
        context_used: List[str],
        confidence: float,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Generate conversation turn structure."""
        return {
            "turn_id": turn_id,
            "user": user_message,
            "assistant": assistant_message,
            "context_used": {
                "count": len(context_used),
                "summaries": [c[:100] + "..." for c in context_used],
            },
            "confidence": confidence,
            "metadata": metadata or {},
        }

    @staticmethod
    def generate_document_insight(
        document_id: str, summary: str, key_points: List[str], topics: List[str], confidence: float
    ) -> Dict[str, Any]:
        """Generate document insight structure."""
        return {
            "document_id": document_id,
            "summary": summary,
            "key_points": key_points,
            "topics": topics,
            "confidence": confidence,
        }
