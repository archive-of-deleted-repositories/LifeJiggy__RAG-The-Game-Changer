"""
Conversational RAG - RAG-The-Game-Changer

Advanced RAG pattern for multi-turn conversations with memory.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

from ..config.pipeline_configs.rag_pipeline import RAGPipeline, RAGResponse

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single turn in conversation."""

    query: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Context for ongoing conversation."""

    conversation_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_metadata: Dict[str, Any] = field(default_factory=dict)

    def add_turn(self, turn: ConversationTurn):
        """Add a turn to conversation."""
        self.turns.append(turn)
        # Keep only last N turns to avoid context overflow
        max_turns = self.session_metadata.get("max_turns", 10)
        if len(self.turns) > max_turns:
            self.turns = self.turns[-max_turns:]

    def get_context_summary(self, max_tokens: int = 2000) -> str:
        """Get summary of conversation context."""
        if not self.turns:
            return ""

        context_parts = []
        current_tokens = 0

        # Add recent turns to context
        for turn in reversed(self.turns[-5:]):  # Last 5 turns
            turn_text = f"User: {turn.query}\nAssistant: {turn.answer}\n"
            estimated_tokens = len(turn_text.split()) * 1.3  # Rough estimate

            if current_tokens + estimated_tokens > max_tokens:
                break

            context_parts.append(turn_text)
            current_tokens += estimated_tokens

        return "\n".join(reversed(context_parts))


class ConversationalRAG:
    """Advanced RAG pattern for conversational AI with memory."""

    def __init__(self, base_pipeline: RAGPipeline, config: Optional[Dict[str, Any]] = None):
        self.pipeline = base_pipeline
        self.config = config or {}

        # Conversation management
        self.conversations: Dict[str, ConversationContext] = {}
        self.max_conversations = self.config.get("max_conversations", 1000)

        # Context enhancement settings
        self.use_contextual_query_rewrite = self.config.get("use_contextual_query_rewrite", True)
        self.use_persona = self.config.get("use_persona", False)
        self.persona = self.config.get("persona", "helpful assistant")

        # Memory settings
        self.long_term_memory_enabled = self.config.get("long_term_memory_enabled", False)
        self.conversation_summary_frequency = self.config.get("conversation_summary_frequency", 5)

    async def start_conversation(
        self,
        conversation_id: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new conversation."""
        if conversation_id is None:
            conversation_id = f"conv_{int(time.time() * 1000)}"

        # Clean up old conversations if needed
        if len(self.conversations) >= self.max_conversations:
            oldest_id = min(self.conversations.keys())
            del self.conversations[oldest_id]
            logger.info(f"Cleaned up old conversation: {oldest_id}")

        # Create new conversation context
        context = ConversationContext(
            conversation_id=conversation_id,
            user_preferences=user_preferences or {},
            session_metadata={
                "max_turns": self.config.get("max_turns_per_conversation", 20),
                "started_at": time.time(),
            },
        )

        self.conversations[conversation_id] = context

        logger.info(f"Started new conversation: {conversation_id}")
        return conversation_id

    async def query(
        self,
        query: str,
        conversation_id: str,
        include_sources: bool = True,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process conversational query."""
        try:
            # Get conversation context
            context = self.conversations.get(conversation_id)
            if not context:
                context = await self.start_conversation(conversation_id)

            # Enhance query with context if enabled
            enhanced_query = await self._enhance_query(query, context)

            # Process query through base pipeline
            response = await self.pipeline.query(
                query=enhanced_query, top_k=top_k or 5, include_sources=include_sources
            )

            # Add conversational elements to response
            conversational_response = self._add_conversational_elements(response, query, context)

            # Store turn in conversation
            turn = ConversationTurn(
                query=query,
                answer=response.answer,
                sources=response.sources,
                metadata={"enhanced_query": enhanced_query, "context_used": len(context.turns) > 0},
            )
            context.add_turn(turn)

            # Generate conversation summary if needed
            if len(context.turns) % self.conversation_summary_frequency == 0:
                await self._generate_conversation_summary(context)

            return {
                "answer": conversational_response,
                "sources": response.sources,
                "conversation_id": conversation_id,
                "turn_number": len(context.turns),
                "enhanced_query": enhanced_query,
                "context_length": len(context.turns),
                "response_time_ms": response.total_time_ms,
            }

        except Exception as e:
            logger.error(f"Error in conversational query: {e}")
            raise

    async def _enhance_query(self, query: str, context: ConversationContext) -> str:
        """Enhance query with conversational context."""
        if not self.use_contextual_query_rewrite or not context.turns:
            return query

        # Build contextual prompt
        recent_context = context.get_context_summary(1000)  # Last 1000 tokens

        if recent_context:
            enhanced_query = f"""Given the following conversation context, rewrite the user's query to be more specific while preserving their intent.

Context:
{recent_context}

User's current query: {query}

Rewritten query:"""

            try:
                # Use LLM to enhance query
                from openai import OpenAI

                client = OpenAI()

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that rewrites queries to be more specific based on conversation context.",
                        },
                        {"role": "user", "content": enhanced_query},
                    ],
                    temperature=0.1,
                    max_tokens=150,
                )

                rewritten = response.choices[0].message.content.strip()
                logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
                return rewritten

            except Exception as e:
                logger.warning(f"Failed to enhance query: {e}")
                return query

        return query

    def _add_conversational_elements(
        self, response: RAGResponse, query: str, context: ConversationContext
    ) -> str:
        """Add conversational elements to response."""
        answer = response.answer

        # Add contextual references
        if len(context.turns) > 1:
            answer = self._add_contextual_references(answer, context)

        # Add persona if enabled
        if self.use_persona:
            answer = self._apply_persona(answer)

        # Add conversational transitions
        answer = self._add_conversational_transitions(answer, context)

        return answer

    def _add_contextual_references(self, answer: str, context: ConversationContext) -> str:
        """Add references to previous conversation."""
        # Simple implementation - can be enhanced with more sophisticated logic
        if "previous" in answer.lower() and len(context.turns) > 1:
            last_turn = context.turns[-2]
            return answer.replace(
                "previous", f"what I mentioned earlier about {last_turn.query[:50]}..."
            )
        return answer

    def _apply_persona(self, answer: str) -> str:
        """Apply persona to response."""
        persona_prefixes = {
            "helpful": "Here's what I found to help you: ",
            "professional": "Based on my analysis: ",
            "casual": "So, here's the deal: ",
        }

        prefix = persona_prefixes.get(self.persona, "")
        if prefix and not answer.startswith(prefix):
            return prefix + answer
        return answer

    def _add_conversational_transitions(self, answer: str, context: ConversationContext) -> str:
        """Add conversational transitions."""
        # Add follow-up suggestions
        if len(context.turns) == 1:  # First turn
            answer += (
                "\n\nIs there anything specific about this topic you'd like to know more about?"
            )
        elif len(context.turns) > 5:  # Long conversation
            answer += "\n\nWould you like me to summarize our conversation so far or explore a different aspect?"

        return answer

    async def _generate_conversation_summary(self, context: ConversationContext):
        """Generate summary of conversation."""
        try:
            # Extract key topics and user interests from conversation
            user_queries = [turn.query for turn in context.turns]

            summary = {
                "turn_count": len(context.turns),
                "key_topics": self._extract_key_topics(user_queries),
                "user_interests": self._identify_user_interests(user_queries),
                "last_activity": context.turns[-1].timestamp if context.turns else None,
                "conversation_duration": time.time()
                - context.session_metadata.get("started_at", time.time()),
            }

            context.session_metadata["summary"] = summary
            logger.info(f"Generated summary for conversation {context.conversation_id}")

        except Exception as e:
            logger.warning(f"Failed to generate conversation summary: {e}")

    def _extract_key_topics(self, queries: List[str]) -> List[str]:
        """Extract key topics from queries."""
        # Simple keyword extraction - can be enhanced with NLP
        topics = set()
        stop_words = {
            "what",
            "how",
            "why",
            "when",
            "where",
            "the",
            "a",
            "an",
            "is",
            "are",
            "in",
            "on",
            "at",
            "to",
        }

        for query in queries:
            words = [w.lower() for w in query.split() if w.lower() not in stop_words and len(w) > 3]
            topics.update(words)

        return list(topics)[:10]  # Top 10 topics

    def _identify_user_interests(self, queries: List[str]) -> List[str]:
        """Identify user interests from queries."""
        # Simple pattern matching - can be enhanced with ML
        interest_patterns = {
            "technical": ["algorithm", "code", "programming", "database", "api"],
            "business": ["market", "revenue", "strategy", "management", "company"],
            "academic": ["research", "study", "paper", "theory", "methodology"],
            "practical": ["how to", "tutorial", "guide", "steps", "implementation"],
        }

        interests = []
        query_text = " ".join(queries).lower()

        for interest, keywords in interest_patterns.items():
            if any(keyword in query_text for keyword in keywords):
                interests.append(interest)

        return interests

    async def get_conversation_history(
        self, conversation_id: str, max_turns: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get conversation history."""
        context = self.conversations.get(conversation_id)
        if not context:
            return {"error": "Conversation not found"}

        turns = context.turns
        if max_turns:
            turns = turns[-max_turns:]

        return {
            "conversation_id": conversation_id,
            "turns": [
                {
                    "query": turn.query,
                    "answer": turn.answer,
                    "sources": turn.sources,
                    "timestamp": turn.timestamp,
                    "metadata": turn.metadata,
                }
                for turn in turns
            ],
            "total_turns": len(context.turns),
            "user_preferences": context.user_preferences,
            "session_metadata": context.session_metadata,
        }

    async def end_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """End a conversation and optionally summarize."""
        context = self.conversations.get(conversation_id)
        if not context:
            return {"error": "Conversation not found"}

        # Generate final summary
        await self._generate_conversation_summary(context)

        # Remove from active conversations
        del self.conversations[conversation_id]

        logger.info(f"Ended conversation: {conversation_id}")
        return {
            "conversation_id": conversation_id,
            "final_summary": context.session_metadata.get("summary", {}),
            "ended_at": time.time(),
        }

    async def get_all_conversations(self) -> List[str]:
        """Get list of all active conversation IDs."""
        return list(self.conversations.keys())

    async def clear_all_conversations(self) -> Dict[str, Any]:
        """Clear all conversations."""
        count = len(self.conversations)
        self.conversations.clear()
        logger.info(f"Cleared {count} conversations")
        return {"cleared_conversations": count}
