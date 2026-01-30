"""
Prompt Engineering - RAG-The-Game-Changer

Advanced prompt engineering strategies for better RAG generation.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PromptConfig:
    """Configuration for prompt engineering."""

    strategy: str = "standard"
    include_system_prompt: bool = True
    include_few_shot_examples: bool = False
    num_examples: int = 3
    enable_chain_of_thought: bool = False
    enable_role_modeling: bool = False
    max_context_length: int = 4000
    confidence_threshold: float = 0.7


@dataclass
class EngineeredPrompt:
    """Engineered prompt with all components."""

    system_prompt: str
    user_prompt: str
    examples: List[str]
    chain_of_thought: str
    role: str
    context_summary: str
    metadata: Dict[str, Any]


class PromptEngineer:
    """Engineer prompts for better RAG generation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.prompt_config = PromptConfig(**self.config)

    def engineer_prompt(
        self, query: str, retrieved_contexts: List[str], sources: Optional[List[Dict]] = None
    ) -> EngineeredPrompt:
        """Engineer an optimized prompt for RAG generation."""

        # 1. Build system prompt
        system_prompt = self._build_system_prompt()

        # 2. Build user prompt with context
        user_prompt = self._build_user_prompt(query, retrieved_contexts, sources)

        # 3. Build few-shot examples if enabled
        examples = []
        if self.prompt_config.include_few_shot_examples:
            examples = self._build_examples()

        # 4. Build chain of thought if enabled
        cot = ""
        if self.prompt_config.enable_chain_of_thought:
            cot = self._build_chain_of_thought(query, retrieved_contexts)

        # 5. Build context summary
        context_summary = self._summarize_contexts(retrieved_contexts)

        # 6. Set role if enabled
        role = "RAG Assistant" if self.prompt_config.enable_role_modeling else ""

        return EngineeredPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            examples=examples,
            chain_of_thought=cot,
            role=role,
            context_summary=context_summary,
            metadata={
                "strategy": self.prompt_config.strategy,
                "num_contexts": len(retrieved_contexts),
            },
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt for RAG assistant."""
        base_prompt = """You are a helpful AI assistant designed to answer questions based on retrieved information.

Your role is to:
1. Carefully analyze the retrieved context
2. Synthesize an accurate answer that directly addresses the user's question
3. Only use information from the retrieved contexts
4. If the information is insufficient, clearly state what you know and don't know
5. Be concise and focused
6. Provide citations when possible"""

        if self.prompt_config.confidence_threshold > 0:
            base_prompt += f"\n\nConfidence Threshold: Only provide answers with confidence above {self.prompt_config.confidence_threshold:.0%}"

        return base_prompt.strip()

    def _build_user_prompt(
        self, query: str, contexts: List[str], sources: Optional[List[Dict]]
    ) -> str:
        """Build user prompt with query and context."""

        # Format contexts
        formatted_contexts = self._format_contexts(contexts, sources)

        # Add context summary
        summary = self._summarize_contexts(contexts)

        # Build prompt
        prompt = f"""Question: {query}

Context Information:
{formatted_contexts}

Instructions:
- Answer the question using only the provided context
- If the context doesn't contain the answer, state that clearly
- Be specific and cite sources when possible
- Provide a complete but concise answer"""

        if len(formatted_contexts) > 0:
            prompt += f"\n\nContext Summary: {summary}"

        return prompt.strip()

    def _format_contexts(self, contexts: List[str], sources: Optional[List[Dict]]) -> str:
        """Format retrieved contexts with optional sources."""
        formatted = ""

        for i, (context, source) in enumerate(zip(contexts, sources or [None] * len(contexts))):
            formatted += f"\n\nContext {i + 1}:\n"
            formatted += f"{context}\n"

            if source:
                formatted += f"\nSource: {source.get('title', 'Unknown')} - {source.get('source', 'Unknown')}"

        return formatted

    def _summarize_contexts(self, contexts: List[str]) -> str:
        """Summarize retrieved contexts."""
        if not contexts:
            return "No context provided"

        # Extract key information
        all_text = " ".join(contexts)
        words = all_text.split()

        # Key themes (first 10 unique words)
        unique_words = list(dict.fromkeys(word.lower() for word in words).keys())
        themes = " ".join(unique_words[:10])

        # Length summary
        total_chars = sum(len(c) for c in contexts)
        avg_length = total_chars / len(contexts) if contexts else 0

        summary = f"Contains {len(contexts)} context(s) with {total_chars} total characters (avg {avg_length:.0f} per context). Key themes: {themes}"

        return summary

    def _build_examples(self) -> List[str]:
        """Build few-shot examples for better prompting."""
        examples = [
            {
                "context": "Paris is the capital and most populous city of France.",
                "query": "What is the capital of France?",
                "answer": "The capital of France is Paris.",
            },
            {
                "context": "The Great Wall of China is a series of fortifications made of stone, brick, and other materials.",
                "query": "How long is the Great Wall of China?",
                "answer": "The Great Wall of China is approximately 13,171 miles (21,196 km) long.",
            },
            {
                "context": "Water is composed of two hydrogen atoms and one oxygen atom, giving it the chemical formula H2O.",
                "query": "What is the chemical formula of water?",
                "answer": "The chemical formula of water is H2O.",
            },
        ]

        return examples[: self.prompt_config.num_examples]

    def _build_chain_of_thought(self, query: str, contexts: List[str]) -> str:
        """Build chain of thought reasoning."""
        cot = """Chain of Thought:

1. Analyze the Question
   - What is the user asking for?
   - What information do I need?

2. Review the Contexts
   - What information is provided?
   - Is the information relevant?
   - Are there any conflicts?

3. Synthesize the Answer
   - Combine relevant information
   - Address the user's question directly
   - Ensure accuracy based only on provided context

4. Verify the Answer
   - Does the answer directly address the question?
   - Is the answer supported by context?
   - Is the answer concise and clear?"""

        return cot

    def format_for_llm(self, engineered_prompt: EngineeredPrompt, model_type: str = "chat") -> str:
        """Format engineered prompt for specific LLM."""

        if model_type == "chat":
            # ChatML format
            messages = []

            if engineered_prompt.system_prompt:
                messages.append({"role": "system", "content": engineered_prompt.system_prompt})

            user_message = {"role": "user", "content": engineered_prompt.user_prompt}

            # Add examples if available
            if engineered_prompt.examples:
                for example in engineered_prompt.examples:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"Context: {example['context']}\nQuery: {example['query']}\nAnswer: {example['answer']}",
                        }
                    )

            messages.append(user_message)
            return str(messages)  # In practice, this would be passed as messages array

        elif model_type == "completion":
            # Completion format
            full_prompt = f"{engineered_prompt.system_prompt}\n\n{engineered_prompt.user_prompt}"

            if engineered_prompt.examples:
                full_prompt += "\n\nExamples:\n"
                for example in engineered_prompt.examples:
                    full_prompt += f"Context: {example['context']}\nQuery: {example['query']}\nAnswer: {example['answer']}\n"

            if engineered_prompt.chain_of_thought:
                full_prompt += f"\n\n{engineered_prompt.chain_of_thought}"

            return full_prompt

        else:
            return engineered_prompt.user_prompt


class AdaptivePromptEngineer(PromptEngineer):
    """Adaptive prompt engineer that adjusts based on query characteristics."""

    def engineer_prompt(
        self, query: str, retrieved_contexts: List[str], sources: Optional[List[Dict]] = None
    ) -> EngineeredPrompt:
        """Engineer adaptive prompt based on query analysis."""

        # Analyze query
        query_type = self._classify_query(query)
        query_complexity = self._assess_complexity(query)

        # Adapt prompt strategy
        base_prompt = super().engineer_prompt(query, retrieved_contexts, sources)

        # Adjust based on query type
        if query_type == "factual":
            base_prompt.system_prompt += (
                "\n\nFocus: Provide accurate, factual information with source citations."
            )
        elif query_type == "analytical":
            base_prompt.system_prompt += (
                "\n\nFocus: Analyze the information deeply and provide comprehensive reasoning."
            )
        elif query_type == "explanatory":
            base_prompt.system_prompt += "\n\nFocus: Explain concepts clearly with examples."

        # Adjust based on complexity
        if query_complexity == "high":
            base_prompt.system_prompt += "\n\nNote: This is a complex question requiring careful analysis and step-by-step reasoning."
        elif query_complexity == "simple":
            base_prompt.system_prompt += "\n\nNote: Provide a direct, concise answer."

        # Update metadata
        base_prompt.metadata.update(
            {"query_type": query_type, "complexity": query_complexity, "adaptive": True}
        )

        return base_prompt

    def _classify_query(self, query: str) -> str:
        """Classify query type."""
        query_lower = query.lower()

        # Factual questions
        if any(word in query_lower for word in ["what", "who", "when", "where", "how many"]):
            return "factual"

        # Analytical questions
        if any(
            word in query_lower for word in ["why", "how", "compare", "difference", "relationship"]
        ):
            return "analytical"

        # Explanatory questions
        if any(word in query_lower for word in ["explain", "describe", "what is", "tell me about"]):
            return "explanatory"

        return "general"

    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity."""
        # Count components
        words = query.split()
        if len(words) <= 5:
            return "simple"
        elif len(words) <= 10:
            return "medium"
        else:
            return "high"
