
### 1. RAG (Retrieval-Augmented Generation) — The Game-Changer for Factual & Up-to-Date Answers

RAG is the technique that fixes one of the biggest LLM weaknesses: **hallucinations** + **outdated/static knowledge**.

**How RAG works (simple walkthrough)**:

1. **User asks a question**  
2. **Retrieval step** — The system searches a knowledge base (documents, PDFs, web pages, your own notes, codebase, etc.) using semantic search (embeddings/vector DB like Pinecone, Weaviate, or even simple grep/keyword + ranking) → finds the most relevant chunks  
3. **Augmentation** — Those retrieved chunks are injected into the prompt as extra **context**  
4. **Generation** — LLM generates the answer **grounded** in that fresh/retrieved context (instead of only its trained parametric memory)

**Why RAG pairs beautifully with what you already know**:

- System-prompt.md → Put your **permanent RAG rules** here (e.g. "Always answer using ONLY the provided context. If unsure → say 'I don't have enough information'. Cite sources with [1], [2]…")
- Skills.md → Create RAG-specific skills like "summarize_context", "answer_only_from_docs", "chain_of_thought_with_evidence"
- MCP → Many modern RAG setups use MCP servers to fetch data dynamically (e.g. query your company DB, GitHub repo, Notion, etc.) instead of static indexes

**Key prompt engineering best practices for RAG** (these make 2×–5× better results):

- **System prompt** (permanent rules):  
  - "You are a helpful expert that ONLY uses information from the provided context."  
  - "Never make up facts. If the context doesn't contain the answer → say so clearly."  
  - "Cite the relevant section with [chunk_id] or page number."  
  - "Be concise unless asked for detail."

- **User prompt structure** (dynamic part):  
  ```
  <context>
  {retrieved_chunks_here — usually separated by --- or numbered}
  </context>

  <question>
  {user question}
  </question>

  Answer the question based ONLY on the context above.
  ```

- Advanced tricks that work extremely well in 2025–2026:
  - Put the most important chunks at the **beginning** and **end** of the context (LLMs pay more attention to edges — primacy/recency bias)
  - Use **XML-style tags** → `<source>…</source>` or `<doc id='3'>…</doc>` (helps model parse & cite better)
  - Add re-ranking instructions: "First rank the chunks by relevance, then answer using the top 3."
  - Fallback rule: "If no relevant context → respond: 'I don't have specific information on this from the available documents.'"




### Next (Priority Order)

1. **Advanced RAG prompting patterns** — master the system/user split, citation forcing, chunk re-ordering, fallback language  

3. **Hybrid approaches** —  RAG + MCP (e.g. RAG → MCP tool call → generate)  
4. **Context compression & ranking** — when context gets too long, summarize or re-rank chunks  
5. **Evaluation** — create test sets to measure how well your prompts + RAG reduce hallucinations



###



**Implementation tips:**
- Use explicit markers like "Step 1:", "Reasoning:", "Action:"
- Encourage the model to verbalize its thought process
- Build in checkpoints for validation
- Allow the model to backtrack if needed

## Critical Context Management

**Context window awareness:**
- Know your model's limits (Claude has up to 200k tokens)
- Prioritize what goes in: most recent messages, relevant system instructions, retrieved docs
- Use summarization for long conversation histories

**Conversation memory:**
- Decide what to preserve across turns
- Use explicit memory structures (facts about the user, conversation goals)
- Implement forgetting strategies for stale information

**Prompt hierarchy:**
- System-level instructions (highest priority)
- User preferences and rules
- Task-specific context
- Examples and few-shot demonstrations
- Retrieved information (RAG)

## Other Important Concepts

**Few-shot learning**: Providing examples in your prompt to guide behavior
**Meta-prompting**: Prompts that generate or improve other prompts
**Tool use / Function calling**: Teaching models when and how to use external tools
**Output formatting**: Structured outputs (JSON, XML, markdown) for downstream processing
**Error handling**: Graceful degradation when the model can't fulfill requests