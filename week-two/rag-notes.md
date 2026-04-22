# Week 2 — RAG Learning Notes

## Topics Covered in This Document

1. [**Why RAG is Needed**](#1-why-rag-is-needed)
2. [**RAG Scenarios**](#2-rag-scenarios)
3. [**Vanilla RAG**](#3-vanilla-rag)

---

# 1. Why RAG is Needed

Retrieval-Augmented Generation (RAG) is a technique that gives a Large Language Model (LLM) access to external information at the time of answering a question. Although LLMs are powerful, they have several built-in limitations. RAG is needed to address those limitations. The main reasons are explained below.

## Knowledge Cutoff Limitations

Every LLM is trained on data collected up to a specific date, called its training cutoff. After that date, the model has no awareness of new events, new documents, or updated facts. This means the model cannot answer questions about recent developments on its own. RAG solves this by retrieving fresh information from external sources at query time and handing it to the model.

## Privacy and Compliance

Organizations often need to work with sensitive data such as customer records, internal policies, or proprietary research. This data should not be part of a publicly trained model's weights. RAG keeps such information inside retrieval systems that the organization owns and controls. The LLM reads from these systems only when needed, so the sensitive data never becomes part of the model itself. This makes privacy and compliance much easier to manage.

## Reducing Hallucinations

A hallucination is when a model produces an answer that sounds correct but is actually wrong. Because RAG grounds the model's response in specific retrieved documents, the output can be traced back to its source. This reduces fabricated answers and also makes it possible to cite the original passages, giving users a way to verify the information.

## Domain-Specific Accuracy

General-purpose LLMs are trained on broad internet data, so they have wide but shallow coverage. They often struggle with specialized fields such as medicine, law, or a company's internal processes. RAG allows the model to reach into curated, domain-specific knowledge bases (technical manuals, legal archives, product documents) and use them as the source of truth.

## Handling Long-Tail Entities

LLMs perform well on popular topics because those topics appear frequently in training data. For rare entities — an obscure assistant professor, a small regional company, an uncommon medication — the model's knowledge is weak and unreliable. RAG fills this gap by retrieving targeted documents about those rare entities whenever they are needed.

---

# 2. RAG Scenarios

RAG is not limited to one use case. It is applied in several common scenarios depending on what kind of information the system needs to bring in. The three main scenarios are described below.

## Retrieving External Knowledge

This is the most common scenario. When a question goes beyond what the LLM already knows — because the information is recent, specialized, or private — the system retrieves relevant documents from an external knowledge base and provides them to the model as supporting context. This plugs the knowledge gap of the LLM.

## Retrieving Context History

In long-running conversations, it is not practical to keep every past message inside the prompt because of context window limits and cost. Instead, RAG stores earlier parts of the conversation in a retrievable form. When the current conversation refers back to an older topic, the system fetches only the relevant history. This allows the assistant to stay coherent over hours, days, or weeks.

## Retrieving In-Context Training Examples

LLMs can learn new tasks directly from examples placed inside the prompt. This is called in-context learning. For a new or unusual task, RAG dynamically retrieves the most relevant examples from a stored pool and adds them to the prompt. This helps the model understand the task without any retraining. This scenario connects directly to the prompting strategies below.

### Prompting Strategies: Zero-shot, One-shot, Few-shot

These strategies describe *how many examples* are provided to the LLM. They are not RAG by themselves, but the few-shot strategy becomes powerful when combined with RAG.

**Zero-shot** — The model is given only the instruction, without any examples. It relies entirely on what it already knows. Example: *"Classify this review as positive or negative: 'The food was amazing.'"*

**One-shot** — The model is given a single worked example before the real task. This helps it understand the expected format. Example: *"Review: 'Terrible service.' → Negative. Now classify: 'The food was amazing.'"*

**Few-shot** — The model is given several examples (usually two to five) before the real task. This is useful for tasks with subtle patterns or strict output formatting.

**Dynamic Few-shot (RAG-powered)** — Instead of hard-coding examples in the prompt, the system retrieves the most relevant examples for each incoming query from a larger pool. The model sees examples that match the specific situation, which significantly improves accuracy on varied tasks.

| Technique        | Uses RAG?                                       |
| ---------------- | ----------------------------------------------- |
| Zero-shot        | No — no examples retrieved                      |
| Static few-shot  | No — examples hardcoded in prompt               |
| Dynamic few-shot | **Yes** — RAG retrieves best examples per query |

### Dynamic Few-shot vs. Prompt Chain

Dynamic few-shot and prompt chaining are sometimes confused, but they are different patterns.

**Dynamic few-shot** is about *what goes into one prompt*. The system retrieves relevant examples, injects them into a single prompt, and makes one LLM call.

**Prompt chain** is about *how a task is split across multiple prompts*. The system makes several sequential LLM calls, where the output of one call becomes the input of the next.

The two can also be combined — a step inside a prompt chain can itself use dynamic few-shot retrieval.

---

# 3. Vanilla RAG

Vanilla RAG is the simplest, baseline version of Retrieval-Augmented Generation. It contains no advanced optimizations and serves as the foundation that every more sophisticated RAG system extends. Understanding vanilla RAG is essential because all improvements are built on top of its basic pipeline.

## The Vanilla RAG Pipeline

The pipeline consists of two parallel flows that converge at a step called *Context Engineering / Augment*, and the result is finally passed to the LLM to generate the response.

```
        ┌─────────┐       ┌──────────┐
Data →  │ Chunks  │  →    │ Vector   │
        └─────────┘       │  Store   │
                          └──────────┘
                               │
                               ▼
Query  →  Retriever  →  Retrieved Chunks  →  Context Engineering (Augment)  →  LLM  →  Generated Response
```

The user's query is passed directly into the Augment step. At the same time, the retriever uses the query to search the vector store and produces a set of relevant chunks, which also flow into the Augment step. Augmentation combines both into a single prompt, which is then sent to the LLM to produce the final response.

## Indexing Stage

Before any queries can be answered, the source documents must be prepared. This is done once, ahead of time. Each document is split into smaller chunks that are semantically coherent. Every chunk is passed through an embedding model, which converts it into a numerical vector that represents its meaning. These vectors are stored in a specialized vector database such as Pinecone, Chroma, or FAISS, which supports fast similarity search.

## Retrieval Stage

When a user submits a query, the system converts the query into an embedding using the same model that was used during indexing. The retriever searches the vector database for the top-k most similar chunks, usually ranked by cosine similarity. The result is a small set of document snippets that are most relevant to the question.

### Top-k in Retrieval

The term **top-k** refers to the number of most-similar chunks that the retriever returns. The letter *k* is a parameter chosen by the system designer, commonly 3, 5, or 10. "Top" means the chunks with the highest similarity scores.

For example, if the vector database contains 10,000 chunks and a user asks *"What's the refund policy?"* with *k = 3*, the database returns the three chunks with the highest similarity scores. Only those three are sent forward into the augmentation step.

Choosing the right value of *k* is a trade-off between recall and noise:

| k value           | Trade-off                                                    |
| ----------------- | ------------------------------------------------------------ |
| **Small** (1–3)   | Precise, less noise, but may miss information. Cheaper.     |
| **Medium** (5–10) | Balanced; common default.                                    |
| **Large** (20+)   | Higher recall but noisy and expensive; can confuse the LLM. |

Most vanilla RAG systems start at *k = 3 to 5* and tune from there. Sending all chunks is impractical because of context window limits, higher cost, slower responses, and the "lost in the middle" problem where the model overlooks important information buried in too much content.

## Context Engineering / Augment Stage

This stage is the bridge between retrieval and generation. The retrieved chunks and the original query are combined into a single, well-structured prompt. The prompt usually has clear section headers such as *Context* and *Question*, along with explicit instructions on how the model should use the context. The quality of this step directly affects how grounded and accurate the final answer will be.

A typical augmented prompt looks like this:

```
Context:
{retrieved_chunk_1}
{retrieved_chunk_2}
{retrieved_chunk_3}

Question: {user_query}

Answer using only the context above.
```

### Why Context and Question Are Sent Together

The retrieved context and the user's question are always sent in one combined prompt, not as separate messages. The reason is how LLMs fundamentally work.

An LLM is **stateless** — each API call is independent. The model has no memory of previous calls. If the context were sent in one call and the question in another, the model would have already forgotten the context by the time the question arrived. Everything the model needs must be inside a single prompt.

The **context** acts as the knowledge source — the "open book" the model reads from. Without it, the model would fall back on its training data, which may be outdated or wrong for the specific case.

The **question** acts as the instruction — it tells the model what to extract, summarize, or reason about using the given context.

Together, they form a complete request: *"Using this context, answer this question."* Neither part is useful on its own.

## Generation Stage

The augmented prompt is sent to the LLM. The LLM reads the context, interprets the question, and produces the final response. Because the relevant information is already present in the prompt, the answer is grounded in the retrieved material rather than relying only on the model's internal memory.

## Why Chunking Matters

Chunking is one of the most important design decisions in a RAG system, for three reasons.

First, chunks must be **semantically coherent**. A good chunk contains a complete idea or section so that it still makes sense when retrieved on its own.

Second, **LLMs have a fixed input window**. Regardless of document size, only a limited number of tokens fit into one prompt. Chunking breaks long documents into manageable pieces that fit within this limit.

Third, **chunking enables precise retrieval**. Instead of fetching an entire document when only one paragraph is relevant, the system fetches just the matching pieces. This avoids clutter in the context and lets the model focus on what is actually useful.

## Limitations of Vanilla RAG

Vanilla RAG is a strong starting point but has known weaknesses. Retrieval quality is fragile — if the retriever fetches the wrong chunks, the answer will be wrong regardless of how capable the LLM is. Naive fixed-size chunking often cuts across sentence or paragraph boundaries, destroying meaning. The retrieval step is single-shot, meaning the system cannot refine its query or re-rank results. Multi-hop questions that need information from multiple documents are not handled well. And when too many irrelevant chunks are included, the LLM can become confused rather than better informed.

## What Comes After Vanilla RAG

More advanced RAG systems improve on these limitations with techniques such as **query rewriting** and HyDE (Hypothetical Document Embeddings) to reformulate the query before retrieval; **re-ranking** to sort candidates more accurately using a second model; **hybrid search** to combine vector similarity with keyword methods like BM25; **multi-hop or agentic RAG** for iterative retrieval across reasoning steps; and **better chunking strategies** based on semantic boundaries or document structure.
