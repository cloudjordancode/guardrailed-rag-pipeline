# Reliability & Guardrails

**Module:** 8 — RAG Intro & Docker Deployment
**Estimated time:** 30 minutes

## Objective

Make your RAG pipeline production-ready by adding 4 guardrails that improve reliability and prevent hallucinations.

## What You'll Build

An enhanced `rag_pipeline.py` built on top of your Lesson 3 pipeline, with these additions:

**Guardrail 1 — Distance threshold filtering**
Only use chunks with a distance below a configurable threshold. If no chunks pass, return a "no relevant information found" response without calling the LLM.

**Guardrail 2 — Confidence levels**
Return a confidence label with every response: `"high"` (best distance < 0.5), `"medium"` (< 1.0), or `"low"` (≥ 1.0).

**Guardrail 3 — Strengthened system prompt**
Explicitly instruct the model: never make up information, say "I don't know" when unsure, always cite sources.

**Guardrail 4 — Structured response**
Return results as a dict: `{"answer", "sources", "confidence", "chunks_retrieved"}` instead of a plain string.

## Reference Code

The starter file (`rag_pipeline.py`) provides a scaffold with TODOs. Start from your `my_rag.py` (Lesson 3) or use the starter.

## Running

```bash
python rag_pipeline.py
```

## Testing

Run 4 queries:

1. In-scope (should be answerable)
2. Partially in-scope (context touches on it but doesn't fully answer)
3. Out-of-scope (completely outside your documents)
4. Ambiguous (could go either way)

## Deliverable

A working `rag_pipeline.py` with all 4 guardrails, printing structured responses for each of the 4 query types.
