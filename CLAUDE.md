# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A graph-grounded, hallucination-resistant RAG system for banking policy documents (graduation project). It combines vector retrieval (Qdrant + nomic-embed-text) with a knowledge graph (NetworkX/Neo4j) to produce cited answers. The LLM (Groq or local Ollama) only verbalizes pre-assembled evidence -- it does not reason about the answer.

## Commands

```bash
# Prerequisites: Ollama running locally with nomic-embed-text pulled
# ollama pull nomic-embed-text

# Build the index (parse docs, chunk, embed, extract entities, build KG) -- run once
python build_index.py

# Launch the Streamlit app
streamlit run app.py

# Run evaluation suite
python run_eval.py                  # all queries
python run_eval.py --quick          # first 5 per dataset
python run_eval.py --dataset single # single-doc only
python run_eval.py --dataset multi  # multi-hop only
```

## Architecture

### Data Pipeline (build_index.py)

`.docx` files in `data/raw_docs/` flow through 5 stages:
1. **Parse** (`src/ingestion/parser.py`) -- Docling extracts headings, paragraphs, tables with section hierarchy
2. **Chunk** (`src/ingestion/chunker.py`) -- Section-aware chunking (~512 tokens, 50 overlap), tables kept atomic
3. **Embed** (`src/ingestion/embedder.py`) -- nomic-embed-text via Ollama, stored in Qdrant (in-memory for dev)
4. **Extract entities** (`src/graph/entity_extractor.py`) -- LLM-based extraction of banking entities (loan types, limits, roles, etc.) with alignment/dedup
5. **Build KG** (`src/graph/kg_builder.py`) -- Dual-layer NetworkX DiGraph with chunk, entity, document, and section nodes linked by semantic relation edges

Output: `data/parsed/chunks.json` and `data/graph/knowledge_graph.json`

### Query Pipeline

1. **HybridRetriever** (`src/retrieval/hybrid_retriever.py`) combines:
   - Vector search (Qdrant cosine similarity)
   - Graph BFS traversal (`src/graph/traversal.py`) -- 2 hops on semantic edges only (structural edges like `belongs_to_document` are skipped)
   - Scoring: `final = 0.4*vector + 0.4*graph + 0.2*policy`
2. **Aggregator** (`src/generation/aggregator.py`) -- LLM verbalizes evidence with inline `[1][2][3]` citations. Strict prompt constrains it to only state what evidence proves.

### Key Design Decisions

- **Config** (`config.py`): Thread-safe singleton via `get_config()`. Two modes: `dev` (Groq API + local Ollama embeds) and `prod` (on-prem vLLM). Falls back to local Ollama LLM if no `GROQ_API_KEY`.
- **Graph backend**: Configurable via `GRAPH_BACKEND` env var. `networkx` (default, in-memory) or `neo4j` (persistent). `KnowledgeGraph` class wraps both; `neo4j_store.py` is a thin bridge adapter.
- **RBAC** (`src/auth/rbac.py`): Three access levels (Public/Confidential/Restricted). Documents mapped to levels in `DOCUMENT_ACCESS_MAP`. Access levels stamped on chunks at build time and enforced at retrieval.
- **LLM calls**: Use OpenAI-compatible client (`openai.OpenAI`) for both Groq and Ollama providers.
- **Traversal** (`src/graph/traversal.py`): BFS skips structural edges (`belongs_to_document`, `belongs_to_section`, `child_of_section`, `table_belongs_to_clause`) to avoid flooding results with same-document chunks.

## Environment

Requires `.env` file (copy from `.env.example`):
- `GROQ_API_KEY` -- Groq API key (optional; falls back to local Ollama `qwen2.5:7b`)
- `GP_ENV` -- `dev` (default) or `prod`
- `GRAPH_BACKEND` -- `networkx` (default) or `neo4j`
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` -- when using Neo4j backend

Python 3.11 or 3.12. Dependencies in `requirements.txt`. Virtual env in `.venv/` or `venv/`.
