# 🏦 

A **graph-grounded, hallucination-resistant** knowledge assistant for banking policy documents.
Built as a graduation project demonstrating that combining vector retrieval with a knowledge graph produces measurably better answers than vector search alone.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [How It Works](#2-how-it-works)
3. [Prerequisites](#3-prerequisites)
4. [Installation](#4-installation)
5. [Configuration](#5-configuration)
6. [Building the Index](#6-building-the-index)
7. [Running the App](#7-running-the-app)
8. [App Features](#8-app-features)
9. [Project Structure](#9-project-structure)
10. [Scoring Formula](#10-scoring-formula)
11. [Troubleshooting](#11-troubleshooting)
12. [Production Mode](#12-production-mode)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                    │
│           Chat │ A/B Comparison │ Graph Explorer                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │ query
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   HybridRetriever                               │
│                                                                 │
│  ┌──────────────────┐        ┌──────────────────────────────┐  │
│  │  Vector Search   │        │     Knowledge Graph BFS      │  │
│  │  (Qdrant +       │        │  (NetworkX DiGraph, 2 hops,  │  │
│  │  nomic-embed)    │        │   semantic edges only)       │  │
│  └────────┬─────────┘        └──────────────┬───────────────┘  │
│           │                                 │                   │
│           └─────────────┬───────────────────┘                   │
│                         ▼                                        │
│              Hybrid Score = 0.4×vector + 0.4×graph + 0.2×policy│
└──────────────────────────┬──────────────────────────────────────┘
                           │ ranked evidence
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Aggregator (LLM)                            │
│   Groq API: llama-3.3-70b-versatile / llama-3.1-8b-instant     │
│   "Only state what the evidence proves — cite everything"       │
└─────────────────────────────────────────────────────────────────┘
```

### Data pipeline (run once with `build_index.py`)

```
.docx files
    │
    ▼ Docling
ParsedElements (heading / paragraph / table)
    │
    ▼ Section-aware chunker
Chunks (~512 tokens, 50-token overlap)
    │
    ├──▶ VectorStore (Qdrant in-memory)
    │        nomic-embed-text embeddings via Ollama
    │
    └──▶ Entity Extractor (Groq fast LLM)
             │
             ▼ Entity alignment (exact-string dedup)
         KnowledgeGraph (NetworkX DiGraph)
             ├── chunk nodes
             ├── entity nodes (cross-document flagged)
             ├── document + section structure
             └── semantic relation edges
                 (mentioned_in, requires, governs, …)
```

---

## 2. How It Works

**The core thesis**: Vector similarity finds *topically relevant* chunks. The knowledge graph additionally finds chunks connected to the same *entities and concepts* — even across different documents — that vector search misses. The LLM only verbalizes what the graph has already proven; it cannot hallucinate facts not present in the evidence.

| Step | What happens |
|------|-------------|
| 1 | User query → embed → retrieve top-K vector candidates |
| 2 | Extract named entities from the query |
| 3 | Match those entities to KG nodes (substring + stemmed token overlap + mention aliases) |
| 4 | BFS from seed chunks + seed entity nodes, **semantic edges only** (2 hops) |
| 5 | Score all discovered chunks: vector score + graph connectivity score + policy score |
| 6 | Re-rank, send top evidence + entity context to LLM with strict citation prompt |
| 7 | LLM outputs answer with `[1][2][3]` inline citations |

---

## 3. Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.11 or 3.12 | 3.10 may work; 3.13 not tested |
| **Ollama** | Latest | For generating embeddings locally |
| **Groq API key** | — | Free tier sufficient; used for LLM inference |
| **Git** | Any | To clone the repo |
| **RAM** | ≥ 8 GB | KG + embeddings are in-memory in dev mode |

### Install Ollama

Download from [https://ollama.com/download](https://ollama.com/download) and install.

Then pull the embedding model (required):

```bash
ollama pull nomic-embed-text
```

Verify Ollama is running:

```bash
ollama list
# Should show nomic-embed-text
```

Ollama runs as a background service on `http://localhost:11434` by default.

### Get a Groq API Key

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up (free)
3. Create an API key under **API Keys**
4. Copy the key — it starts with `gsk_`

> **No Groq key?** The system falls back to a local Ollama LLM automatically (slower). See [Configuration](#5-configuration).

---

## 4. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd GpPoc

# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Install PyVis for the graph visualizer
pip install pyvis
```

> **Windows note**: If `pip install -r requirements.txt` fails for `gliner` or `docling`, install them separately:
> ```bash
> pip install docling gliner
> ```

---

## 5. Configuration

Copy the example environment file:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and set your Groq API key:

```dotenv
# Required for LLM inference (chat answers + entity extraction)
GROQ_API_KEY=gsk_YOUR_KEY_HERE

# Environment: "dev" (default) uses Groq + local Ollama embeddings
# "prod" uses fully on-prem vLLM (see Production Mode section)
GP_ENV=dev
```

### What each variable controls

| Variable | Required | Default | Effect |
|----------|----------|---------|--------|
| `GROQ_API_KEY` | **Yes** (recommended) | — | Enables Groq API. Without it, system falls back to local Ollama LLM (needs `ollama pull qwen2.5:7b`) |
| `GP_ENV` | No | `dev` | `dev` = Groq + Ollama embeddings; `prod` = fully on-prem vLLM |

### Model defaults (dev mode)

| Role | Model | Speed |
|------|-------|-------|
| Main LLM (chat answers) | `llama-3.3-70b-versatile` via Groq | ~3 s/query |
| Fast LLM (entity extraction during build) | `llama-3.1-8b-instant` via Groq | ~0.5 s/chunk |
| Embeddings | `nomic-embed-text` via local Ollama | local |

---

## 6. Building the Index

This step parses all `.docx` documents, creates embeddings, extracts entities, and builds the knowledge graph. **Run it once** before launching the app.

```bash
python build_index.py
```

Expected output:

```
============================================================
Banking Knowledge AI — Index Builder
Environment: dev
LLM: groq/llama-3.1-8b-instant
Embedding: ollama/nomic-embed-text
============================================================

[1/5] Parsing documents...
  Total elements: 477
  Types: {'heading': 123, 'paragraph': 298, 'table': 56}

[2/5] Chunking...
  Total chunks: 212
  Saved chunks to data/parsed/chunks.json

[3/5] Embedding and indexing into vector store...
  Embedding chunk 1/212...
  ...
  Collection: {'status': 'ok', 'vectors_count': 212}

[4/5] Extracting entities...
  Extracting entities from 212 chunks...
  Processing chunk 1/212...
  ...
  Extracted ~1700 entities and ~900 relations

[5/5] Building knowledge graph...
  Added 212 chunk nodes. Graph: 380 nodes, 640 edges
  Added entities and relations. Graph: 2100 nodes, 7200 edges
  Found 236 cross-document entity links
  Saved graph to data/graph/knowledge_graph.json
```

### Output files produced

| File | Description |
|------|-------------|
| `data/parsed/chunks.json` | All chunks with metadata |
| `data/graph/knowledge_graph.json` | Full NetworkX DiGraph serialized to JSON |
| `data/parsed/` | (Docling intermediate parse files) |

> **If the build crashes mid-way** (e.g. Ollama connection drops during embedding):
> The build is not restartable by default — just re-run `python build_index.py`. The embedding cache in memory prevents duplicate Ollama calls *within* a single run. For very large doc sets consider running in smaller batches.

### Adding your own documents

1. Put your `.docx` files in `data/raw_docs/`
2. Re-run `python build_index.py`

Supported formats: **Word `.docx` only** (Docling parses headings, paragraphs, and tables with section hierarchy preserved).

---

## 7. Running the App

```bash
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

> Streamlit will auto-reload on code changes. To disable: `streamlit run app.py --server.runOnSave false`

---

## 8. App Features

### 💬 Chat page

The main interface. Type a question about banking policies and get a cited, graph-grounded answer.

**Sidebar controls:**

| Control | Description |
|---------|-------------|
| Retrieval Mode | **Hybrid** (Vector + Graph), **Vector-Only**, or **A/B Comparison** |
| Results to retrieve | Top-K slider (2–15 chunks) |
| Answer Generation Model | Choose between 70B (more accurate) and 8B (faster) |
| Show Evidence Details | Toggle to show/hide retrieved evidence with scores |
| Show Graph Info | Toggle to show BFS subgraph stats and detected query entities |

**A/B Comparison mode** runs both modes side by side and shows:
- Doc coverage delta (how many more documents hybrid retrieval covers)
- Entity count delta
- Which chunks the graph uniquely discovered that vector search missed
- Score breakdown per result: `vector` · `graph` · `policy` → `final`

### 🌐 Graph Explorer page

Interactive PyVis visualization of the full knowledge graph.

**Controls:**
- **Node types to show** — filter to entities only, chunks only, etc.
- **Max nodes to render** — cap for browser performance (default 300)
- **Focus on document** — zoom into a single document's subgraph

**Visual encoding:**

| Color | Shape | Node type |
|-------|-------|-----------|
| 🔴 Red | Dot | Entity |
| 🔵 Blue | Box | Chunk |
| 🟢 Green | Diamond | Document |
| 🟠 Orange | Triangle | Section |

- **Hover** any node → see full metadata (entity type, value, source doc, chunk content preview)
- **Hover** any edge → see relation type (e.g. `requires`, `governs`, `mentioned_in`)
- **Scroll** to zoom, **drag** to rearrange, **click and drag** nodes to pin them
- Grey edges = structural (belongs_to_document, belongs_to_section)
- White edges = semantic relations between entities

---

## 9. Project Structure

```
GpPoc/
├── app.py                      # Streamlit UI (Chat + Graph Explorer)
├── build_index.py              # One-time index builder
├── config.py                   # Dev/prod configuration dataclasses
├── requirements.txt
├── .env                        # Your secrets (not committed)
├── .env.example                # Template
│
├── data/
│   ├── raw_docs/               # Put your .docx files here
│   ├── parsed/
│   │   └── chunks.json         # Generated by build_index.py
│   └── graph/
│       └── knowledge_graph.json  # Generated by build_index.py
│
└── src/
    ├── ingestion/
    │   ├── parser.py           # Docling Word document parser
    │   ├── chunker.py          # Section-aware token chunker
    │   └── embedder.py         # Ollama embeddings + Qdrant vector store
    │
    ├── graph/
    │   ├── entity_extractor.py # LLM entity/relation extraction + alignment
    │   ├── kg_builder.py       # NetworkX KG + PyVis visualization
    │   └── traversal.py        # BFS retrieval + entity matching + scoring
    │
    ├── retrieval/
    │   └── hybrid_retriever.py # Orchestrates vector + graph + policy scoring
    │
    └── generation/
        └── aggregator.py       # LLM answer generation with strict citation prompt
```

---

## 10. Scoring Formula

Every retrieved chunk gets a final score composed of three signals:

$$\text{score} = \alpha \cdot s_{\text{vector}} + \beta \cdot s_{\text{graph}} + \gamma \cdot s_{\text{policy}}$$

| Weight | Signal | Description |
|--------|--------|-------------|
| α = 0.4 | `vector_score` | Cosine similarity between query embedding and chunk embedding |
| β = 0.4 | `graph_score` | Graph connectivity: entity count (0.3) + query entity match (0.5) + cross-doc bonus (0.2) |
| γ = 0.2 | `policy_score` | Keyword-based policy compliance signal |

When graph is disabled (Vector-Only mode), weights renormalize to α = 0.8, γ = 0.2.


