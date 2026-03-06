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
11. [Sample Queries](#11-sample-queries)
12. [Troubleshooting](#12-troubleshooting)
13. [Production Mode](#13-production-mode)

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

---

## 11. Sample Queries

All queries below reference content that actually exists in the 30 policy documents in `data/raw_docs/`. Use **A/B Comparison** mode to see the graph's advantage on multi-hop queries.

---

### 💰 Credit Risk & Lending *(Credit_Risk_Policy)*

> Single-document retrieval — both vector and hybrid should answer well.

| # | Query | Expected answer |
|---|-------|-----------------|
| 1 | What is the single borrower concentration limit? | ≤ 20% of Tier 1 capital (= EGP 500M) |
| 2 | What are the credit approval authority tiers? | Branch ≤ EGP 5M → Regional ≤ 25M → HO ≤ 100M → Board > 100M |
| 3 | When does a DTI ratio trigger escalation? | DTI > 40% → escalate to Regional Credit Committee |
| 4 | What is the connected party exposure limit? | ≤ 25% Tier 1 (= EGP 625M) |

---

### 💱 FX & Treasury *(FX_Policy + Treasury_Operations_Manual)*

| # | Query | Expected answer |
|---|-------|-----------------|
| 1 | What is the aggregate net open position limit? | ≤ 25% of Tier 1 capital; individual currency ≤ 15% |
| 2 | What are the FX transaction limits for corporate clients? | Single: USD 5M; Daily: USD 20M |
| 3 | What triggers the FX stop-loss protocol? | Loss > 2% of position value; cumulative > 5% → suspend trading |
| 4 | By when must trades be logged in TMS-T? | Within 15 minutes; daily positions reported by 16:00 |
| 5 | What happens when a counterparty limit is breached? | Report CRO within 1 hour, BRC at next meeting |

---

### 🏛️ Capital & Liquidity *(Capital_Adequacy + Liquidity_Policy)*

| # | Query | Expected answer |
|---|-------|-----------------|
| 1 | What are the minimum capital adequacy ratios? | CET1 ≥ 7.0%, Tier 1 ≥ 8.5%, Total ≥ 12.5% |
| 2 | What is the bank's Tier 1 capital amount? | EGP 2,500,000,000 |
| 3 | What are the Contingency Funding Plan stages? | Stage 1: LCR < 120%; Stage 2: LCR < 110%; Stage 3: LCR < 100% → Board + Central Bank |
| 4 | At what capital ratio are dividends restricted? | < 13% → max 60%; < 12.5% → suspend dividends entirely |
| 5 | What is the FX liquidity buffer requirement? | ≥ 15% of FX liabilities; < 20% → Treasury Committee emergency meeting within 4 hours |

---

### 🔍 AML & Sanctions *(AML_Policy + Sanctions_Compliance)*

| # | Query | Expected answer |
|---|-------|-----------------|
| 1 | What cash transaction threshold triggers an MLRO report? | > EGP 300K |
| 2 | How quickly must a suspicious transaction report be filed with the FIU? | Within 3 business days |
| 3 | What is the aggregate transaction alert threshold over a 30-day window? | > EGP 500K |
| 4 | What happens when sanctions screening returns an exact match? | Auto-freeze → Sanctions Officer → CCO (1 hr) → CEO + Legal (2 hr) → Central Bank (24 hr) |
| 5 | How long must CDD records be retained after the end of a relationship? | 5 years; STR records indefinitely |

---

### 🖥️ IT, Cyber & Data Governance *(IT_Security + Data_Governance + Digital_Banking)*

| # | Query | Expected answer |
|---|-------|-----------------|
| 1 | What are the IT disaster recovery objectives? | RTO 4 hours, RPO 1 hour; secondary DC in Alexandria |
| 2 | What data quality standards apply to regulatory reporting data? | Completeness ≥ 99%, Accuracy ≥ 99.5%, Timeliness ≤ 1 business day |
| 3 | What authentication level is required for digital transactions over EGP 50,000? | Level 3: username + password + device + OTP + biometric |
| 4 | How long must transaction records be retained? | 10 years |
| 5 | What classification does a cybersecurity incident with > EGP 500K loss receive? | P1 Critical — CRO 1hr, BRC 4hr, Central Bank 24hr + Op Risk process |

---

### 📊 Tables & Numeric Lookups

> Targets table-type chunks — tests cross-modal retrieval.

| # | Query | Source document | Expected format |
|---|-------|-----------------|-----------------|
| 1 | What are the LTV ratios for each collateral class? | Credit Risk Policy | Table: Residential 80%, Commercial 70%, Working capital 75%, Project finance 65%, Personal secured 85% |
| 2 | What are the operational risk category ratings and review frequencies? | Operational Risk Framework | Table: 6 categories with severity, owner, and frequency |
| 3 | What are the BCM recovery time objectives for each critical system? | BCM Policy | Table: Payment 4hr, Core banking 4hr, FX trading 2hr, ATM 8hr, SWIFT 2hr |
| 4 | What are the VaR limits by trading desk? | Market Risk Framework | Aggregate USD 5M; FX USD 2M; Fixed Income USD 2.5M; Equities USD 0.5M |
| 5 | What are the Contingency Funding Plan stage thresholds? | Liquidity Policy | Stage 1 (LCR<120%), Stage 2 (<110%), Stage 3 (<100%) |

---

### 🔗 2-Document Multi-hop *(test in A/B Comparison mode)*

> Each query requires connecting facts from exactly 2 documents. Hybrid mode should pull in the second document via entity links.

| # | Query | Doc 1 → Doc 2 | Why graph helps |
|---|-------|----------------|-----------------|
| 1 | What NOP breach notification is required by FX Policy, and how does that breach interact with the Capital Adequacy ratios? | FX Policy → Capital Adequacy | Entity "NOP" + "Tier 1" links across both docs |
| 2 | When a counterparty limit is breached in Treasury Operations, what escalation chain does the Governance Policy define for the CRO and BRC? | Treasury Ops → Governance | Entities "CRO" + "BRC" are cross-document |
| 3 | What training must Treasury staff complete under HR Policy, and what specific desk certifications does the Treasury Operations Manual require? | HR Policy → Treasury Ops | Entity "Treasury Department" / "FX Trading Desk" cross-links |
| 4 | If the TMS detects an alert above EGP 300K as defined by AML Policy, what IT Security incident response procedures apply? | AML Policy → IT Security | Entities "TMS", "EGP 300K", "MLRO" cross-link |
| 5 | What vendor risk tier is a cloud provider classified as under the Vendor Risk Policy, and what Board approvals does the Outsourcing Policy require for cloud? | Vendor Risk → Outsourcing | Entity "Tier-1" + "Board IT Committee" cross-link |
| 6 | What IFRS 9 staging criteria move a loan to Stage 3, and what capital conservation actions are triggered when provisions cause capital to fall below 12.5%? | IFRS 9 → Capital Adequacy | Entities "Stage 3", "90+ DPD", "capital buffer" cross-link |

---

### 🔗 3+ Document Multi-hop *(strongest graph advantage)*

> Each query spans 3 or more documents. Vector search alone is unlikely to retrieve all relevant chunks — the graph must traverse entity links to discover distant evidence.

| # | Query | Documents spanned | Key cross-doc entities |
|---|-------|-------------------|----------------------|
| 1 | If a borrower defaults on an FX-denominated mortgage, what IFRS 9 provisions apply, what operational risk reporting is required, and how does this affect the capital adequacy ratio? | Credit Risk → IFRS 9 → Op Risk → Capital Adequacy | "default", "Stage 3", "EGP 5M loss", "Tier 1" |
| 2 | When a sanctions screening match is found on a SWIFT payment, what freeze procedures apply, how must the compliance monitoring program investigate it, and what is the employee disciplinary process? | Sanctions → Payment Systems → Compliance Monitoring → HR Policy | "SWIFT", "Sanctions Officer", "CCO", "critical finding" |
| 3 | If the bank's VaR exceeds 100% of its limit, what market risk actions are taken, how does the stress testing framework evaluate the impact, and what recovery plan triggers are activated? | Market Risk → Stress Testing → Recovery Plan → Capital Adequacy | "VaR", "BRC", "CET1 < 10.5%", "recovery trigger" |
| 4 | How does the climate risk framework affect credit underwriting for carbon-intensive sectors, what IFRS 9 forward-looking overlays should be applied, and how are these stress-tested? | Climate Risk → Credit Risk → IFRS 9 → Stress Testing | "carbon-intensive", "NGFS scenarios", "forward-looking overlay" |
| 5 | If a Tier-1 vendor (PayTech) experiences a data breach, what vendor risk procedures are triggered, what IT security incident response is required, what fraud prevention measures activate, and how does BCM ensure service continuity? | Vendor Risk → IT Security → Fraud Prevention → BCM | "PayTech", "Tier-1", "P1 incident", "CMT activation" |
| 6 | When a digital banking pre-approved loan defaults and the borrower is flagged as a PEP, what digital lending rules were required at origination, what AML enhanced due diligence applies, and what credit risk provisioning must the bank book? | Digital Banking → AML → Credit Risk → IFRS 9 | "EGP 500K", "PEP", "EDD", "Stage 2/3" |
| 7 | What approval chain is required for a related-party transaction that exceeds 5% of Tier 1, how must Internal Audit review it, and what public disclosures are required under IAS 24? | Related Party → Internal Audit → Governance → Capital Adequacy | "5% Tier 1", "BACC", "IAS 24", "30-day review" |
| 8 | How does the remuneration policy's malus/clawback mechanism interact with a trading desk loss that breaches VaR limits, and what Board risk committee oversight is involved? | Remuneration → Market Risk → Governance | "malus", "risk-adjusted P&L", "VaR capital charges", "BRC" |

---

### 🧪 Hallucination Stress Tests

> Ask about things **not** in any of the 30 documents. The system should respond with "insufficient evidence" rather than fabricating an answer.

| # | Query | Expected behaviour |
|---|-------|-------------------|
| 1 | What is the bank's policy on cryptocurrency custody services? | "Insufficient evidence" — crypto explicitly prohibited in Investment Policy, but no custody policy exists |
| 2 | What was the bank's net profit in 2024? | "Insufficient evidence" — financial statements not in policy docs |
| 3 | Who is the current CEO of the bank? | "Insufficient evidence" — no named individuals in any doc |
| 4 | What is the bank's mobile banking app download count? | "Insufficient evidence" — operational metrics not in policies |
| 5 | What cryptocurrency exchange partnerships does the bank have? | "Insufficient evidence" — no such partnerships documented |

---

## 12. Troubleshooting

### `ModuleNotFoundError: No module named 'ollama'`
```bash
venv\Scripts\pip.exe install ollama
```

### `ModuleNotFoundError: No module named 'pyvis'`
```bash
venv\Scripts\pip.exe install pyvis
```
> Always use `venv\Scripts\pip.exe` on Windows to ensure packages install into the venv, not the system Python.

### `ConnectionRefusedError` during embedding (Ollama)
Ollama is not running. Start it:
```bash
ollama serve
```
Then verify in a new terminal:
```bash
ollama list
# nomic-embed-text should appear
```

### Build crashes mid-embedding
This is usually an Ollama timeout. The system has retry logic (3 attempts with exponential backoff). If it still fails, check Ollama memory usage — the embedding model needs ~500 MB RAM. Re-run `python build_index.py` from scratch.

### App shows "System not initialized. Run `python build_index.py` first."
The files `data/parsed/chunks.json` and `data/graph/knowledge_graph.json` don't exist yet. Run the build step.

### Graph Explorer is blank / no nodes shown
- The max nodes cap may be too low — increase the slider
- Ensure node types are selected in the multiselect
- If the graph JSON is missing, rebuild the index

### Groq API rate limit errors
The free Groq tier has limits (30 req/min for 8B, 14 req/min for 70B). During `build_index.py`, entity extraction calls the fast LLM once per chunk. For 212 chunks this takes ~7 minutes at normal pace. If you hit limits, the extractor will log errors and skip those chunks (graph will still build with fewer entities).

### `No module named 'docling'`
```bash
pip install docling
```
Docling requires Python 3.11/3.12. On Python 3.13 it may fail — downgrade Python or use a conda environment.

---


