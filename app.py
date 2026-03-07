"""
Streamlit Chat UI for the Banking Knowledge AI System.
Displays: answers, citations, evidence paths, graph visualization, confidence scores.
Features: RBAC login, document filtering, streaming answers.
"""

import json
import os
import re
import html as html_mod
import tempfile
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

from config import get_config
from src.ingestion.chunker import Chunk
from src.ingestion.embedder import VectorStore

try:
    import ollama as _ollama_client
except ImportError:
    _ollama_client = None
from src.graph.kg_builder import KnowledgeGraph
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.aggregator import generate_response
from src.auth.rbac import (
    authenticate,
    get_accessible_docs,
    get_access_level,
    AccessLevel,
    UserProfile,
)


# ── Available-model discovery ──


def _discover_models(config) -> list[dict]:
    """
    Build a list of selectable LLM models.

    Each entry: {"label": "display name", "model": "raw model id",
                 "provider": "groq" | "ollama_chat"}
    Order: configured models first, then any extra Ollama models.
    """
    seen: set[str] = set()
    models: list[dict] = []

    def _add(model: str, provider: str):
        key = f"{provider}::{model}"
        if key in seen:
            return
        seen.add(key)
        tag = "Groq" if provider == "groq" else "Ollama" if "ollama" in provider else provider
        models.append({"label": f"{model}  ({tag})", "model": model, "provider": provider})

    # 1. Configured models (always shown)
    _add(config.main_llm.model, config.main_llm.provider)
    if config.fast_llm.model != config.main_llm.model or config.fast_llm.provider != config.main_llm.provider:
        _add(config.fast_llm.model, config.fast_llm.provider)

    # 2. Locally installed Ollama models
    if _ollama_client is not None:
        try:
            for m in _ollama_client.list().models:
                name = getattr(m, "model", "") or getattr(m, "name", "")
                if name:
                    _add(name, "ollama_chat")
        except Exception:
            pass  # Ollama server not running — skip

    return models


# ── Page config ──
st.set_page_config(
    page_title="Banking Knowledge AI",
    page_icon="🏦",
    layout="wide",
)

# ── Access level display helpers ──
_ACCESS_BADGES = {
    1: ("🟢", "Public"),
    2: ("🟡", "Confidential"),
    3: ("🔴", "Restricted"),
}


def _access_badge(level: int) -> str:
    """Return emoji + label for a given access level."""
    emoji, label = _ACCESS_BADGES.get(level, ("⚪", "Unknown"))
    return f"{emoji} {label}"


def _render_login_page():
    """Render the login page and return True if authenticated."""
    st.title("Banking Knowledge Chatbot")
    st.caption("Graph-grounded, hallucination-resistant knowledge assistant")
    st.divider()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("🔐 Login")
        st.markdown(
            "Sign in with your credentials to access the banking knowledge base."
        )

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            user = authenticate(username, password)
            if user:
                st.session_state.user = user
                st.session_state.messages = []  # fresh chat on login
                st.rerun()
            else:
                st.error("Invalid username or password.")

        with st.expander("Demo accounts"):
            st.markdown(
                "| Username | Password | Role | Access |\n"
                "|---|---|---|---|\n"
                "| `teller` | `teller123` | Bank Teller | 🟢 Public |\n"
                "| `risk_analyst` | `risk123` | Risk Analyst | 🟡 Confidential |\n"
                "| `compliance` | `comp123` | Compliance Officer | 🟡 Confidential |\n"
                "| `cro` | `cro123` | Chief Risk Officer | 🔴 Restricted |"
            )


@st.cache_resource
def init_system():
    """Initialize the full pipeline (cached across reruns)."""
    config = get_config()

    # Check for pre-built index
    graph_path = Path(config.graph_dir) / "knowledge_graph.json"
    chunks_path = Path(config.parsed_dir) / "chunks.json"

    if not (graph_path.exists() and chunks_path.exists()):
        return None, None, None

    # Load graph
    kg = KnowledgeGraph(config)
    kg.load(graph_path)
    stats = kg.get_stats()
    print(f"Loaded graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges")

    # Load chunks and embed
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    chunks = [Chunk(**c) for c in chunks_data]

    vector_store = VectorStore(config)
    vector_store.index_chunks(chunks)

    retriever = HybridRetriever(vector_store, kg, config)
    return retriever, kg, config


def _linkify_citations(text: str, msg_idx: int) -> str:
    """Replace [1], [2], etc. in answer text with clickable anchor links.
    HTML-escapes the LLM output first to prevent XSS."""
    safe_text = html_mod.escape(text)
    def _replace_citation(match):
        num = match.group(1)
        return f'<a href="#cite-{msg_idx}-{num}" style="color: #4A90D9; text-decoration: none; font-weight: bold;">[{num}]</a>'
    return re.sub(r'\[(\d+)\]', _replace_citation, safe_text)


def _render_citations(citations: list[dict], msg_idx: int):
    """Render citations with anchor targets so inline links can scroll to them."""
    for i, c in enumerate(citations, 1):
        emoji = "📊" if c.get("element_type") == "table" else "📄"
        st.markdown(
            f'<div id="cite-{msg_idx}-{i}" style="padding: 4px 0;">'
            f'{emoji} <b>[{i}]</b> {c["doc_name"]} | {c["section_path"]} '
            f'(score: {c["relevance_score"]:.3f})</div>',
            unsafe_allow_html=True,
        )


def _render_citations_with_badges(citations: list[dict], msg_idx: int):
    """Render citations with RBAC access-level badges."""
    for i, c in enumerate(citations, 1):
        emoji = "📊" if c.get("element_type") == "table" else "📄"
        doc_name = c["doc_name"]
        level = get_access_level(doc_name)
        badge = _access_badge(level)
        st.markdown(
            f'<div id="cite-{msg_idx}-{i}" style="padding: 4px 0;">'
            f'{emoji} <b>[{i}]</b> {doc_name} {badge} | {c["section_path"]} '
            f'(score: {c["relevance_score"]:.3f})</div>',
            unsafe_allow_html=True,
        )


# ── Graph Explorer ──────────────────────────────────────────────

def _render_graph_explorer(kg: KnowledgeGraph):
    """Full-page interactive PyVis graph explorer."""
    st.header("🌐 Knowledge Graph Explorer")

    stats = kg.get_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", stats["total_nodes"])
    c2.metric("Edges", stats["total_edges"])
    c3.metric("Entities", stats["node_types"].get("entity", 0))
    c4.metric("Cross-doc entities", sum(
        1 for _, d in kg.graph.nodes(data=True)
        if d.get("cross_document")
    ))

    st.divider()

    # ── Filters ──
    col_f1, col_f2, col_f3 = st.columns(3)

    all_node_types = sorted(stats["node_types"].keys())
    with col_f1:
        selected_types = st.multiselect(
            "Node types to show",
            all_node_types,
            default=["entity", "chunk"],
            help="Showing all types on a large graph can be slow",
        )

    with col_f2:
        max_nodes = st.slider("Max nodes to render", 50, 2000, 300, step=50)

    with col_f3:
        # Optional: focus on a specific document
        doc_nodes = [
            d.get("name", n)
            for n, d in kg.graph.nodes(data=True)
            if d.get("node_type") == "document"
        ]
        focus_doc = st.selectbox("Focus on document (optional)", ["All"] + sorted(doc_nodes))

    # ── Build filtered subgraph ──
    if not selected_types:
        st.warning("Select at least one node type.")
        return

    # If focusing on a specific doc, restrict to that doc's subgraph
    filter_types = selected_types if selected_types else None

    if focus_doc != "All":
        # Get nodes belonging to this document
        doc_node_id = f"doc::{focus_doc}"
        focused_nodes = set()
        for n, d in kg.graph.nodes(data=True):
            ntype = d.get("node_type", "")
            if ntype not in (selected_types or all_node_types):
                continue
            if ntype == "document" and d.get("name") == focus_doc:
                focused_nodes.add(n)
            elif d.get("doc_name") == focus_doc:
                focused_nodes.add(n)
            elif d.get("source_doc") == focus_doc:
                focused_nodes.add(n)
            elif ntype == "entity":
                # Include entities connected to chunks from this doc
                for neighbor in list(kg.graph.successors(n)) + list(kg.graph.predecessors(n)):
                    nd = kg.graph.nodes.get(neighbor, {})
                    if nd.get("doc_name") == focus_doc:
                        focused_nodes.add(n)
                        break

        # Build a temporary subgraph
        sub = kg.graph.subgraph(focused_nodes).copy()
        temp_kg = KnowledgeGraph.__new__(KnowledgeGraph)
        temp_kg.config = kg.config
        temp_kg.graph = sub
        temp_kg.entity_id_map = {}
        temp_kg._VIS_STYLE = kg._VIS_STYLE
        net = temp_kg.to_pyvis(filter_node_types=filter_types, max_nodes=max_nodes)
    else:
        net = kg.to_pyvis(filter_node_types=filter_types, max_nodes=max_nodes)

    # ── Legend ──
    st.markdown(
        """
        **Legend:**
        🔴 **Entity** (dot) &nbsp;|&nbsp;
        🔵 **Chunk** (box) &nbsp;|&nbsp;
        🟢 **Document** (diamond) &nbsp;|&nbsp;
        🟠 **Section** (triangle)
        &nbsp;&nbsp;•&nbsp; Grey edges = structural &nbsp;|&nbsp; White edges = semantic relations
        &nbsp;&nbsp;•&nbsp; **Hover** any node or edge for details &nbsp;|&nbsp; **Scroll** to zoom &nbsp;|&nbsp; **Drag** to rearrange
        """,
        unsafe_allow_html=True,
    )

    # ── Render ──
    with st.spinner("Rendering graph..."):
        html = net.generate_html()
        # PyVis generates full HTML page; embed it via an iframe component
        components.html(html, height=780, scrolling=True)

    # ── Edge type breakdown ──
    with st.expander("📊 Edge type breakdown"):
        st.json(stats["edge_types"])


def main():
    # ── Login gate ──
    if "user" not in st.session_state:
        _render_login_page()
        return

    user: UserProfile = st.session_state.user

    st.title("🏦 Banking Knowledge AI")
    st.caption("Graph-grounded, hallucination-resistant knowledge assistant")

    # ── Initialize system ──
    result = init_system()
    if result[0] is None:
        st.error("System not initialized. Run `python build_index.py` first.")
        st.stop()
    retriever, kg, config = result

    # ── Top-level page selector ──
    page = st.sidebar.radio("Page", ["💬 Chat", "🌐 Graph Explorer"], index=0)

    # ── Sidebar ──
    with st.sidebar:
        # ── User profile card ──
        st.markdown("---")
        st.markdown(f"### {user.level_label}")
        st.markdown(f"**{user.display_name}** · `{user.role}`")
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        st.markdown("---")

        st.header("⚙️ Settings")

        mode = st.radio(
            "Retrieval Mode",
            ["🔗 Hybrid (Vector + Graph)", "📊 Vector-Only", "⚔️ A/B Comparison"],
            index=0,
            help="Compare both modes side-by-side to see the graph layer's impact",
        )
        use_graph = mode == "🔗 Hybrid (Vector + Graph)"
        ab_mode = mode == "⚔️ A/B Comparison"

        top_k = st.slider("Results to retrieve", 2, 15, 5)

        available_models = _discover_models(config)
        model_labels = [m["label"] for m in available_models]
        sel_idx = st.selectbox(
            "Answer Generation Model",
            range(len(model_labels)),
            format_func=lambda i: model_labels[i],
            index=0,
            help="Pick any configured Groq or locally-installed Ollama model",
        )
        _sel = available_models[sel_idx]
        llm_model = _sel["model"]
        llm_provider = _sel["provider"]

        # ── Document filter (RBAC-aware) ──
        accessible_docs = get_accessible_docs(user)
        doc_options = ["All accessible documents"] + [
            f"{d}  {_access_badge(get_access_level(d))}" for d in accessible_docs
        ]
        selected_doc_display = st.selectbox("📄 Document Filter", doc_options, index=0)
        # Extract raw doc name from display string (strip badge)
        if selected_doc_display == "All accessible documents":
            doc_filter = None
        else:
            doc_filter = selected_doc_display.split("  ")[0].strip()

        show_evidence = st.toggle("Show Evidence Details", value=True)
        show_graph_info = st.toggle("Show Graph Info", value=False)
        stream_answers = st.toggle("Stream Answers", value=True,
                                   help="Show tokens as they arrive from the LLM")

        st.divider()
        st.header("📊 System Info")

        if kg:
            stats = kg.get_stats()
            st.metric("Graph Nodes", stats["total_nodes"])
            st.metric("Graph Edges", stats["total_edges"])
            if stats.get("node_types"):
                st.json(stats["node_types"])
        st.metric("Accessible Docs", f"{len(accessible_docs)} / 30")

        mode_label = "🔗 Hybrid (Vector + Graph)" if use_graph else "📊 Vector-Only"
        st.info(f"Mode: {mode_label}")

    # ────────────────────────────────────────────────────────────
    # PAGE: Graph Explorer
    # ────────────────────────────────────────────────────────────
    if page == "🌐 Graph Explorer":
        _render_graph_explorer(kg)
        return

    # ────────────────────────────────────────────────────────────
    # PAGE: Chat
    # ────────────────────────────────────────────────────────────

    # ── Chat interface ──
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("citations"):
                st.markdown(_linkify_citations(msg["content"], msg_idx), unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("📎 Citations", expanded=True):
                    _render_citations(msg["citations"], msg_idx)
            if msg.get("evidence_details"):
                with st.expander("🔍 Evidence Details"):
                    st.json(msg["evidence_details"])

    # Chat input
    if prompt := st.chat_input("Ask a question about banking policies..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if ab_mode:
            # ── A/B Comparison Mode ──
            col_a, col_b = st.columns(2)

            with st.spinner("Running A/B comparison..."):
                retrieval_vector = retriever.retrieve(query=prompt, top_k=top_k, use_graph=False,
                                                      user_access_level=user.access_level, doc_filter=doc_filter)
                retrieval_hybrid = retriever.retrieve(query=prompt, top_k=top_k, use_graph=True,
                                                      user_access_level=user.access_level, doc_filter=doc_filter)
                resp_vector = generate_response(query=prompt, retrieval=retrieval_vector, config=config, model_override=llm_model, provider_override=llm_provider)
                resp_hybrid = generate_response(query=prompt, retrieval=retrieval_hybrid, config=config, model_override=llm_model, provider_override=llm_provider)

            with col_a:
                st.subheader("📊 Vector-Only")
                st.markdown(resp_vector.get("answer", "No response."))
                conf_a = resp_vector.get("confidence", 0)
                st.caption(f"Confidence: {conf_a:.2f}")

                with st.expander("Score Breakdown", expanded=True):
                    for i, r in enumerate(retrieval_vector.results, 1):
                        section = " > ".join(r.section_path) if r.section_path else "root"
                        st.markdown(
                            f"**[{i}]** {r.doc_name} | {section}\n\n"
                            f"  `vector={r.vector_score:.3f}` · "
                            f"`graph={r.graph_score:.3f}` · "
                            f"`policy={r.policy_score:.3f}` → "
                            f"**final={r.final_score:.3f}**"
                        )
                        if r.connected_entities:
                            st.caption(f"  Entities: {', '.join(r.connected_entities)}")

            with col_b:
                st.subheader("🔗 Hybrid (Vector + Graph)")
                st.markdown(resp_hybrid.get("answer", "No response."))
                conf_b = resp_hybrid.get("confidence", 0)
                st.caption(f"Confidence: {conf_b:.2f}")

                with st.expander("Score Breakdown", expanded=True):
                    for i, r in enumerate(retrieval_hybrid.results, 1):
                        section = " > ".join(r.section_path) if r.section_path else "root"
                        st.markdown(
                            f"**[{i}]** {r.doc_name} | {section}\n\n"
                            f"  `vector={r.vector_score:.3f}` · "
                            f"`graph={r.graph_score:.3f}` · "
                            f"`policy={r.policy_score:.3f}` → "
                            f"**final={r.final_score:.3f}**"
                        )
                        if r.connected_entities:
                            st.caption(f"  Entities: {', '.join(r.connected_entities)}")

            # Show ranking differences
            vec_chunks = [r.chunk_id for r in retrieval_vector.results]
            hyb_chunks = [r.chunk_id for r in retrieval_hybrid.results]
            only_in_hybrid = [cid for cid in hyb_chunks if cid not in vec_chunks]
            only_in_vector = [cid for cid in vec_chunks if cid not in hyb_chunks]
            reranked = vec_chunks != hyb_chunks

            st.divider()
            st.subheader("📈 Retrieval Analysis")

            # Quantitative metrics
            m1, m2, m3, m4 = st.columns(4)
            vec_docs = set(r.doc_name for r in retrieval_vector.results)
            hyb_docs = set(r.doc_name for r in retrieval_hybrid.results)
            hyb_entities = set()
            for r in retrieval_hybrid.results:
                hyb_entities.update(r.connected_entities)
            vec_entities = set()
            for r in retrieval_vector.results:
                vec_entities.update(r.connected_entities)

            m1.metric("Doc Coverage (Vec)", f"{len(vec_docs)} docs")
            m2.metric("Doc Coverage (Hyb)", f"{len(hyb_docs)} docs",
                       delta=f"+{len(hyb_docs) - len(vec_docs)}" if len(hyb_docs) > len(vec_docs) else None)
            m3.metric("Entities (Vec)", f"{len(vec_entities)}")
            m4.metric("Entities (Hyb)", f"{len(hyb_entities)}",
                       delta=f"+{len(hyb_entities) - len(vec_entities)}" if len(hyb_entities) > len(vec_entities) else None)

            # Query entity detection
            if retrieval_hybrid.query_entities:
                st.info(f"🔍 **Query entities detected:** {', '.join(retrieval_hybrid.query_entities)}")

            if only_in_hybrid:
                st.success(f"**Graph pulled in {len(only_in_hybrid)} chunk(s)** not found by vector search alone")
                for cid in only_in_hybrid:
                    r = next(x for x in retrieval_hybrid.results if x.chunk_id == cid)
                    st.markdown(f"  + **{r.doc_name}** | {' > '.join(r.section_path)} — "
                                f"graph={r.graph_score:.3f}, vector={r.vector_score:.3f}")
                    if r.connected_entities:
                        st.caption(f"    via entities: {', '.join(r.connected_entities)}")
            if only_in_vector:
                st.info(f"**{len(only_in_vector)} chunk(s) in vector-only** were displaced by graph-connected chunks")
            if reranked and not only_in_hybrid:
                st.info("**Same chunks retrieved, but re-ranked** by graph connectivity scores")
                for i, (vc, hc) in enumerate(zip(vec_chunks, hyb_chunks), 1):
                    if vc != hc:
                        st.caption(f"  Position {i}: vector had `{vc[:20]}...` → hybrid has `{hc[:20]}...`")
            if not reranked and not only_in_hybrid:
                st.warning("⚠️ No difference — try a multi-hop question that spans multiple policies")

            # Save to history (hybrid answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"**[A/B Comparison]**\n\n**Vector-Only:** {resp_vector.get('answer', '')}\n\n**Hybrid:** {resp_hybrid.get('answer', '')}",
                "citations": resp_hybrid.get("citations", []),
            })

        else:
            # ── Single mode ──
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    retrieval = retriever.retrieve(
                        query=prompt,
                        top_k=top_k,
                        use_graph=use_graph,
                        user_access_level=user.access_level,
                        doc_filter=doc_filter,
                    )

                if stream_answers:
                    # Streaming path — tokens rendered as they arrive
                    stream_resp = generate_response(
                        query=prompt,
                        retrieval=retrieval,
                        config=config,
                        model_override=llm_model,
                        provider_override=llm_provider,
                        stream=True,
                    )

                    raw_stream = stream_resp.get("stream")
                    citations = stream_resp.get("citations", [])
                    conf = stream_resp.get("confidence", 0)

                    if raw_stream is None:
                        # Evidence insufficient — no stream, just the refusal answer
                        answer = stream_resp.get("answer", "Insufficient evidence.")
                        st.markdown(answer)
                    else:
                        # Wrap the provider-specific stream into a plain text generator
                        def _token_iter(raw):
                            """Yield text tokens from either Groq (OpenAI) or Ollama streams."""
                            for chunk in raw:
                                # Groq / OpenAI-compatible
                                if hasattr(chunk, "choices"):
                                    delta = chunk.choices[0].delta
                                    if delta and delta.content:
                                        yield delta.content
                                # Ollama stream (Pydantic ChatResponse objects)
                                elif hasattr(chunk, "message"):
                                    content = getattr(chunk.message, "content", "")
                                    if content:
                                        yield content

                        answer = st.write_stream(_token_iter(raw_stream))

                    msg_idx = len(st.session_state.messages)

                    # Confidence badge
                    if conf > 0.7:
                        st.success(f"Confidence: {conf:.2f} ✓")
                    elif conf > 0.4:
                        st.warning(f"Confidence: {conf:.2f} ⚠")
                    else:
                        st.error(f"Confidence: {conf:.2f} — Low evidence")

                    # Citations with access badges
                    if citations and show_evidence:
                        with st.expander("📎 Citations", expanded=True):
                            _render_citations_with_badges(citations, msg_idx)

                    # Evidence paths
                    if show_evidence and retrieval.results:
                        _has_paths = any(r.evidence_path for r in retrieval.results)
                        if _has_paths:
                            with st.expander("🗺️ Evidence Paths"):
                                for r in retrieval.results:
                                    if r.evidence_path:
                                        path_str = " → ".join(
                                            f"[{s['from_type']}]{s['from'][:25]} --{s['relation']}--> [{s['to_type']}]{s['to'][:25]}"
                                            for s in r.evidence_path
                                        )
                                        st.markdown(f"**{r.doc_name}**: {path_str}")

                    # Graph info
                    if show_graph_info and use_graph:
                        with st.expander("🔗 Graph Traversal Info"):
                            st.markdown(f"**Query entities detected:** {', '.join(retrieval.query_entities) or 'None'}")
                            st.markdown(f"**Evidence sufficient:** {retrieval.evidence_sufficient}")
                            if retrieval.subgraph:
                                st.metric("Subgraph nodes", len(retrieval.subgraph.get("nodes", [])))
                                st.metric("Subgraph edges", len(retrieval.subgraph.get("edges", [])))

                    # Save to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer if isinstance(answer, str) else str(answer),
                        "citations": citations,
                    })

                else:
                    # Non-streaming path (original behavior)
                    response = generate_response(
                        query=prompt,
                        retrieval=retrieval,
                        config=config,
                        model_override=llm_model,
                        provider_override=llm_provider,
                    )

                    # Display answer
                    answer = response.get("answer", "No response generated.")
                    citations = response.get("citations", [])
                    msg_idx = len(st.session_state.messages)

                    if citations:
                        st.markdown(_linkify_citations(answer, msg_idx), unsafe_allow_html=True)
                    else:
                        st.markdown(answer)

                    # Confidence badge
                    conf = response.get("confidence", 0)
                    if conf > 0.7:
                        st.success(f"Confidence: {conf:.2f} ✓")
                    elif conf > 0.4:
                        st.warning(f"Confidence: {conf:.2f} ⚠")
                    else:
                        st.error(f"Confidence: {conf:.2f} — Low evidence")

                    # Citations with access badges
                    if citations and show_evidence:
                        with st.expander("📎 Citations", expanded=True):
                            _render_citations_with_badges(citations, msg_idx)

                    # Evidence details
                    if show_evidence:
                        retrieval_results = response.get("retrieval_results", [])
                        if retrieval_results:
                            with st.expander("🔍 Retrieved Evidence"):
                                for i, r in enumerate(retrieval_results, 1):
                                    st.markdown(f"**Evidence {i}** (Score: {r['final_score']:.3f})")
                                    st.markdown(f"*Source: {r['doc_name']} | {' > '.join(r['section_path'])}*")
                                    st.code(r["content"][:300] + ("..." if len(r["content"]) > 300 else ""))
                                    if r.get("connected_entities"):
                                        st.caption(f"Connected entities: {', '.join(r['connected_entities'])}")
                                    st.divider()

                    # Evidence paths
                    if show_evidence and retrieval.results:
                        _has_paths = any(r.evidence_path for r in retrieval.results)
                        if _has_paths:
                            with st.expander("🗺️ Evidence Paths"):
                                for r in retrieval.results:
                                    if r.evidence_path:
                                        path_str = " → ".join(
                                            f"[{s['from_type']}]{s['from'][:25]} --{s['relation']}--> [{s['to_type']}]{s['to'][:25]}"
                                            for s in r.evidence_path
                                        )
                                        st.markdown(f"**{r.doc_name}**: {path_str}")

                    # Graph info
                    if show_graph_info and use_graph:
                        with st.expander("🔗 Graph Traversal Info"):
                            st.markdown(f"**Query entities detected:** {', '.join(retrieval.query_entities) or 'None'}")
                            st.markdown(f"**Evidence sufficient:** {retrieval.evidence_sufficient}")
                            if retrieval.subgraph:
                                st.metric("Subgraph nodes", len(retrieval.subgraph.get("nodes", [])))
                                st.metric("Subgraph edges", len(retrieval.subgraph.get("edges", [])))

                    # Save to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "evidence_details": retrieval_results[:3] if show_evidence else None,
                    })


if __name__ == "__main__":
    main()
