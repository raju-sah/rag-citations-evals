"""Streamlit 2-tab application:
Tab 1: Interactive Legal Q&A Demo with inline citation badges and expandable sources drawer.
Tab 2: Ablations & Evaluation Dashboard rendering experiments/ablations.csv.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ragcite.citations import validate_citations
from ragcite.generate import (
    GenerationResult,
    answer_with_citations,
    check_conversational_query,
)
from ragcite.rerank import get_reranker, rerank
from ragcite.retrieve import HybridRetriever

st.set_page_config(
    page_title="RAG with Citations — Nepal Legal Corpus",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom badge styling
st.markdown(
    """
    <style>
    .cite-badge {
        background-color: #2e5b88;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85em;
        margin: 0 2px;
        display: inline-block;
    }
    .status-pill-ok {
        background-color: #e6f4ea;
        color: #137333;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 500;
        font-size: 0.9em;
        border: 1px solid #ceead6;
        display: inline-block;
        margin-bottom: 12px;
    }
    .status-pill-warn {
        background-color: #fef7e0;
        color: #b06000;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 500;
        font-size: 0.9em;
        border: 1px solid #feefc3;
        display: inline-block;
        margin-bottom: 12px;
    }
    /* Hide Deploy button */
    .stAppDeployButton,
    div[data-testid="stAppDeployButton"] {
        display: none !important;
        visibility: hidden !important;
    }
    /* Claude-style embedded chat input (seamless borderless design) */
    div[data-testid="stChatInput"] {
        border-radius: 18px !important;
        border: none !important;
        outline: none !important;
        background-color: #212126 !important;
        box-shadow: none !important;
        padding: 4px 8px !important;
        margin-bottom: 12px !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    div[data-testid="stChatInput"] > div {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    div[data-testid="stChatInput"] textarea {
        font-size: 1.02rem !important;
        line-height: 1.5 !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    div[data-testid="stChatInput"] textarea:focus {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"] {
        background-color: #d97706 !important;
        color: white !important;
        border-radius: 50% !important;
        width: 34px !important;
        height: 34px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: transform 0.15s ease, background-color 0.15s ease !important;
    }
    div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"]:hover:not(:disabled) {
        background-color: #f59e0b !important;
        transform: scale(1.06) !important;
    }
    div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"]:disabled {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: rgba(255, 255, 255, 0.25) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_retriever():
    index_dir = Path("data/index")
    if not (index_dir / "bm25").exists() or not (index_dir / "faiss.index").exists():
        return None
    try:
        ret = HybridRetriever(index_dir=index_dir)
        _ = ret.embedder  # Pre-warm query embedder in memory
        return ret
    except (OSError, ValueError, RuntimeError) as err:
        st.sidebar.error(f"Failed to load retriever: {err}")
        return None


@st.cache_resource(show_spinner=False)
def get_cached_reranker():
    try:
        rr = get_reranker()
        _ = rr.model  # Pre-warm fast CPU cross-encoder in memory
        return rr
    except Exception:  # noqa: BLE001
        return None


def _render_with_citation_badges(text: str) -> str:
    """Convert inline [1] or [1, 2] markers into styled HTML citation badges."""
    return re.sub(
        r"\[(\d+(?:\s*,\s*\d+)*)\]",
        r"<span class='cite-badge'>[\1]</span>",
        text,
    )


tab_qa, tab_eval = st.tabs(["⚖️ Interactive Q&A Demo", "📊 Ablations & Metrics Dashboard"])

with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        options=["hybrid", "dense", "bm25"],
        format_func=lambda x: {
            "hybrid": "Hybrid (RRF k=60)",
            "dense": "Dense Only (FAISS)",
            "bm25": "Sparse Only (BM25)",
        }[x],
        index=0,
    )
    reranker_enabled = st.toggle("Enable Cross-Encoder Reranker", value=True)
    route_devanagari = st.toggle("Language-Aware Routing (Devanagari → EN)", value=True)
    top_k_candidates = st.slider("Top Candidates Fetched", min_value=5, max_value=30, value=15)
    final_k = st.slider("Final Context Chunks (k)", min_value=1, max_value=10, value=5)

    st.markdown("---")
    st.markdown("**Corpus Scope:** Nepal Law")
    st.caption("Constitution 2015 · Labour Act · Muluki Codes · FITTA · NHRC · Corruption Act")

# -------------------------------------------------------------
# TAB 1: Interactive Q&A Demo
# -------------------------------------------------------------
with tab_qa:
    st.subheader("Nepal Legal Assistant with Sentence-Level Citations")
    st.caption(
        "Ask a legal question in English or Nepali. The pipeline retrieves relevant acts, "
        "enforces sentence-level citations `[n]`, and validates citation provenance."
    )

    example_queries = [
        "What freedoms are guaranteed to citizens under Article 17 of the Constitution of Nepal?",
        "नेपालको संविधानको धारा १६ मा सम्मानपूर्वक बाँच्न पाउने हक सम्बन्धमा के व्यवस्था गरिएको छ?",
        "Under what conditions does the Constitution prohibit capital punishment?",
        "संविधानको धारा ३१ अनुसार शिक्षा सम्बन्धी हकमा के के पर्दछन्?",
        "What does the Constitution of Nepal state regarding the licensing of cryptocurrency exchanges and token staking?",
    ]

    # Initialize query and result state
    if "query_input_val" not in st.session_state:
        st.session_state["query_input_val"] = ""
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    def apply_chip():
        chosen = st.session_state.get("chip_selection")
        if chosen:
            st.session_state["query_input_val"] = chosen

    retriever = get_retriever()
    _ = get_cached_reranker()
    if retriever is None:
        st.warning(
            "⚠️ Retrieval indexes not yet found in `data/index/`. "
            "Please ensure `data/processed/chunks.jsonl` is indexed."
        )

    # Embedded input container with submit button inside the textarea
    with st.container():
        submitted_query = st.chat_input(
            "Enter your legal question in English or Nepali...",
            key="query_input_val",
            disabled=(retriever is None),
        )

    # Chip-style example questions below the input field
    st.markdown("💡 **Suggested questions (click to fill):**")
    st.pills(
        "Suggested questions",
        options=example_queries,
        key="chip_selection",
        on_change=apply_chip,
        label_visibility="collapsed",
        wrap=True,
    )

    if submitted_query:
        t_pipeline_start = time.perf_counter()
        conv_reply = check_conversational_query(submitted_query)
        if conv_reply is not None:
            t_total = time.perf_counter() - t_pipeline_start
            st.session_state["last_result"] = {
                "question": submitted_query,
                "is_conversational": True,
                "candidates": [],
                "final_sources": [],
                "gen_result": GenerationResult(
                    question=submitted_query,
                    answer=conv_reply,
                    sources=[],
                    model="instant-router",
                    provider="local",
                    cached=True,
                    latency_s=t_total,
                    prompt="",
                ),
                "val_report": {
                    "num_sentences": 0,
                    "num_citations": 0,
                    "invalid_citations": [],
                    "uncited_sentences": [],
                    "structural_citation_recall": 1.0,
                    "structural_citation_precision": 1.0,
                    "is_refusal": False,
                    "valid": True,
                },
                "total_latency_s": t_total,
            }
        else:
            with st.spinner("Retrieving candidates & generating cited answer..."):
                # 1. Retrieve
                candidates = retriever.retrieve(
                    query=submitted_query,
                    top_k=top_k_candidates,
                    mode=retrieval_mode,
                    route_devanagari=route_devanagari,
                )

                # 2. Rerank
                if reranker_enabled and candidates:
                    final_sources = rerank(
                        query=submitted_query,
                        candidates=candidates,
                        top_k=final_k,
                    )
                else:
                    final_sources = candidates[:final_k]

                # 3. Generate with citations
                gen_result = answer_with_citations(
                    question=submitted_query,
                    sources=final_sources,
                )

                # 4. Structural validation
                val_report = validate_citations(gen_result.answer, num_sources=len(final_sources))
                t_total = time.perf_counter() - t_pipeline_start

                st.session_state["last_result"] = {
                    "question": submitted_query,
                    "is_conversational": False,
                    "candidates": candidates,
                    "final_sources": final_sources,
                    "gen_result": gen_result,
                    "val_report": val_report,
                    "total_latency_s": t_total,
                }

    # Render results if available
    if st.session_state.get("last_result") is not None:
        res = st.session_state["last_result"]
        is_conv = res.get("is_conversational", False)
        candidates = res["candidates"]
        final_sources = res["final_sources"]
        gen_result = res["gen_result"]
        val_report = res["val_report"]
        total_latency_s = res.get("total_latency_s", gen_result.latency_s)

        # Render citation validation status
        if is_conv:
            st.markdown(
                "<div class='status-pill-ok'>💬 Instant Conversational Assistant (No retrieval needed for greeting)</div>",
                unsafe_allow_html=True,
            )
        elif val_report["is_refusal"]:
            st.markdown(
                "<div class='status-pill-warn'>🛡️ Refusal Triggered: Model recognized question is not answerable from sources.</div>",
                unsafe_allow_html=True,
            )
        elif val_report["valid"]:
            st.markdown(
                f"<div class='status-pill-ok'>✓ All {val_report['num_sentences']} sentences cited ({val_report['num_citations']} total citations). Zero hallucinated refs.</div>",
                unsafe_allow_html=True,
            )
        else:
            uncited_count = len(val_report["uncited_sentences"])
            invalid_count = len(val_report["invalid_citations"])
            st.markdown(
                f"<div class='status-pill-warn'>⚠️ Citation Warning: {uncited_count} uncited sentences, {invalid_count} out-of-bounds citations.</div>",
                unsafe_allow_html=True,
            )

        # Render Answer with styled citation badges
        st.markdown("### Answer")
        st.markdown(_render_with_citation_badges(gen_result.answer), unsafe_allow_html=True)

        # Provenance metadata
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Retrieved Candidates", len(candidates))
        col_m2.metric("Final Sources Cited", len(final_sources))
        col_m3.metric("End-to-End Latency", f"{total_latency_s:.2f}s")
        col_m4.metric("LLM Cached", "Yes" if gen_result.cached else "No")

        # Sources drawer
        if final_sources:
            st.markdown("### Retrieved Sources Drawer")
            for idx, src in enumerate(final_sources, start=1):
                with st.expander(f"[{idx}] {src.header_path} (Pages {src.page_start}–{src.page_end})", expanded=(idx <= 2)):
                    st.markdown(f"**Section ID:** `{src.section_id}` | **Score:** `{src.score:.4f}`")
                    st.text_area(
                        label=f"Source Text [{idx}]",
                        value=src.text,
                        height=180,
                        key=f"src_view_{idx}",
                        disabled=True,
                    )

# -------------------------------------------------------------
# TAB 2: Ablations & Metrics Dashboard
# -------------------------------------------------------------
with tab_eval:
    st.subheader("Evaluation Results & Ablation Table")
    st.caption("Live view of `experiments/ablations.csv` tracking retrieval, generation, and citation metrics.")

    csv_path = Path("experiments/ablations.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if not df.empty and len(df) > 0:
            st.dataframe(df, width="stretch")
            # High-level metrics view
            st.markdown("#### Primary Headline Metrics")
            cols = st.columns(min(len(df.columns), 4))
            for i, col_name in enumerate(["recall@5", "faithfulness", "cite_precision", "cite_recall"]):
                if col_name in df.columns:
                    val = df[col_name].iloc[-1]
                    cols[i].metric(col_name.upper(), f"{val:.3f}" if isinstance(val, (int, float)) else str(val))
        else:
            st.info("No experiment rows in `experiments/ablations.csv` yet. Runs will appear here in Phase 2.")
    else:
        st.info("`experiments/ablations.csv` not found. Evaluator will generate it during Phase 2.")
