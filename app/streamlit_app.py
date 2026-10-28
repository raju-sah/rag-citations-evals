"""Streamlit 2-tab application:
Tab 1: Interactive Legal Q&A Demo with inline citation badges and expandable sources drawer.
Tab 2: Ablations & Evaluation Dashboard rendering experiments/ablations.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ragcite.citations import validate_citations
from ragcite.generate import answer_with_citations
from ragcite.rerank import rerank
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
        return HybridRetriever(index_dir=index_dir)
    except (OSError, ValueError, RuntimeError) as err:
        st.sidebar.error(f"Failed to load retriever: {err}")
        return None


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
    route_devanagari = st.toggle("Language-Aware Routing (Devanagari → Dense)", value=True)
    top_k_candidates = st.slider("Top Candidates Fetched", min_value=5, max_value=30, value=20)
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
        "What fundamental freedoms are guaranteed under Article 17 of the Constitution?",
        "नेपालको संविधानको धारा १६ मा सम्मानपूर्वक बाँच्न पाउने हक सम्बन्धी के व्यवस्था छ?",
        "Under the Labour Act 2017, what are the mandatory types of employment contracts?",
        "What penalties apply to a public servant convicted of taking bribery?",
        "What does the Consumer Protection Act say about space travel liability and passenger rockets?",
    ]

    selected_example = st.selectbox("Or choose an example query:", [""] + example_queries, index=0)
    query_input = st.text_input(
        "Enter your legal question:",
        value=selected_example if selected_example else "",
        placeholder="e.g. What does Article 17 say about freedom of expression?",
    )

    retriever = get_retriever()
    if retriever is None:
        st.warning(
            "⚠️ Retrieval indexes not yet found in `data/index/`. "
            "Please ensure `data/processed/chunks.jsonl` is indexed."
        )

    if st.button("Run RAG Pipeline", type="primary", disabled=(retriever is None or not query_input.strip())):
        with st.spinner("Retrieving candidates & generating cited answer..."):
            # 1. Retrieve
            candidates = retriever.retrieve(
                query=query_input,
                top_k=top_k_candidates,
                mode=retrieval_mode,
                route_devanagari=route_devanagari,
            )

            # 2. Rerank
            if reranker_enabled and candidates:
                final_sources = rerank(
                    query=query_input,
                    candidates=candidates,
                    top_k=final_k,
                )
            else:
                final_sources = candidates[:final_k]

            # 3. Generate with citations
            gen_result = answer_with_citations(
                question=query_input,
                sources=final_sources,
            )

            # 4. Structural validation
            val_report = validate_citations(gen_result.answer, num_sources=len(final_sources))

        # Render citation validation status
        if val_report["is_refusal"]:
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

        # Render Answer
        st.markdown("### Answer")
        st.write(gen_result.answer)

        # Provenance metadata
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Retrieved Candidates", len(candidates))
        col_m2.metric("Final Sources Cited", len(final_sources))
        col_m3.metric("Latency", f"{gen_result.latency_s:.2f}s")
        col_m4.metric("LLM Cached", "Yes" if gen_result.cached else "No")

        # Sources drawer
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
            st.dataframe(df, use_container_width=True)
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
