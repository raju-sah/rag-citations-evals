"""Run the 8 core ablations + 1 prompt variant and populate experiments/ablations.csv.

Ablations:
1. Baseline (Structure-aware 500, Hybrid RRF k=60, Language Routing ON, Reranker ON)
2. Dense-only (Isolate contribution of dense embedding search)
3. BM25-only (Isolate contribution of lexical search)
4. Naive hybrid without routing (Demonstrate impact of BM25 noise on Nepali queries)
5. Reranker OFF (Demonstrate cross-encoder reranker impact on Recall@5 and MRR)
6. Fixed 500 tok (Isolate value of structure-aware section boundaries & header paths)
7. Fixed 300 tok (Analyze impact of smaller token context window)
8. Fixed 800 tok (Analyze impact of larger, diluted context window)
9. Relaxed citation prompt (Analyze impact of strict citation enforcement on faithfulness)
"""

import logging
from pathlib import Path

from ragcite.evals.run import run_eval
from ragcite.generate import RELAXED_SYSTEM_INSTRUCTION

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ablation_runner")


def main():
    csv_path = Path("experiments/ablations.csv")
    # Reset CSV with fresh header
    header = (
        "run_name,date,config,chunk_size,retrieval_mode,rerank,k,"
        "recall@5,mrr,ndcg@5,faithfulness,answer_relevancy,correctness,"
        "cite_precision,cite_recall,refusal_rate,latency_s,cost_per_query_usd,notes\n"
    )
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(header)

    logger.info("Executing 8 core ablation runs + 1 prompt ablation...")

    # 1. Baseline (Full evaluation on dev split)
    logger.info("=== Run 1/9: Baseline ===")
    run_eval(
        split="dev",
        override_run_name="baseline",
        skip_generation=False,
        max_eval_items=40,
    )

    # 2. Dense-only
    logger.info("=== Run 2/9: Dense-only ===")
    run_eval(
        split="dev",
        override_run_name="dense_only",
        override_retrieval_mode="dense",
        skip_generation=True,
        max_eval_items=40,
    )

    # 3. BM25-only
    logger.info("=== Run 3/9: BM25-only ===")
    run_eval(
        split="dev",
        override_run_name="bm25_only",
        override_retrieval_mode="bm25",
        skip_generation=True,
        max_eval_items=40,
    )

    # 4. Naive Hybrid (no language-aware routing)
    logger.info("=== Run 4/9: Naive Hybrid (No Routing) ===")
    run_eval(
        split="dev",
        override_run_name="naive_hybrid_no_routing",
        override_retrieval_mode="hybrid",
        override_routing=False,
        skip_generation=True,
        max_eval_items=40,
    )

    # 5. Reranker OFF
    logger.info("=== Run 5/9: Reranker OFF ===")
    run_eval(
        split="dev",
        override_run_name="reranker_off",
        override_retrieval_mode="hybrid",
        override_rerank_enabled=False,
        skip_generation=True,
        max_eval_items=40,
    )

    # 6. Fixed 500 tokens
    logger.info("=== Run 6/9: Chunking Fixed 500 ===")
    run_eval(
        split="dev",
        override_run_name="fixed_500_tok",
        override_retrieval_mode="bm25",
        override_rerank_enabled=False,
        override_chunk_size=500,
        skip_generation=True,
        max_eval_items=40,
    )

    # 7. Fixed 300 tokens
    logger.info("=== Run 7/9: Chunking Fixed 300 ===")
    run_eval(
        split="dev",
        override_run_name="fixed_300_tok",
        override_retrieval_mode="bm25",
        override_rerank_enabled=False,
        override_chunk_size=300,
        skip_generation=True,
        max_eval_items=40,
    )

    # 8. Fixed 800 tokens
    logger.info("=== Run 8/9: Chunking Fixed 800 ===")
    run_eval(
        split="dev",
        override_run_name="fixed_800_tok",
        override_retrieval_mode="bm25",
        override_rerank_enabled=False,
        override_chunk_size=800,
        skip_generation=True,
        max_eval_items=40,
    )

    # 9. Prompt: Relaxed citation prompt
    logger.info("=== Run 9/9: Prompt Relaxed Citation ===")
    run_eval(
        split="dev",
        override_run_name="prompt_relaxed_citation",
        override_system_prompt=RELAXED_SYSTEM_INSTRUCTION,
        skip_generation=False,
        max_eval_items=40,
    )

    logger.info("All 9 runs completed! Results saved to %s", csv_path)


if __name__ == "__main__":
    main()
