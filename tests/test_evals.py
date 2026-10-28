from pathlib import Path

from ragcite.evals.judge import LLMJudge
from ragcite.evals.metrics import (
    bootstrap_ci,
    compute_citation_metrics,
    compute_cohen_kappa,
    mrr,
    ndcg_at_k,
    recall_at_k,
)


def test_retrieval_metrics_advanced():
    retrieved = ["doc:s1", "doc:s2", "doc:s3", "doc:s4", "doc:s5"]
    gold = {"doc:s2", "doc:s5"}

    # Recall
    assert recall_at_k(retrieved, gold, k=1) == 0.0
    assert recall_at_k(retrieved, gold, k=2) == 0.5
    assert recall_at_k(retrieved, gold, k=5) == 1.0

    # MRR (first hit is at rank 2)
    assert mrr(retrieved, gold) == 0.5

    # NDCG
    score_k5 = ndcg_at_k(retrieved, gold, k=5)
    assert 0.0 < score_k5 <= 1.0


def test_citation_metrics():
    judgments = [
        {"is_refusal": False, "num_citations": 2, "supported_citations": 2, "is_sentence_entailed": True},
        {"is_refusal": False, "num_citations": 2, "supported_citations": 1, "is_sentence_entailed": True},
        {"is_refusal": False, "num_citations": 1, "supported_citations": 0, "is_sentence_entailed": False},
        {"is_refusal": True, "num_citations": 0, "supported_citations": 0, "is_sentence_entailed": False},
    ]
    p, r = compute_citation_metrics(judgments)
    # total citations = 5, supported = 3 -> precision = 3/5 = 0.6
    assert round(p, 2) == 0.60
    # non-refusal sentences = 3, entailed sentences = 2 -> recall = 2/3 = 0.67
    assert round(r, 2) == 0.67


def test_bootstrap_ci():
    vals = [0.8, 0.82, 0.85, 0.79, 0.88, 0.84]
    mean_val, low, high = bootstrap_ci(vals, n_bootstraps=200, ci=0.95)
    assert low <= mean_val <= high


def test_cohen_kappa():
    human = [1, 1, 0, 0, 1, 0]
    judge = [1, 1, 0, 0, 1, 0]
    perfect_kappa = compute_cohen_kappa(human, judge)
    assert perfect_kappa == 1.0


def test_judge_calibration():
    calib_file = Path("evals/golden/calibration_40.jsonl")
    if calib_file.exists():
        judge = LLMJudge()
        results = judge.run_calibration(calib_file)
        assert results["num_items"] == 40
        assert "cohen_kappa_overall" in results
        assert "accuracy" in results
        assert results["accuracy"] > 0.0
