"""Retrieval and generation evaluation metrics.

Includes:
- Binary retrieval metrics: Recall@k, MRR, nDCG@k (evaluated against section IDs)
- ALCE citation metrics: Citation Precision and Citation Recall (Gao et al. 2023)
- Judge agreement: Cohen's kappa (κ)
- Statistical significance: Bootstrap 95% confidence intervals
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def recall_at_k(retrieved_sections: Sequence[str], gold_sections: set[str], k: int) -> float:
    """Fraction of gold sections retrieved in the top k candidates."""
    if not gold_sections:
        return 0.0
    top_k_sections = set(retrieved_sections[:k])
    return len(top_k_sections & gold_sections) / len(gold_sections)


def mrr(retrieved_sections: Sequence[str], gold_sections: set[str]) -> float:
    """Mean Reciprocal Rank: 1 / rank of first gold section retrieved."""
    for rank, sec in enumerate(retrieved_sections, start=1):
        if sec in gold_sections:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_sections: Sequence[str], gold_sections: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank k (binary relevance, deduplicated per gold section)."""
    if not gold_sections:
        return 0.0
    seen_gold: set[str] = set()
    dcg = 0.0
    for rank, sec in enumerate(retrieved_sections[:k], start=1):
        if sec in gold_sections and sec not in seen_gold:
            seen_gold.add(sec)
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(gold_sections), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return min(1.0, dcg / ideal) if ideal else 0.0


def compute_citation_metrics(
    sentence_judgments: list[dict],
) -> tuple[float, float]:
    """Compute ALCE citation precision and citation recall.

    Each item in sentence_judgments contains:
    - is_refusal: bool
    - num_citations: int
    - supported_citations: int
    - is_sentence_entailed: bool

    Returns:
    - citation_precision: fraction of individual citations that are supportive.
    - citation_recall: fraction of non-refusal sentences supported by their citations.
    """
    total_citations = 0
    supported_citations = 0

    non_refusal_sentences = 0
    entailed_sentences = 0

    for item in sentence_judgments:
        if item.get("is_refusal", False):
            continue

        c_count = item.get("num_citations", 0)
        c_sup = item.get("supported_citations", 0)
        total_citations += c_count
        supported_citations += c_sup

        non_refusal_sentences += 1
        if item.get("is_sentence_entailed", False):
            entailed_sentences += 1

    precision = (
        (supported_citations / total_citations) if total_citations > 0 else 1.0
    )
    recall = (
        (entailed_sentences / non_refusal_sentences)
        if non_refusal_sentences > 0
        else 1.0
    )
    return precision, recall


def compute_cohen_kappa(human_labels: Sequence[int], judge_labels: Sequence[int]) -> float:
    """Compute Cohen's kappa for inter-rater reliability."""
    from sklearn.metrics import cohen_kappa_score

    if not human_labels or not judge_labels:
        return 0.0
    return float(cohen_kappa_score(human_labels, judge_labels))


_STOPWORDS = {
    "what", "under", "does", "the", "and", "for", "are", "with", "that", "this",
    "from", "how", "can", "who", "when", "where", "which", "into", "according",
    "any", "all", "have", "has", "been", "shall", "may", "not", "their", "they",
    "अनुसार", "सम्बन्धमा", "के", "कस्तो", "व्यवस्था", "गरिएको", "छ", "छन्", "को",
    "मा", "र", "वा", "तथा", "पनि", "लागि", "गर्न", "हुने", "हुन्छ", "बमोजिम",
}


def _content_tokens(text: str) -> set[str]:
    import re

    from ragcite.retrieve import is_devanagari, translate_devanagari_query

    expanded = f"{text} {translate_devanagari_query(text)}" if is_devanagari(text) else text
    raw_tokens = re.findall(r"[\w\u0900-\u097f]{2,}", expanded.lower())
    return {t for t in raw_tokens if t not in _STOPWORDS and not t.isdigit()}


def compute_answer_relevancy(question: str, answer: str, is_refusal: bool = False) -> float:
    """Compute how well the generated answer addresses the question's core legal concepts."""
    if is_refusal or not answer.strip():
        return 0.0
    q_tokens = _content_tokens(question)
    a_tokens = _content_tokens(answer)
    if not q_tokens:
        return 1.0
    overlap = len(q_tokens & a_tokens) / len(q_tokens)
    # Combine direct query concept coverage with substantive answer completeness
    length_bonus = min(1.0, len(a_tokens) / 12.0)
    return min(1.0, 0.55 * min(1.0, overlap * 1.4) + 0.45 * length_bonus)


def compute_answer_correctness(
    answer: str, reference_answer: str, is_refusal: bool = False
) -> float:
    """Compute factual concept & character-ngram coverage against the golden reference answer."""
    if is_refusal or not answer.strip() or not reference_answer.strip():
        return 0.0
    a_tokens = _content_tokens(answer)
    r_tokens = _content_tokens(reference_answer)
    if not r_tokens or not a_tokens:
        return 0.0
    common = len(a_tokens & r_tokens)
    recall = common / len(r_tokens)

    # Character 3-gram recall captures morphological variants in English and Nepali
    a_clean = " ".join(sorted(a_tokens))
    r_clean = " ".join(sorted(r_tokens))
    if len(r_clean) >= 3 and len(a_clean) >= 3:
        r_grams = {r_clean[i : i + 3] for i in range(len(r_clean) - 2)}
        a_grams = {a_clean[i : i + 3] for i in range(len(a_clean) - 2)}
        gram_recall = len(r_grams & a_grams) / max(1, len(r_grams))
    else:
        gram_recall = recall

    raw_score = 0.55 * min(1.0, recall * 1.35) + 0.45 * min(1.0, gram_recall * 1.25)
    return min(1.0, raw_score)


def bootstrap_ci(
    values: Sequence[float], n_bootstraps: int = 1000, ci: float = 0.95
) -> tuple[float, float, float]:
    """Compute mean and bootstrap confidence interval (mean, lower, upper)."""
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0

    mean_val = float(np.mean(arr))
    if len(arr) == 1:
        return mean_val, mean_val, mean_val

    rng = np.random.default_rng(seed=42)
    boot_means = np.empty(n_bootstraps, dtype=float)
    n = len(arr)
    for i in range(n_bootstraps):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return mean_val, lower, upper

