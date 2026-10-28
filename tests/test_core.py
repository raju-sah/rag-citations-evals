from ragcite.citations import parse_citations, validate_citations
from ragcite.evals.metrics import mrr, ndcg_at_k, recall_at_k


def test_parse_citations():
    ans = "Paris is the capital [1]. It has the Eiffel Tower [2, 3]. Nice city."
    sents = parse_citations(ans)
    assert [s.citations for s in sents] == [[1], [2, 3], []]


def test_validate_citations():
    rep = validate_citations("A is B [1]. C is D [7].", num_sources=5)
    assert rep["invalid_citations"] == [7]
    assert rep["structural_citation_recall"] == 1.0
    assert not rep["valid"]


def test_retrieval_metrics():
    retrieved = ["c3", "c1", "c9"]
    gold = {"c1"}
    assert recall_at_k(retrieved, gold, 1) == 0.0
    assert recall_at_k(retrieved, gold, 2) == 1.0
    assert mrr(retrieved, gold) == 0.5
    assert 0 < ndcg_at_k(retrieved, gold, 3) < 1
