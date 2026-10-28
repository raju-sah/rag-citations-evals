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


def test_nepali_sentence_citations():
    ans = "नेपालको संविधान सर्वोच्च कानून हो [1]। यसमा मौलिक हकको व्यवस्था छ [2]।"
    sents = parse_citations(ans)
    assert len(sents) == 2
    assert sents[0].citations == [1]
    assert sents[1].citations == [2]


def test_refusal_validation():
    refusal_ans = "The provided sources do not contain sufficient information to answer this question."
    rep = validate_citations(refusal_ans, num_sources=5)
    assert rep["is_refusal"] is True
    assert rep["valid"] is True
    assert rep["structural_citation_recall"] == 1.0



def test_retrieval_metrics():
    retrieved = ["c3", "c1", "c9"]
    gold = {"c1"}
    assert recall_at_k(retrieved, gold, 1) == 0.0
    assert recall_at_k(retrieved, gold, 2) == 1.0
    assert mrr(retrieved, gold) == 0.5
    assert 0 < ndcg_at_k(retrieved, gold, 3) < 1
