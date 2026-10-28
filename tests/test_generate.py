import tempfile

from ragcite.generate import LLMClient, answer_with_citations, format_prompt
from ragcite.retrieve import RetrievedCandidate


def _make_sample_sources():
    return [
        RetrievedCandidate(
            chunk_id="c1",
            doc_id="constitution",
            section_id="constitution:art17",
            header_path="Constitution of Nepal (2015) > Article 17",
            text="Every citizen shall have freedom of speech.",
            page_start=15,
            page_end=15,
            score=0.9,
        ),
        RetrievedCandidate(
            chunk_id="c2",
            doc_id="labour_act",
            section_id="labour_act:s10",
            header_path="Labour Act 2017 > Section 10",
            text="Employment contracts must be executed in writing.",
            page_start=6,
            page_end=6,
            score=0.85,
        ),
    ]


def test_format_prompt():
    sources = _make_sample_sources()
    prompt = format_prompt("What rights exist?", sources)
    assert "[1] Source: Constitution of Nepal (2015) > Article 17 (Pages 15–15)" in prompt
    assert "[2] Source: Labour Act 2017 > Section 10 (Pages 6–6)" in prompt
    assert "QUESTION: What rights exist?" in prompt


def test_llm_client_caching():
    with tempfile.TemporaryDirectory() as tmp_dir:
        client = LLMClient(
            provider="gemini",
            model="gemini-2.5-flash",
            cache_dir=tmp_dir,
            api_key=None,  # triggers fallback mock
        )
        prompt = "Test prompt"
        ans1, cached1, _ = client.generate(prompt)
        assert cached1 is False
        assert len(ans1) > 0

        # Second call should be cached
        ans2, cached2, _ = client.generate(prompt)
        assert cached2 is True
        assert ans1 == ans2


def test_answer_with_citations():
    with tempfile.TemporaryDirectory() as tmp_dir:
        sources = _make_sample_sources()
        result = answer_with_citations(
            "What rights exist?",
            sources,
            cache_dir=tmp_dir,
            api_key=None,
        )
        assert result.question == "What rights exist?"
        assert len(result.sources) == 2
        assert len(result.answer) > 0
