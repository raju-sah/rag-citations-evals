import tempfile
from pathlib import Path

from ragcite.chunk import (
    Chunk,
    chunk_fixed_window,
    chunk_structure_aware,
    estimate_tokens,
    load_chunks_jsonl,
    save_chunks_jsonl,
)
from ragcite.ingest import DocumentPage


def test_estimate_tokens():
    en_text = "The Constitution of Nepal is the supreme law."
    ne_text = "नेपालको संविधान सर्वोच्च कानून हो।"
    assert estimate_tokens(en_text) > 0
    assert estimate_tokens(ne_text) > 0


def test_chunk_structure_aware():
    sample_text = (
        "Part-3 Fundamental Rights\n"
        "16. Right to live with dignity:\n"
        "(1) Each person shall have the right to live with dignity.\n"
        "(2) No law shall be made providing for death penalty.\n\n"
        "17. Right to freedom:\n"
        "(1) No person shall be deprived of his or her personal liberty except in accordance with law.\n"
        "(2) Every citizen shall have the following freedoms:\n"
        "(a) freedom of opinion and expression;\n"
        "(b) freedom to assemble peacefully."
    )
    pages = [
        DocumentPage(
            doc_id="constitution",
            title="Constitution of Nepal (2015)",
            page_number=1,
            text=sample_text,
            source_path="data/raw/Constitution-of-Nepal.pdf",
        )
    ]

    chunks = chunk_structure_aware(pages, max_tokens=200, min_section_tokens=10)
    assert len(chunks) >= 2
    # Verify section_ids
    section_ids = [c.section_id for c in chunks]
    assert "constitution:art16" in section_ids or "constitution:art_16" in section_ids
    # Verify header path is prepended
    assert "Constitution of Nepal (2015)" in chunks[0].text
    assert "Article 16" in chunks[0].text or "16" in chunks[0].text


def test_chunk_fixed_window():
    words = "word " * 600
    pages = [
        DocumentPage(
            doc_id="doc1",
            title="Document One",
            page_number=1,
            text=words,
            source_path="fake.pdf",
        )
    ]
    chunks = chunk_fixed_window(pages, chunk_size=300, overlap_tokens=30)
    assert len(chunks) >= 2
    assert chunks[0].chunk_id.startswith("doc1:fixed_300:c")


def test_chunks_jsonl_roundtrip():
    c = Chunk(
        chunk_id="test:s1:c1",
        doc_id="test",
        section_id="test:s1",
        header_path="Test Document > Section 1",
        text="Test Document > Section 1\n\nContent",
        content="Content",
        page_start=1,
        page_end=1,
        token_count=10,
    )
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        save_chunks_jsonl([c], tmp_path)
        loaded = load_chunks_jsonl(tmp_path)
        assert len(loaded) == 1
        assert loaded[0].chunk_id == c.chunk_id
        assert loaded[0].section_id == c.section_id
        assert loaded[0].text == c.text
    finally:
        tmp_path.unlink(missing_ok=True)
