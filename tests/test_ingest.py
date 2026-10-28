from pathlib import Path

from ragcite.ingest import clean_page_text, derive_doc_metadata, load_pdf


def test_derive_doc_metadata():
    doc_id, title = derive_doc_metadata(Path("Constitution-of-Nepal.pdf"))
    assert doc_id == "constitution"
    assert "Constitution of Nepal" in title

    doc_id2, title2 = derive_doc_metadata(Path("Some_Unknown_Act.pdf"))
    assert doc_id2 == "some_unknown_act"
    assert title2 == "Some Unknown Act"


def test_clean_page_text():
    raw = (
        "www.lawcommission.gov.np\n"
        "12\n"
        "Section 10. Employment contract.\n"
        "An employer shall conclude an employment contract.\n"
        "- 12 -\n"
    )
    cleaned = clean_page_text(raw)
    assert "www.lawcommission.gov.np" not in cleaned
    assert "Section 10. Employment contract." in cleaned
    assert "An employer shall conclude an employment contract." in cleaned


def test_load_pdf_on_sample_corpus():
    sample_pdf = Path("data/raw/Prevention-of-Corruption_Act.pdf")
    if sample_pdf.exists():
        pages = load_pdf(sample_pdf)
        assert len(pages) > 0
        assert pages[0].doc_id == "prevention_of_corruption_act_2002"
        assert len(pages[0].text) > 20
