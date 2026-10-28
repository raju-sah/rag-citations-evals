"""Load raw documents (PDF) into clean page-level records with metadata."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pypdf

logger = logging.getLogger(__name__)

DOC_METADATA: dict[str, tuple[str, str]] = {
    "Constitution-of-Nepal.pdf": ("constitution", "Constitution of Nepal (2015)"),
    "The-Labour-Act-2017.pdf": ("labour_act_2017", "Labour Act, 2017"),
    "National-Civil-Code.pdf": ("civil_code_2017", "Muluki Civil Code, 2017"),
    "National-Penal-code.pdf": ("penal_code_2017", "Muluki Penal Code, 2017"),
    "National-Criminal-Procedure-Code.pdf": (
        "criminal_procedure_code_2017",
        "Muluki Criminal Procedure Code, 2017",
    ),
    "Industrial-Enterprises-Act-2020.pdf": (
        "industrial_enterprises_act_2020",
        "Industrial Enterprises Act, 2020",
    ),
    "Foreign-Investment-and-Technology-Transfer-Act-.pdf": (
        "fitta_2019",
        "Foreign Investment and Technology Transfer Act, 2019",
    ),
    "Environment-Protection-Act.pdf": (
        "environment_protection_act_2019",
        "Environment Protection Act, 2019",
    ),
    "National-Human-Rights-Commission-Act.pdf": (
        "nhrc_act_2012",
        "National Human Rights Commission Act, 2012",
    ),
    "Prevention-of-Corruption_Act.pdf": (
        "prevention_of_corruption_act_2002",
        "Prevention of Corruption Act, 2002",
    ),
}

_RUNNING_HEADER_PATTERNS = [
    re.compile(r"^\s*www\.lawcommission\.gov\.np\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*-\s*\d+\s*-\s*$", re.MULTILINE),
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),
    re.compile(
        r"^\s*Constitution of Nepal 2015, Unofficial English Translation\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
]


@dataclass(frozen=True)
class DocumentPage:
    """A single page extracted from a document."""

    doc_id: str
    title: str
    page_number: int  # 1-indexed
    text: str
    source_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def clean_page_text(raw_text: str) -> str:
    """Strip running boilerplate headers, footers, and solitary page numbers."""
    text = raw_text
    for pattern in _RUNNING_HEADER_PATTERNS:
        text = pattern.sub("", text)
    # Normalize excessive newlines/spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def derive_doc_metadata(file_path: Path) -> tuple[str, str]:
    """Return (doc_id, title) for a given file name."""
    name = file_path.name
    if name in DOC_METADATA:
        return DOC_METADATA[name]
    # Fallback to sanitized stem
    stem = file_path.stem.lower()
    doc_id = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    title = file_path.stem.replace("-", " ").replace("_", " ").title()
    return doc_id, title


def load_pdf(pdf_path: str | Path) -> list[DocumentPage]:
    """Extract page records from a single PDF document."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc_id, title = derive_doc_metadata(path)
    reader = pypdf.PdfReader(str(path))
    pages: list[DocumentPage] = []

    for idx, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except (pypdf.errors.PdfReadError, OSError, ValueError) as err:
            logger.warning("Failed to extract text from %s page %d: %s", path.name, idx, err)
            raw_text = ""

        cleaned = clean_page_text(raw_text)
        if cleaned:
            pages.append(
                DocumentPage(
                    doc_id=doc_id,
                    title=title,
                    page_number=idx,
                    text=cleaned,
                    source_path=str(path),
                )
            )

    return pages


def load_documents(
    corpus_dir: str | Path, pattern: str = "*.pdf"
) -> list[DocumentPage]:
    """Scan directory and load all matching PDFs into DocumentPage objects."""
    dir_path = Path(corpus_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {dir_path}")

    pdf_files = sorted(dir_path.glob(pattern))
    all_pages: list[DocumentPage] = []
    for pdf_file in pdf_files:
        all_pages.extend(load_pdf(pdf_file))

    return all_pages

