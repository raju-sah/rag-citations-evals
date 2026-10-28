"""Structure-aware and fixed-window document chunking.

Produces chunks with legal hierarchy metadata (doc_id, section_id, header_path, page numbers)
and serializes to data/processed/chunks.jsonl.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from ragcite.ingest import DocumentPage

logger = logging.getLogger(__name__)

# Pattern detecting Parts, Chapters, Schedules, Articles, and Sections
_HEADING_RE = re.compile(
    r"(?:^|\n)\s*(?P<heading>"
    r"(?:Part\s*[-–—]?\s*(?:\d+|[IVXLCDM]+)[^\n]*)|"
    r"(?:Chapter\s*[-–—]?\s*(?:\d+|[IVXLCDM]+)[^\n]*)|"
    r"(?:Schedule\s*[-–—]?\s*(?:\d+|[IVXLCDM]+)[^\n]*)|"
    r"(?:(?:Article|Section)\s*(?P<sec_num_word>\d+)[.:\s][^\n]*)|"
    r"(?:(?P<sec_num_start>\d+)\.\s+(?P<sec_title>[A-Z][^\n:]+):)"
    r")",
    re.IGNORECASE,
)

# Pattern detecting sub-clauses: (1), (2), (a), (b)
_SUBCLAUSE_RE = re.compile(r"(?=(?:^|\n)\s*\((?:\d+|[a-zA-Z])\)\s+)")


@dataclass(frozen=True)
class Chunk:
    """A retrievable text chunk with legal breadcrumb metadata."""

    chunk_id: str
    doc_id: str
    section_id: str
    header_path: str
    text: str
    content: str
    page_start: int
    page_end: int
    token_count: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(**data)


def estimate_tokens(text: str) -> int:
    """Fast, tokenizer-agnostic token estimator.

    Accounts for subword splitting in English (~1.3 tokens/word) and Devanagari characters.
    """
    devanagari_chars = sum(1 for ch in text if "\u0900" <= ch <= "\u097f")
    words = text.split()
    latin_words = len(words)
    return max(1, int(latin_words * 1.3) + int(devanagari_chars * 0.5))


def _split_subsections(
    content: str, max_tokens: int, overlap_tokens: int
) -> list[str]:
    """Split a large section at subclause boundaries (1), (2) with overlap."""
    clauses = [c.strip() for c in _SUBCLAUSE_RE.split(content) if c.strip()]
    if not clauses or len(clauses) == 1:
        # Fallback to paragraph or sentence chunking
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if len(paragraphs) <= 1:
            words = content.split()
            step = max(1, max_tokens - overlap_tokens)
            return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), step)]
        clauses = paragraphs

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for clause in clauses:
        clause_toks = estimate_tokens(clause)
        if current_parts and (current_tokens + clause_toks > max_tokens):
            chunk_str = "\n\n".join(current_parts)
            chunks.append(chunk_str)
            # Apply overlap if possible
            if overlap_tokens > 0 and len(current_parts) > 1:
                current_parts = [current_parts[-1], clause]
                current_tokens = estimate_tokens(current_parts[0]) + clause_toks
            else:
                current_parts = [clause]
                current_tokens = clause_toks
        else:
            current_parts.append(clause)
            current_tokens += clause_toks

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def _extract_section_units(
    pages: Sequence[DocumentPage],
) -> Iterator[dict]:
    """Parse pages into discrete section units with hierarchical breadcrumbs."""
    if not pages:
        return

    doc_id = pages[0].doc_id
    doc_title = pages[0].title

    current_part = ""
    current_chapter = ""
    current_section = "preliminary"
    current_section_title = "Preamble"
    current_buffer: list[str] = []
    page_start = pages[0].page_number
    page_end = page_start

    for page in pages:
        page_num = page.page_number
        lines = page.text.split("\n")

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            match = _HEADING_RE.match(stripped)
            if match:
                heading_raw = match.group("heading").strip()
                heading_lower = heading_raw.lower()

                if heading_lower.startswith("part"):
                    current_part = heading_raw.rstrip(":- ")
                    continue
                elif heading_lower.startswith(("chapter", "schedule")):
                    current_chapter = heading_raw.rstrip(":- ")
                    continue

                # Section / Article boundary
                sec_num = match.group("sec_num_word") or match.group("sec_num_start") or ""
                sec_title = match.group("sec_title") or heading_raw

                # Flush current section
                if current_buffer:
                    content_str = "\n".join(current_buffer).strip()
                    if content_str:
                        yield {
                            "doc_id": doc_id,
                            "doc_title": doc_title,
                            "part": current_part,
                            "chapter": current_chapter,
                            "section_num": current_section,
                            "section_title": current_section_title,
                            "content": content_str,
                            "page_start": page_start,
                            "page_end": page_end,
                        }
                    current_buffer = []

                current_section = sec_num if sec_num else "sec"
                current_section_title = sec_title.rstrip(":- ")
                page_start = page_num

            current_buffer.append(stripped)
            page_end = page_num

    if current_buffer:
        content_str = "\n".join(current_buffer).strip()
        if content_str:
            yield {
                "doc_id": doc_id,
                "doc_title": doc_title,
                "part": current_part,
                "chapter": current_chapter,
                "section_num": current_section,
                "section_title": current_section_title,
                "content": content_str,
                "page_start": page_start,
                "page_end": page_end,
            }


def chunk_structure_aware(
    pages: Sequence[DocumentPage],
    max_tokens: int = 500,
    overlap_tokens: int = 60,
    min_section_tokens: int = 80,
) -> list[Chunk]:
    """Structure-aware chunker: section-level chunks with prepended legal header paths."""
    # Group pages by doc_id
    doc_pages: dict[str, list[DocumentPage]] = {}
    for p in pages:
        doc_pages.setdefault(p.doc_id, []).append(p)

    all_chunks: list[Chunk] = []

    for doc_id, p_list in doc_pages.items():
        units = list(_extract_section_units(p_list))
        # Merge tiny definitions/subsections (< min_section_tokens) into neighbors
        merged_units: list[dict] = []
        for u in units:
            toks = estimate_tokens(u["content"])
            if (
                merged_units
                and toks < min_section_tokens
                and (merged_units[-1]["chapter"] == u["chapter"])
            ):
                prev = merged_units[-1]
                prev["content"] += f"\n\n{u['content']}"
                prev["page_end"] = max(prev["page_end"], u["page_end"])
                if u["section_num"] != "sec":
                    prev["section_title"] += f" & {u['section_title']}"
            else:
                merged_units.append(u)

        for unit in merged_units:
            doc_title = unit["doc_title"]
            part = unit["part"]
            chapter = unit["chapter"]
            sec_num = unit["section_num"]
            sec_title = unit["section_title"]
            content = unit["content"]
            p_start = unit["page_start"]
            p_end = unit["page_end"]

            # Construct section_id e.g. "constitution:art_17" or "labour_act_2017:s12"
            prefix = "art" if "constitution" in doc_id else "s"
            sec_identifier = f"{prefix}{sec_num}" if sec_num != "sec" else "preamble"
            section_id = f"{doc_id}:{sec_identifier}"

            # Construct hierarchical header path
            crumbs = [doc_title]
            if part:
                crumbs.append(part)
            if chapter:
                crumbs.append(chapter)
            crumbs.append(f"{'Article' if 'constitution' in doc_id else 'Section'} {sec_num}: {sec_title}" if sec_num != "sec" else sec_title)
            header_path = " › ".join(crumbs)

            content_tokens = estimate_tokens(content)

            if content_tokens <= max_tokens:
                chunk_id = f"{section_id}:c1"
                chunk_text = f"{header_path}\n\n{content}"
                all_chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        section_id=section_id,
                        header_path=header_path,
                        text=chunk_text,
                        content=content,
                        page_start=p_start,
                        page_end=p_end,
                        token_count=estimate_tokens(chunk_text),
                    )
                )
            else:
                # Split large section at subclauses
                subparts = _split_subsections(content, max_tokens, overlap_tokens)
                for c_idx, sub in enumerate(subparts, start=1):
                    chunk_id = f"{section_id}:c{c_idx}"
                    chunk_text = f"{header_path} (Part {c_idx})\n\n{sub}"
                    all_chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            section_id=section_id,
                            header_path=header_path,
                            text=chunk_text,
                            content=sub,
                            page_start=p_start,
                            page_end=p_end,
                            token_count=estimate_tokens(chunk_text),
                        )
                    )

    return all_chunks


def chunk_fixed_window(
    pages: Sequence[DocumentPage],
    chunk_size: int = 500,
    overlap_tokens: int = 60,
) -> list[Chunk]:
    """Naive fixed token window chunker (used for ablation experiments)."""
    doc_pages: dict[str, list[DocumentPage]] = {}
    for p in pages:
        doc_pages.setdefault(p.doc_id, []).append(p)

    all_chunks: list[Chunk] = []

    for doc_id, p_list in doc_pages.items():
        doc_title = p_list[0].title
        # Flatten words with page tracking
        word_records: list[tuple[str, int]] = []
        for p in p_list:
            for w in p.text.split():
                word_records.append((w, p.page_number))

        step = max(1, chunk_size - overlap_tokens)
        chunk_idx = 1
        for start_idx in range(0, len(word_records), step):
            slice_records = word_records[start_idx : start_idx + chunk_size]
            if not slice_records:
                continue

            text = " ".join(w for w, _ in slice_records)
            p_start = slice_records[0][1]
            p_end = slice_records[-1][1]
            chunk_id = f"{doc_id}:fixed_{chunk_size}:c{chunk_idx}"

            all_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    section_id=f"{doc_id}:fixed",
                    header_path=doc_title,
                    text=text,
                    content=text,
                    page_start=p_start,
                    page_end=p_end,
                    token_count=estimate_tokens(text),
                )
            )
            chunk_idx += 1

    return all_chunks


def chunk_documents(
    pages: Sequence[DocumentPage],
    strategy: str = "structure_aware",
    max_tokens: int = 500,
    overlap_tokens: int = 60,
    min_section_tokens: int = 80,
) -> list[Chunk]:
    """Route to chosen chunking strategy."""
    if strategy == "structure_aware":
        return chunk_structure_aware(
            pages,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            min_section_tokens=min_section_tokens,
        )
    elif strategy == "fixed":
        return chunk_fixed_window(
            pages, chunk_size=max_tokens, overlap_tokens=overlap_tokens
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def save_chunks_jsonl(chunks: Sequence[Chunk], output_path: str | Path) -> None:
    """Save chunks to JSON Lines format."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n" for chunk in chunks)
    logger.info("Saved %d chunks to %s", len(chunks), out)


def load_chunks_jsonl(input_path: str | Path) -> list[Chunk]:
    """Load chunks from JSON Lines format."""
    inp = Path(input_path)
    if not inp.exists():
        raise FileNotFoundError(f"Chunks file not found: {inp}")

    chunks: list[Chunk] = []
    with open(inp, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks


def build_chunks(
    corpus_dir: str | Path = "data/raw",
    output_path: str | Path = "data/processed/chunks.jsonl",
    strategy: str = "structure_aware",
    max_tokens: int = 500,
    overlap_tokens: int = 60,
    min_section_tokens: int = 80,
) -> list[Chunk]:
    """Ingest raw documents and produce serialized chunks."""
    from ragcite.ingest import load_documents

    logger.info("Loading documents from %s ...", corpus_dir)
    pages = load_documents(corpus_dir)
    logger.info("Loaded %d pages across all documents.", len(pages))

    logger.info("Chunking using strategy '%s'...", strategy)
    chunks = chunk_documents(
        pages,
        strategy=strategy,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        min_section_tokens=min_section_tokens,
    )
    save_chunks_jsonl(chunks, output_path)
    return chunks


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Chunk raw legal documents")
    parser.add_argument("--corpus-dir", default="data/raw", help="Path to raw PDFs")
    parser.add_argument("--output", default="data/processed/chunks.jsonl", help="Output JSONL path")
    parser.add_argument(
        "--strategy",
        default="structure_aware",
        choices=["structure_aware", "fixed"],
        help="Chunking strategy",
    )
    parser.add_argument("--max-tokens", type=int, default=500, help="Max tokens per chunk")
    parser.add_argument("--overlap-tokens", type=int, default=60, help="Overlap tokens")
    args = parser.parse_args()

    build_chunks(
        corpus_dir=args.corpus_dir,
        output_path=args.output,
        strategy=args.strategy,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )


