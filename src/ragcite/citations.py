"""Citation parsing and structural validation.

Supports both Latin sentence punctuation (. ! ?) and Devanagari danda (।).
Semantic entailment validation (does chunk [n] actually support the claim?)
lives in ``ragcite.evals``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENT_RE = re.compile(r"(?<!\b\d\.)(?<=[.!?।])\s+(?=[A-Z\"'(\u0900-\u097f])")
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

REFUSAL_PHRASES = [
    "do not contain sufficient information",
    "does not contain sufficient information",
    "not enough information in the provided sources",
    "cannot answer this question",
    "not mentioned in the provided sources",
    "पर्याप्त जानकारी उपलब्ध छैन",
    "उल्लेख गरिएको छैन",
    "जानकारी छैन",
]


@dataclass
class CitedSentence:
    """A sentence parsed from an answer, along with its cited source indices."""

    text: str
    citations: list[int] = field(default_factory=list)

    @property
    def clean_text(self) -> str:
        """Text of sentence with [n] citation tags stripped."""
        return _CITE_RE.sub("", self.text).strip()


def is_refusal(answer: str) -> bool:
    """Check if the answer is a model refusal."""
    lower = answer.lower()
    return any(p in lower or p in answer for p in REFUSAL_PHRASES)


def normalize_citation_brackets(text: str) -> str:
    """Normalize full-width brackets 【1】 and Devanagari numerals [१] into standard [1]."""
    cleaned = text.replace("【", "[").replace("】", "]")
    return re.sub(
        r"\[([0-9०-९,\s]+)\]",
        lambda m: f"[{m.group(1).translate(_DEVANAGARI_DIGITS)}]",
        cleaned,
    )


def split_sentences(answer: str) -> list[str]:
    """Sentence splitter handling both English punctuation (. ! ?) and Devanagari purna viram (।)."""
    normalized = normalize_citation_brackets(answer.strip())
    if not normalized:
        return []
    # Attach trailing citations placed after sentence-ending punctuation (e.g. "text. [1]" or "पाठ। [1]")
    normalized = re.sub(
        r"([.!?।])\s*((?:\[\d+(?:\s*,\s*\d+)*\]\s*)+)",
        r" \2\1 ",
        normalized,
    )
    parts = _SENT_RE.split(normalized)
    out: list[str] = []
    for s in parts:
        cleaned = s.strip()
        if not cleaned:
            continue
        # Ignore bare list markers like "1." or "(a)"
        stripped_claim = _CITE_RE.sub("", cleaned).strip(" .!?।-•*()0123456789")
        if len(stripped_claim) < 3:
            continue
        out.append(cleaned)
    return out


def parse_citations(answer: str) -> list[CitedSentence]:
    """Split an answer into sentences and extract ``[n]`` / ``[n, m]`` markers per sentence."""
    out: list[CitedSentence] = []
    for sent in split_sentences(answer):
        nums: list[int] = []
        for group in _CITE_RE.findall(sent):
            nums.extend(int(x.strip()) for x in group.split(",") if x.strip())
        out.append(CitedSentence(text=sent, citations=nums))
    return out


def validate_citations(answer: str, num_sources: int) -> dict:
    """Structural checks: every cited index exists, and every non-refusal sentence has a citation."""
    sents = parse_citations(answer)
    all_cites = [c for s in sents for c in s.citations]
    invalid = sorted({c for c in all_cites if c < 1 or c > num_sources})
    uncited = [s.text for s in sents if not s.citations]

    refused = is_refusal(answer)
    if refused:
        # Refusal answers validly do not require citations
        structural_recall = 1.0
        is_valid = not invalid
    else:
        structural_recall = (
            (len(sents) - len(uncited)) / len(sents) if sents else 0.0
        )
        is_valid = not invalid and not uncited

    structural_precision = (
        (len(all_cites) - len(invalid)) / len(all_cites) if all_cites else 1.0
    )

    return {
        "num_sentences": len(sents),
        "num_citations": len(all_cites),
        "invalid_citations": invalid,
        "uncited_sentences": uncited,
        "structural_citation_recall": structural_recall,
        "structural_citation_precision": structural_precision,
        "is_refusal": refused,
        "valid": is_valid,
    }

