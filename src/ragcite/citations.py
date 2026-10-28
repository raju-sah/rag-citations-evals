"""Citation parsing and structural validation.

Semantic validation (does chunk [n] actually support the sentence?) lives in
``ragcite.evals`` and uses an NLI model or an LLM judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


@dataclass
class CitedSentence:
    text: str
    citations: list[int] = field(default_factory=list)


def split_sentences(answer: str) -> list[str]:
    """Naive sentence splitter; good enough for citation checks."""
    return [s.strip() for s in _SENT_RE.split(answer.strip()) if s.strip()]


def parse_citations(answer: str) -> list[CitedSentence]:
    """Split an answer into sentences and extract ``[n]`` / ``[n, m]`` markers per sentence."""
    out: list[CitedSentence] = []
    for sent in split_sentences(answer):
        nums: list[int] = []
        for group in _CITE_RE.findall(sent):
            nums.extend(int(x) for x in group.split(","))
        out.append(CitedSentence(text=sent, citations=nums))
    return out


def validate_citations(answer: str, num_sources: int) -> dict:
    """Structural checks: every cited index exists, and every sentence has a citation."""
    sents = parse_citations(answer)
    all_cites = [c for s in sents for c in s.citations]
    invalid = sorted({c for c in all_cites if c < 1 or c > num_sources})
    uncited = [s.text for s in sents if not s.citations]
    return {
        "num_sentences": len(sents),
        "num_citations": len(all_cites),
        "invalid_citations": invalid,
        "uncited_sentences": uncited,
        "structural_citation_recall": (len(sents) - len(uncited)) / len(sents) if sents else 0.0,
        "valid": not invalid and not uncited,
    }
