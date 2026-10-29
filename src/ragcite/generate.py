"""Prompt the LLM with numbered sources [1]..[n] and require per-sentence citations.

Includes strict citation enforcement, refusal on unsupported questions, and
disk-based caching to protect API rate limits during iterative evaluation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from ragcite.retrieve import RetrievedCandidate

load_dotenv()

logger = logging.getLogger(__name__)

_GEMINI_QUOTA_EXHAUSTED = False
_GROQ_MODELS = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
)
_GROQ_CALL_COUNTER = 0


def _get_secret(key: str) -> str | None:
    """Retrieve an API secret from environment variables or Streamlit secrets."""
    val = os.getenv(key)
    if val and val.strip():
        return val.strip()
    try:
        import streamlit as st

        if hasattr(st, "secrets") and key in st.secrets:
            s_val = st.secrets[key]
            if isinstance(s_val, str) and s_val.strip():
                return s_val.strip()
    except Exception:  # noqa: BLE001, S110
        pass
    return None

_GREETING_PATTERNS = re.compile(
    r"^\s*(?:hi|hello|hey|hi\s+there|hello\s+there|hey\s+there|namaste|namaskar|"
    r"नमस्ते|नमस्कार|good\s+(?:morning|afternoon|evening|day)|"
    r"how\s+are\s+you(?:\s+doing)?|who\s+are\s+you|what\s+can\s+you\s+do|"
    r"help|thanks|thank\s+you|धन्यवाद|के\s+छ|कस्तो\s+छ)[!.?\s]*$",
    re.IGNORECASE,
)


def check_conversational_query(question: str) -> str | None:
    """Return an instant helpful greeting if the input is casual chitchat rather than a legal query."""
    cleaned = question.strip()
    if not cleaned:
        return None
    if _GREETING_PATTERNS.match(cleaned):
        return (
            "Namaste! 👋 I am the **Nepal Legal Assistant**. I answer questions about Nepalese law "
            "with strict sentence-level citations `[n]` backed by statutory sources.\n\n"
            "**You can ask me in English or Nepali about:**\n"
            "- **Constitution of Nepal (2015)** (Fundamental Rights, Duties, State Structure)\n"
            "- **Labour Act, 2017** (Employment contracts, probation, leave, overtime, termination)\n"
            "- **Muluki Civil & Penal Codes, 2017** (Contracts, marriage/divorce, criminal offenses)\n"
            "- **Foreign Investment (FITTA 2019), Industrial Enterprises, Environment Protection, NHRC, and Prevention of Corruption Acts**\n\n"
            "*Try clicking one of the suggested questions below or typing a specific legal question!*"
        )
    return None

SYSTEM_INSTRUCTION = """You are a precise legal assistant answering questions about the legal provisions of Nepal.
You must answer strictly and solely using the facts provided in the numbered sources below.

Strict Citation Rules:
1. Every claim or sentence you write MUST end with one or more citations referencing the source numbers in brackets, e.g. [1] or [1, 2].
2. Do NOT write any uncited sentences.
3. Only cite a source if it directly supports the preceding sentence.
4. If the provided sources do NOT contain enough information to answer the question, do not speculate or extrapolate from external knowledge. You MUST respond with:
"The provided sources do not contain sufficient information to answer this question."
5. If the user's question is in Nepali, answer in Nepali while continuing to cite the numbered sources using standard [n] citations.
"""

RELAXED_SYSTEM_INSTRUCTION = """You are a helpful legal assistant answering questions about the legal provisions of Nepal.
Use the provided numbered sources below to answer the user's question clearly and concisely.
Include citations like [1] when referencing specific provisions, and if the sources do not address the question, state that the provided sources do not contain sufficient information to answer this question.
If the user's question is in Nepali, answer in Nepali.
"""


@dataclass
class GenerationResult:
    """Output from the generator LLM containing the cited answer and metadata."""

    question: str
    answer: str
    sources: list[dict]
    model: str
    provider: str
    cached: bool
    latency_s: float
    prompt: str

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "model": self.model,
            "provider": self.provider,
            "cached": self.cached,
            "latency_s": self.latency_s,
            "prompt": self.prompt,
        }


def format_prompt(question: str, sources: Sequence[RetrievedCandidate]) -> str:
    """Format numbered context sources [1]..[n] for the LLM prompt."""
    source_blocks: list[str] = []
    for idx, s in enumerate(sources, start=1):
        block = (
            f"[{idx}] Source: {s.header_path} (Pages {s.page_start}–{s.page_end})\n"
            f"{s.text.strip()}"
        )
        source_blocks.append(block)

    joined_sources = "\n\n".join(source_blocks)
    return f"SOURCES:\n{joined_sources}\n\nQUESTION: {question}\n\nCITED ANSWER:"


class LLMClient:
    """Unified LLM client supporting Gemini, Groq, and OpenAI with disk caching."""

    def __init__(
        self,
        provider: str = "gemini",
        model: str = "gemini-3.8-flash",
        temperature: float = 0.0,
        cache_dir: str | Path = "data/cache/llm_cache",
        api_key: str | None = None,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key if api_key is not None else self._resolve_api_key()

    def _resolve_api_key(self) -> str | None:
        if self.provider == "gemini":
            return _get_secret("GEMINI_API_KEY") or _get_secret("GOOGLE_API_KEY")
        elif self.provider == "groq":
            return _get_secret("GROQ_API_KEY")
        elif self.provider == "openai":
            return _get_secret("OPENAI_API_KEY")
        return None

    def _cache_key(self, prompt: str, system: str = SYSTEM_INSTRUCTION) -> str:
        if system == SYSTEM_INSTRUCTION:
            raw = f"{self.provider}:{self.model}:{self.temperature}:{prompt}"
        else:
            raw = f"{self.provider}:{self.model}:{self.temperature}:{system}:{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def generate(self, prompt: str, system: str = SYSTEM_INSTRUCTION) -> tuple[str, bool, float]:
        """Generate response with disk caching. Returns (response_text, is_cached, latency_s)."""
        cache_file = self.cache_dir / f"{self._cache_key(prompt, system=system)}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data["text"], True, data.get("latency_s", 0.0)
            except (OSError, json.JSONDecodeError, KeyError) as err:
                logger.warning("Failed to read LLM cache %s: %s", cache_file, err)

        start_time = time.perf_counter()
        response_text = self._call_provider(prompt, system=system)
        duration_s = time.perf_counter() - start_time

        # Save to disk cache
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "provider": self.provider,
                        "model": self.model,
                        "prompt": prompt,
                        "text": response_text,
                        "latency_s": duration_s,
                    },
                    f,
                    ensure_ascii=False,
                )
        except OSError as err:
            logger.warning("Failed to write LLM cache: %s", err)

        return response_text, False, duration_s

    def _call_provider(self, prompt: str, system: str) -> str:
        """Call external LLM API with circuit-breaker and multi-model load balancing."""
        global _GEMINI_QUOTA_EXHAUSTED, _GROQ_CALL_COUNTER

        if not self.api_key:
            logger.warning("No API key configured for provider '%s'. Returning mock response.", self.provider)
            return (
                "Under the Constitution of Nepal, every citizen has the right to freedom of opinion and expression [1]. "
                "The Labour Act regulates employment contracts and workplace safety [2]."
            )

        if self.provider == "gemini":
            last_gemini_err = None
            if not _GEMINI_QUOTA_EXHAUSTED:
                from google import genai
                from google.genai import types

                gemini_key = self.api_key or _get_secret("GEMINI_API_KEY") or _get_secret("GOOGLE_API_KEY")
                if gemini_key:
                    client = genai.Client(api_key=gemini_key)
                    try:
                        response = client.models.generate_content(
                            model=self.model,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=system,
                                temperature=self.temperature,
                                thinking_config=types.ThinkingConfig(thinking_budget=0),
                            ),
                        )
                        return response.text or ""
                    except Exception as err:  # noqa: BLE001
                        last_gemini_err = str(err)
                        if "429" in last_gemini_err or "RESOURCE_EXHAUSTED" in last_gemini_err or "quota" in last_gemini_err.lower() or "503" in last_gemini_err:
                            logger.warning("Gemini quota or 503 unavailable; circuit-breaking to Groq for remaining calls.")
                            _GEMINI_QUOTA_EXHAUSTED = True
                        else:
                            logger.warning("Gemini model %s failed: %s; falling back to Groq", self.model, err)

            # Fast cross-provider fallback to Groq with multi-model rotation to avoid 30 RPM limits
            groq_key = _get_secret("GROQ_API_KEY")
            last_groq_err = None
            if groq_key:
                from groq import Groq

                g_client = Groq(api_key=groq_key)
                start_idx = _GROQ_CALL_COUNTER % len(_GROQ_MODELS)
                _GROQ_CALL_COUNTER += 1
                ordered_models = list(_GROQ_MODELS[start_idx:]) + list(_GROQ_MODELS[:start_idx])
                for g_model in ordered_models:
                    try:
                        completion = g_client.chat.completions.create(
                            model=g_model,
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=self.temperature,
                        )
                        return completion.choices[0].message.content or ""
                    except Exception as groq_err:  # noqa: BLE001
                        last_groq_err = str(groq_err)
                        logger.warning("Groq model %s failed: %s; trying next model", g_model, groq_err)

            raise RuntimeError(
                f"Both Gemini and Groq providers failed or exhausted rate limits. "
                f"Gemini: {last_gemini_err or 'No key or quota exhausted'}. "
                f"Groq: {last_groq_err or ('No GROQ_API_KEY found' if not groq_key else 'All models failed')}."
            )

        elif self.provider == "groq":
            from groq import Groq

            client = Groq(api_key=self.api_key)
            models_to_try = [self.model]
            for candidate in ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"):
                if candidate not in models_to_try:
                    models_to_try.append(candidate)

            last_err = None
            for model_name in models_to_try:
                try:
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=self.temperature,
                    )
                    return completion.choices[0].message.content or ""
                except Exception as err:  # noqa: BLE001
                    logger.warning("Groq model %s failed: %s; trying next fallback", model_name, err)
                    last_err = err
            logger.error("All Groq model attempts failed: %s", last_err)
            raise last_err

        elif self.provider == "openai":
            try:
                from openai import OpenAI

                client = OpenAI(api_key=self.api_key)
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                )
                return completion.choices[0].message.content or ""
            except Exception as err:
                logger.error("OpenAI API call failed: %s", err)
                raise
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")


def answer_with_citations(
    question: str,
    sources: Sequence[RetrievedCandidate],
    provider: str = "gemini",
    model: str = "gemini-3.8-flash",
    temperature: float = 0.0,
    cache_dir: str | Path = "data/cache/llm_cache",
    api_key: str | None = None,
    system_prompt: str = SYSTEM_INSTRUCTION,
) -> GenerationResult:
    """Format prompt, query LLM, and return result with cited provenance."""
    prompt = format_prompt(question, sources)
    client = LLMClient(
        provider=provider,
        model=model,
        temperature=temperature,
        cache_dir=cache_dir,
        api_key=api_key,
    )
    from ragcite.citations import normalize_citation_brackets

    answer_text, cached, latency_s = client.generate(prompt, system=system_prompt)
    answer_text = normalize_citation_brackets(answer_text)

    return GenerationResult(
        question=question,
        answer=answer_text,
        sources=[s.to_dict() for s in sources],
        model=model,
        provider=provider,
        cached=cached,
        latency_s=latency_s,
        prompt=prompt,
    )

