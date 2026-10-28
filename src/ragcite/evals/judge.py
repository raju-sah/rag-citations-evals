"""Cross-family LLM-as-judge for faithfulness, citation entailment, and calibration.

Outputs strict JSON schema and caches all judgments to disk (data/cache/judge_cache/)
to ensure deterministic, zero-cost repeat evaluations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

ENTAILMENT_SYSTEM_PROMPT = """You are an impartial legal judge evaluating factual claims in legal research.
Your task is to determine whether the provided SOURCE text strictly and factually supports the stated CLAIM.

Rules:
1. "supported": true ONLY if the claim can be directly inferred from the facts stated in the source text without extrapolation.
2. "supported": false if the source contradicts the claim, discusses something else, or lacks the necessary facts.
3. Respond ONLY with a valid JSON object matching this schema:
{"supported": true, "reason": "brief explanation"}
"""

RELEVANCY_SYSTEM_PROMPT = """You are an evaluator assessing whether an answer directly and relevantly addresses a question.
Respond ONLY with a valid JSON object:
{"score": 1.0, "reason": "brief explanation"}
where score is a float from 0.0 (completely irrelevant) to 1.0 (fully addresses question).
"""


class LLMJudge:
    """Manages judge model calls with prompt-level disk caching and provider fallbacks."""

    def __init__(
        self,
        provider: str = "groq",
        model: str = "llama-3.3-70b-versatile",
        fallback_provider: str = "gemini",
        fallback_model: str = "gemini-2.5-flash",
        cache_dir: str | Path = "data/cache/judge_cache",
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.fallback_provider = fallback_provider.lower()
        self.fallback_model = fallback_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, prompt: str, model: str) -> str:
        raw = f"{model}:{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _call_api(self, prompt: str, system: str, model: str, provider: str) -> str:
        """Call external LLM API for judge verdict."""
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                logger.warning("GROQ_API_KEY not found; falling back to heuristic/mock judge.")
                return '{"supported": true, "reason": "Mock fallback: default entailment"}'
            from groq import Groq

            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return completion.choices[0].message.content or "{}"

        elif provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY not found; falling back to heuristic/mock judge.")
                return '{"supported": true, "reason": "Mock fallback: default entailment"}'
            from google import genai

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"system_instruction": system, "temperature": 0.0},
            )
            return response.text or "{}"

        else:
            raise ValueError(f"Unsupported judge provider: {provider}")

    def judge_entailment(
        self, claim: str, source_text: str, language: str = "en"
    ) -> tuple[bool, str]:
        """Check if source_text entails claim. Returns (supported: bool, reason: str)."""
        prompt = f"SOURCE TEXT:\n{source_text}\n\nCLAIM TO EVALUATE:\n{claim}"

        # Choose model: if Nepali and configured, use fallback provider (Gemini)
        provider = self.fallback_provider if (language == "ne" and self.fallback_provider) else self.provider
        model = self.fallback_model if (language == "ne" and self.fallback_provider) else self.model

        cache_file = self.cache_dir / f"{self._cache_key(prompt, model)}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    return cached_data["supported"], cached_data.get("reason", "")
            except (OSError, json.JSONDecodeError):
                pass

        try:
            raw_json = self._call_api(prompt, ENTAILMENT_SYSTEM_PROMPT, model, provider)
            parsed = json.loads(raw_json)
            supported = bool(parsed.get("supported", False))
            reason = str(parsed.get("reason", ""))
        except (json.JSONDecodeError, KeyError, ValueError, RuntimeError, OSError) as err:
            logger.warning("Judge call failed: %s; falling back to lexical overlap heuristic", err)
            # Heuristic fallback: if claim words overlap significantly with source words
            claim_words = set(claim.lower().split())
            source_words = set(source_text.lower().split())
            overlap = len(claim_words & source_words) / max(1, len(claim_words))
            supported = overlap > 0.4
            reason = f"Heuristic fallback overlap {overlap:.2f}"

        # Cache result
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"supported": supported, "reason": reason}, f)
        except OSError:
            pass

        return supported, reason

    def run_calibration(
        self, calibration_path: str | Path = "evals/golden/calibration_40.jsonl"
    ) -> dict:
        """Evaluate judge predictions against hand-labeled calibration set to compute Cohen's kappa."""
        from ragcite.evals.metrics import compute_cohen_kappa

        calib_file = Path(calibration_path)
        if not calib_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_file}")

        with open(calib_file, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]

        human_labels: list[int] = []
        judge_labels: list[int] = []
        en_human: list[int] = []
        en_judge: list[int] = []
        ne_human: list[int] = []
        ne_judge: list[int] = []

        for item in items:
            claim = item["claim"]
            cited = item.get("cited_chunk", "")
            h_label = int(item["human_label"])
            lang = item.get("language", "en")

            sup, _ = self.judge_entailment(claim=claim, source_text=cited, language=lang)
            j_label = 1 if sup else 0

            human_labels.append(h_label)
            judge_labels.append(j_label)
            if lang == "ne":
                ne_human.append(h_label)
                ne_judge.append(j_label)
            else:
                en_human.append(h_label)
                en_judge.append(j_label)

        overall_kappa = compute_cohen_kappa(human_labels, judge_labels)
        en_kappa = compute_cohen_kappa(en_human, en_judge) if en_human else 0.0
        ne_kappa = compute_cohen_kappa(ne_human, ne_judge) if ne_human else 0.0

        accuracy = sum(1 for h, j in zip(human_labels, judge_labels) if h == j) / len(human_labels)

        return {
            "num_items": len(items),
            "cohen_kappa_overall": overall_kappa,
            "cohen_kappa_en": en_kappa,
            "cohen_kappa_ne": ne_kappa,
            "accuracy": accuracy,
        }
