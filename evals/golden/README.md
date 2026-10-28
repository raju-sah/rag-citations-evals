# Golden Evaluation Dataset

This directory contains the curated golden set and human calibration items for evaluating retrieval, citation faithfulness, and answer quality.

## Files

- `golden.jsonl`: 200 questions with reference answers, gold section IDs, language, and category labels.
- `calibration_40.jsonl`: 40 hand-verified `(claim, cited_chunk)` pairs for measuring LLM-as-judge agreement (Cohen's $\kappa$).

## Dataset Stratification

| Dimension | Distribution | Description |
|---|---|---|
| **Split** | 147 dev / 53 held-out test | Development split used for hyperparameter tuning; test split strictly held out. |
| **Language** | 134 English (67%) / 66 Nepali (33%) | Evaluates native English retrieval and cross-lingual retrieval via `bge-m3`. |
| **Answerability** | 170 answerable (85%) / 30 unanswerable (15%) | Tests refusal rate and resistance to hallucination on out-of-scope legal queries. |
| **Category** | 148 single-section, 22 multi-section, 30 unanswerable | Mix of direct statutory reference lookups, paraphrased inquiries, and synthesis questions. |

## Gold Section ID Ground Truth

Ground-truth targets are indexed at the **Section ID** level (e.g. `constitution:art17`, `labour_act_2017:s10`, `civil_code_2017:s111`) rather than transient chunk IDs. A retrieved chunk counts as a hit if its `section_id` intersects the target `gold_section_ids`, ensuring ground truth remains 100% valid across all chunk-size ablations (300 vs 500 vs 800 tokens).

## 40-Item Human Calibration Set

To validate the judge LLM (avoiding self-preference bias), 40 individual claim-to-chunk pairs (20 English, 20 Nepali) were manually annotated with binary entailment labels (`1` = supported, `0` = unsupported / hallucinated / fact contradiction). The eval harness computes Cohen's $\kappa$ on this set to confirm judge reliability prior to running full evaluation sweeps.

