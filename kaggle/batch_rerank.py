"""Kaggle GPU batch cross-encoder scoring script.

Scores all (query, candidate_chunk) pairs using BAAI/bge-reranker-v2-m3 on GPU.
Outputs: rerank_scores.json
"""

import json
import time
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

INPUT_FILE = Path("pairs_to_score.json")
OUTPUT_FILE = Path("rerank_scores.json")
MODEL_NAME = "BAAI/bge-reranker-v2-m3"
BATCH_SIZE = 64


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    if not INPUT_FILE.exists():
        # Check Kaggle input directory or current dir
        candidates = list(Path("/kaggle/input").glob("**/pairs_to_score.json")) + list(Path(".").glob("**/pairs_to_score.json"))
        if candidates:
            pairs_path = candidates[0]
        else:
            raise FileNotFoundError(f"Cannot find {INPUT_FILE} in /kaggle/input or current dir")
    else:
        pairs_path = INPUT_FILE

    print(f"Reading pairs from {pairs_path}...")
    with open(pairs_path, "r", encoding="utf-8") as f:
        pairs_data = json.load(f)

    print(f"Total pairs to score: {len(pairs_data)}")
    pair_tuples = [(p["query"], p["text"]) for p in pairs_data]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CrossEncoder {MODEL_NAME} on {device}...")
    model = CrossEncoder(MODEL_NAME, device=device)

    print(f"Predicting scores in batches of {BATCH_SIZE}...")
    t0 = time.time()
    scores = model.predict(pair_tuples, batch_size=BATCH_SIZE, show_progress_bar=True)
    duration = time.time() - t0
    print(f"Scoring completed in {duration:.2f} seconds ({len(scores)/duration:.1f} pairs/sec)!")

    # Format output dictionary
    cache_dict = {}
    for item, score in zip(pairs_data, scores):
        key = f"{item['query']}::{item['chunk_id']}"
        cache_dict[key] = float(score)

    print(f"Writing {len(cache_dict)} scores to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_dict, f, ensure_ascii=False)

    print(f"Done! {OUTPUT_FILE.stat().st_size} bytes written.")


if __name__ == "__main__":
    main()
