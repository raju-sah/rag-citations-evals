"""Kaggle T4 script to batch-embed corpus chunks with BAAI/bge-m3.

Usage on Kaggle:
  python embed_corpus.py --chunks chunks.jsonl --output_dir ./out
"""

import argparse
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


def embed_and_index(
    chunks_path: str,
    output_dir: str,
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 64,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading chunks from {chunks_path}...")
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks.")
    texts = [c["text"] for c in chunks]

    print(f"Loading {model_name} on GPU...")
    model = SentenceTransformer(model_name, device="cuda")

    print(f"Encoding {len(texts)} texts (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    print(f"Building FAISS IndexFlatIP (dim={dim})...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss_file = out / "faiss.index"
    faiss.write_index(index, str(faiss_file))

    metadata_file = out / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"Successfully saved FAISS index to {faiss_file} and metadata to {metadata_file}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="chunks.jsonl")
    parser.add_argument("--output_dir", default="out")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    embed_and_index(args.chunks, args.output_dir, args.model, args.batch_size)
