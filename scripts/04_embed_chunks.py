"""Embed every chunk once and cache the vectors.

Resumable: partial progress is checkpointed, so a quota pause or a dropped
connection costs only the current batch rather than the whole run.

Run:  python scripts/04_embed_chunks.py [--dim 768] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agraph.llm import GeminiClient, load_env  # noqa: E402

OUT = ROOT / "artifacts" / "embeddings"
PART = OUT / "partial.jsonl"
FINAL = OUT / "chunks.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=768, help="output dimensionality")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="", help="override the embedding model")
    args = ap.parse_args()

    load_env()
    with open(ROOT / "artifacts" / "chunks.pkl", "rb") as fh:
        chunks = pickle.load(fh)
    if args.limit:
        chunks = chunks[: args.limit]
    OUT.mkdir(parents=True, exist_ok=True)

    done: dict[str, list[float]] = {}
    if PART.exists():
        with open(PART, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    done[d["id"]] = d["v"]
        print(f"resuming: {len(done)} chunks already embedded")

    todo = [c for c in chunks if c.chunk_id not in done]
    print(f"embedding {len(todo)} of {len(chunks)} chunks at dim={args.dim}")

    client = GeminiClient()
    if args.model:
        client.embed_model = args.model
    print(f"model: {client.embed_model}")
    t0 = time.perf_counter()
    with open(PART, "a", encoding="utf-8") as fh:
        for i in range(0, len(todo), args.batch):
            batch = todo[i : i + args.batch]
            vecs = _embed(client, [c.embed_text for c in batch], args.dim)
            for c, v in zip(batch, vecs):
                done[c.chunk_id] = v
                fh.write(json.dumps({"id": c.chunk_id, "v": v}) + "\n")
            fh.flush()
            n = i + len(batch)
            rate = n / max(time.perf_counter() - t0, 1e-6)
            eta = (len(todo) - n) / max(rate, 1e-6)
            print(f"  {n}/{len(todo)}  {rate:.1f}/s  eta {eta/60:.1f} min", flush=True)

    ids = [c.chunk_id for c in chunks if c.chunk_id in done]
    mat = np.asarray([done[i] for i in ids], dtype=np.float32)
    mat /= np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9, None)
    np.savez_compressed(FINAL, ids=np.array(ids, dtype=object), matrix=mat)
    print(f"\nwrote {FINAL}  shape={mat.shape}")


def _embed(client: GeminiClient, texts: list[str], dim: int) -> list[list[float]]:
    """Embed one batch, honouring the requested output dimensionality."""
    from google.genai import types

    for attempt in range(6):
        try:
            client._limiter.wait()
            resp = client._client.models.embed_content(
                model=client.embed_model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT", output_dimensionality=dim
                ),
            )
            return [list(e.values) for e in resp.embeddings]
        except Exception as exc:  # noqa: BLE001
            if attempt == 5:
                raise
            wait = min(2**attempt * 3, 90)
            print(f"    retry in {wait}s ({str(exc)[:90]})", flush=True)
            time.sleep(wait)
    return []


if __name__ == "__main__":
    main()
