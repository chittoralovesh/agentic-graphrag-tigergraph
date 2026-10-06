"""Embed every chunk with a local ONNX model and push the vectors to TigerVector.

Gemini's free embedding tier caps at 1,000 requests/day, which does not cover
16,528 chunks, and gemini-embedding-2 silently returns a single vector per
request regardless of batch size. A local model removes both problems: no
quota, no network dependency, and judges can reproduce the vector lane without
an API key.

BAAI/bge-small-en-v1.5 (384-dim) runs at roughly 700 chunks/second on CPU.

Run:  python scripts/11_embed_local.py [--push] [--model BAAI/bge-small-en-v1.5]
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agraph.llm import load_env  # noqa: E402

OUT = ROOT / "artifacts" / "embeddings"
FINAL = OUT / "chunks.npz"
GRAPH = "OlympicGraphRAG"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--push", action="store_true", help="also upsert vectors into TigerGraph")
    ap.add_argument("--batch", type=int, default=250, help="TigerGraph upsert batch")
    args = ap.parse_args()

    from fastembed import TextEmbedding

    with open(ROOT / "artifacts" / "chunks.pkl", "rb") as fh:
        chunks = pickle.load(fh)
    print(f"embedding {len(chunks)} chunks with {args.model} ...")

    model = TextEmbedding(args.model)
    t0 = time.perf_counter()
    vectors = list(model.embed([c.embed_text for c in chunks]))
    elapsed = time.perf_counter() - t0

    mat = np.asarray(vectors, dtype=np.float32)
    mat /= np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9, None)
    ids = [c.chunk_id for c in chunks]

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FINAL, ids=np.array(ids, dtype=object), matrix=mat)
    print(
        f"  {len(ids)} vectors, dim={mat.shape[1]}, "
        f"{elapsed:.1f}s ({len(ids)/elapsed:.0f}/s)\n  wrote {FINAL}"
    )

    if not args.push:
        print("\n(skipping TigerGraph push; pass --push to load vectors)")
        return

    load_env()
    from agraph.store.tigergraph import connect

    conn = connect(graphname=GRAPH, verify=False)
    conn.graphname = GRAPH
    print(f"\npushing vectors into TigerGraph ({GRAPH}.Chunk.embedding) ...")

    t0 = time.perf_counter()
    sent = 0
    for i in range(0, len(ids), args.batch):
        batch_ids = ids[i : i + args.batch]
        batch_vecs = mat[i : i + args.batch]
        payload = {
            cid: {"embedding": vec.tolist()} for cid, vec in zip(batch_ids, batch_vecs)
        }
        sent += conn.upsertVertices("Chunk", list(payload.items()))
        if (i // args.batch) % 10 == 0:
            print(f"  {i + len(batch_ids)}/{len(ids)}", flush=True)
    print(f"  upserted {sent} chunk vectors in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
