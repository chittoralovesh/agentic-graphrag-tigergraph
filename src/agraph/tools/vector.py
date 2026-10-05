"""Dense + lexical retrieval over the chunked corpus.

Two backends behind one interface:

``LocalVectorIndex``   numpy cosine over cached embeddings - used for
                       development and for runs where the database is not up.
``TigerVectorIndex``   TigerVector ANN inside TigerGraph - the submission path.

Both expose ``search(query, top_k)``, so the pipelines never learn which one
they are talking to.

A BM25 lexical index is included so the RAG baseline can be run as hybrid
retrieval. A weak baseline would inflate the agentic pipeline's apparent win,
which is exactly the conclusion this project is supposed to test rather than
assume.
"""

from __future__ import annotations

import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from ..chunking import Chunk


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float

    def cite(self) -> str:
        return self.doc_id


class VectorIndex(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[Hit]: ...


# --------------------------------------------------------------------------
# Embedding store
# --------------------------------------------------------------------------

class EmbeddingStore:
    """Chunk embeddings on disk, so a re-run never re-pays the quota."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.ids: list[str] = []
        self.matrix: np.ndarray | None = None

    def exists(self) -> bool:
        return self.path.exists()

    def save(self, ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        m = np.asarray(vectors, dtype=np.float32)
        m /= np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)
        np.savez_compressed(self.path, ids=np.array(list(ids), dtype=object), matrix=m)
        self.ids, self.matrix = list(ids), m

    def load(self) -> None:
        data = np.load(self.path, allow_pickle=True)
        self.ids = list(data["ids"])
        self.matrix = data["matrix"].astype(np.float32)


class LocalVectorIndex:
    """Exact cosine search. 16.5k chunks is far too small to need ANN."""

    def __init__(self, store: EmbeddingStore, chunks: dict[str, Chunk], embed_fn):
        if store.matrix is None:
            store.load()
        self.store = store
        self.chunks = chunks
        self.embed_fn = embed_fn
        self._pos = {cid: i for i, cid in enumerate(store.ids)}

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        q = np.asarray(self.embed_fn([query], task="RETRIEVAL_QUERY")[0], dtype=np.float32)
        q /= max(float(np.linalg.norm(q)), 1e-9)
        scores = self.store.matrix @ q
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        out = []
        for i in idx:
            cid = self.store.ids[int(i)]
            c = self.chunks[cid]
            out.append(Hit(cid, c.doc_id, c.title, c.text, float(scores[int(i)])))
        return out


# --------------------------------------------------------------------------
# Lexical (BM25)
# --------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


class BM25Index:
    """Compact BM25 so the baseline can be dense, lexical or hybrid."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.docs = [_tok(c.embed_text) for c in self.chunks]
        self.len = np.array([len(d) for d in self.docs], dtype=np.float32)
        self.avg = float(self.len.mean()) if len(self.len) else 1.0
        self.index: dict[str, list[tuple[int, int]]] = {}
        for i, toks in enumerate(self.docs):
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                self.index.setdefault(t, []).append((i, c))
        self.N = len(self.docs)

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        scores = np.zeros(self.N, dtype=np.float32)
        for t in _tok(query):
            posting = self.index.get(t)
            if not posting:
                continue
            idf = math.log(1 + (self.N - len(posting) + 0.5) / (len(posting) + 0.5))
            for i, tf in posting:
                denom = tf + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                scores[i] += idf * (tf * (self.k1 + 1)) / denom
        k = min(top_k, self.N)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        out = []
        for i in idx:
            if scores[i] <= 0:
                continue
            c = self.chunks[int(i)]
            out.append(Hit(c.chunk_id, c.doc_id, c.title, c.text, float(scores[int(i)])))
        return out


def reciprocal_rank_fusion(runs: Sequence[Sequence[Hit]], k: int = 60, top_k: int = 5) -> list[Hit]:
    """Blend ranked lists without needing comparable score scales."""
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for run in runs:
        for rank, hit in enumerate(run, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            best.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    out = []
    for cid, s in ordered:
        h = best[cid]
        out.append(Hit(h.chunk_id, h.doc_id, h.title, h.text, s))
    return out


class HybridIndex:
    """Dense + BM25 fused with RRF."""

    def __init__(self, dense: VectorIndex, lexical: BM25Index, pool: int = 20):
        self.dense, self.lexical, self.pool = dense, lexical, pool

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        return reciprocal_rank_fusion(
            [self.dense.search(query, self.pool), self.lexical.search(query, self.pool)],
            top_k=top_k,
        )


# --------------------------------------------------------------------------
# TigerVector backend
# --------------------------------------------------------------------------

class TigerVectorIndex:
    """ANN search against a vector attribute on the Chunk vertex."""

    def __init__(self, conn, embed_fn, vertex: str = "Chunk", attr: str = "embedding"):
        self.conn = conn
        self.embed_fn = embed_fn
        self.vertex = vertex
        self.attr = attr

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        vec = self.embed_fn([query], task="RETRIEVAL_QUERY")[0]
        rows = self.conn.runInstalledQuery(
            "chunk_vector_search", {"query_vector": vec, "k": top_k}
        )
        out: list[Hit] = []
        for row in _flatten(rows):
            attrs = row.get("attributes", row)
            out.append(
                Hit(
                    chunk_id=str(row.get("v_id") or attrs.get("chunk_id", "")),
                    doc_id=str(attrs.get("doc_id", "")),
                    title=str(attrs.get("title", "")),
                    text=str(attrs.get("text", "")),
                    score=float(attrs.get("score", 0.0)),
                )
            )
        return out


def _flatten(result) -> list[dict]:
    out: list[dict] = []
    for block in result or []:
        if isinstance(block, dict):
            for value in block.values():
                if isinstance(value, list):
                    out.extend(v for v in value if isinstance(v, dict))
        elif isinstance(block, list):
            out.extend(b for b in block if isinstance(b, dict))
    return out


def load_chunks(path: str | Path) -> dict[str, Chunk]:
    with open(path, "rb") as fh:
        chunks = pickle.load(fh)
    return {c.chunk_id: c for c in chunks}
