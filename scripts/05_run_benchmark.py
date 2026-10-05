"""Run the three pipelines over the evaluation set and record every trace.

Usage
    python scripts/05_run_benchmark.py --limit 15
    python scripts/05_run_benchmark.py --pipelines rag,graphrag,agentic
    python scripts/05_run_benchmark.py --split hidden --pipelines agentic

Writes one JSONL of traces per pipeline into artifacts/runs/<split>/, plus a
scored results file. Traces carry per-step tokens, latency, tools and cited
documents, which is what the metrics dashboard and the agentic-behaviour
reporting are built from.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agraph.corpus import iter_questions  # noqa: E402
from agraph.eval.judge import grade  # noqa: E402
from agraph.llm import GeminiClient, load_env  # noqa: E402
from agraph.pipelines.agentic import AgenticGraphRAG  # noqa: E402
from agraph.pipelines.graphrag import GraphRAGPipeline  # noqa: E402
from agraph.pipelines.rag import RAGPipeline  # noqa: E402
from agraph.tools.vector import (  # noqa: E402
    BM25Index,
    EmbeddingStore,
    HybridIndex,
    LocalVectorIndex,
)


def build_index(chunks, llm, mode: str):
    """Pick the retrieval backend: bm25, dense, or hybrid."""
    bm25 = BM25Index(chunks)
    if mode == "bm25":
        return bm25, "bm25"
    store = EmbeddingStore(ROOT / "artifacts" / "embeddings" / "chunks.npz")
    if not store.exists():
        print("  ! embeddings not ready, falling back to BM25")
        return bm25, "bm25"
    by_id = {c.chunk_id: c for c in chunks}
    dense = LocalVectorIndex(store, by_id, llm.embed)
    if mode == "dense":
        return dense, "dense"
    return HybridIndex(dense, bm25), "hybrid"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipelines", default="rag,graphrag,agentic")
    ap.add_argument("--split", default="public", choices=["public", "hidden"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--retrieval", default="bm25", choices=["bm25", "dense", "hybrid"])
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    load_env()
    llm = GeminiClient(use_cache=not args.no_cache)

    with open(ROOT / "artifacts" / "graph.pkl", "rb") as fh:
        graph = pickle.load(fh)
    with open(ROOT / "artifacts" / "chunks.pkl", "rb") as fh:
        chunks = pickle.load(fh)
    index, mode = build_index(chunks, llm, args.retrieval)
    print(f"retrieval backend: {mode}")

    qfile = ROOT / "data" / "questions" / f"eval_{args.split}.jsonl"
    questions = list(iter_questions(qfile))
    if args.limit:
        questions = questions[: args.limit]

    registry = {
        "rag": lambda: RAGPipeline(index, llm, top_k=args.top_k),
        "graphrag": lambda: GraphRAGPipeline(graph, index, llm, top_k=args.top_k),
        "agentic": lambda: AgenticGraphRAG(
            graph, index, llm, max_steps=args.max_steps, top_k=args.top_k
        ),
    }
    wanted = [p.strip() for p in args.pipelines.split(",") if p.strip()]

    outdir = ROOT / "artifacts" / "runs" / args.split
    outdir.mkdir(parents=True, exist_ok=True)
    scored = args.split == "public"

    summary = {}
    for name in wanted:
        if name not in registry:
            print(f"unknown pipeline {name!r}, skipping")
            continue
        pipe = registry[name]()
        print(f"\n=== {name} on {len(questions)} {args.split} questions")
        rows, traces = [], []
        t0 = time.perf_counter()

        for i, q in enumerate(questions, start=1):
            try:
                result = pipe.run(q["qid"], q["question"], q.get("qtype", ""))
            except Exception as exc:  # noqa: BLE001 - one bad question must not end the run
                print(f"  [{i}] {q['qid']} FAILED: {str(exc)[:120]}")
                continue
            tr = result.trace
            traces.append(tr.to_dict())

            row = {
                "qid": q["qid"],
                "qtype": q.get("qtype", ""),
                "pipeline": name,
                "answer": result.text,
                "n_steps": tr.n_steps,
                "tokens": tr.tokens.total,
                "prompt_tokens": tr.tokens.prompt,
                "completion_tokens": tr.tokens.completion,
                "seconds": tr.seconds,
                "n_citations": len(tr.citations),
                "tools_used": tr.tools_used,
                "stop_reason": tr.stop_reason,
                "strategy_changes": tr.strategy_changes,
            }
            if scored:
                refs = [str(a) for a in q.get("answer", [])]
                g = grade(llm, q["question"], result.text, refs)
                row.update(correct=g["correct"], grade_method=g["method"], grade_why=g["why"])
                gold = set(q.get("gold_doc_ids") or [])
                if gold:
                    got = set(tr.citations)
                    row["doc_recall"] = round(len(gold & got) / len(gold), 4)
                    row["doc_recall_complete"] = bool(gold <= got)
                mark = "ok " if g["correct"] else "MISS"
                print(
                    f"  [{i:>3}] {mark} {q['qid']:<9} {q.get('qtype',''):<12}"
                    f" steps={tr.n_steps} tok={tr.tokens.total:<6} {str(result.text)[:40]!r}"
                )
            else:
                print(f"  [{i:>3}] {q['qid']:<9} steps={tr.n_steps} tok={tr.tokens.total}")
            rows.append(row)

        elapsed = time.perf_counter() - t0
        (outdir / f"traces_{name}.jsonl").write_text(
            "\n".join(json.dumps(t, ensure_ascii=False) for t in traces) + "\n",
            encoding="utf-8",
        )
        (outdir / f"results_{name}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )

        tok = sum(r["tokens"] for r in rows)
        line = {
            "pipeline": name,
            "n": len(rows),
            "total_tokens": tok,
            "mean_tokens": round(tok / max(len(rows), 1), 1),
            "mean_steps": round(sum(r["n_steps"] for r in rows) / max(len(rows), 1), 2),
            "seconds": round(elapsed, 1),
        }
        if scored:
            ncorrect = sum(1 for r in rows if r.get("correct"))
            line["accuracy"] = round(ncorrect / max(len(rows), 1), 4)
            line["correct"] = ncorrect
        summary[name] = line
        print(f"  -> {json.dumps(line)}")

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote traces + results to {outdir}")
    print(
        f"llm: {llm.n_calls} live calls, {llm.n_cache_hits} cache hits, "
        f"{llm.session.total} tokens this session"
    )


if __name__ == "__main__":
    main()
