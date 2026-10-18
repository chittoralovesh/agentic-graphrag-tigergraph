"""Measure real end-to-end latency per pipeline, with the LLM cache disabled.

The benchmark caches answers on disk so re-runs cost nothing, which makes the
`seconds` field in a cached run meaningless as a response time (a cache hit
returns in roughly zero). This runs a fresh sample with caching off, so the
reported latency is what a user would actually experience.

Run:  python scripts/12_latency.py [--n 12]
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from agraph.corpus import iter_questions  # noqa: E402
from agraph.llm import GeminiClient, load_env  # noqa: E402
from agraph.pipelines.agentic import AgenticGraphRAG  # noqa: E402
from agraph.pipelines.graphrag import GraphRAGPipeline  # noqa: E402
from agraph.pipelines.rag import RAGPipeline  # noqa: E402
from agraph.tools.vector import BM25Index  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="questions per pipeline")
    args = ap.parse_args()

    load_env()
    llm = GeminiClient(use_cache=False)  # cache off: we are timing real calls
    print(f"model: {llm.model}")

    with open(ROOT / "artifacts" / "graph.pkl", "rb") as fh:
        graph = pickle.load(fh)
    with open(ROOT / "artifacts" / "chunks.pkl", "rb") as fh:
        chunks = pickle.load(fh)
    index = BM25Index(chunks)

    questions = list(iter_questions(ROOT / "data" / "questions" / "eval_public.jsonl"))
    # Spread the sample across question types rather than taking the first n,
    # since step count (and so latency) varies sharply by type.
    by_type: dict[str, list] = {}
    for q in questions:
        by_type.setdefault(q["qtype"], []).append(q)
    sample: list = []
    i = 0
    while len(sample) < args.n:
        added = False
        for t in sorted(by_type):
            if i < len(by_type[t]) and len(sample) < args.n:
                sample.append(by_type[t][i])
                added = True
        if not added:
            break
        i += 1

    pipelines = {
        "rag": RAGPipeline(index, llm, top_k=5),
        "graphrag": GraphRAGPipeline(graph, index, llm, top_k=5),
        "agentic": AgenticGraphRAG(graph, index, llm, max_steps=4, top_k=5),
    }

    report = {}
    for name, pipe in pipelines.items():
        times: list[float] = []
        print(f"\n=== {name} on {len(sample)} questions (cache off)")
        for q in sample:
            t0 = time.perf_counter()
            try:
                pipe.run(q["qid"], q["question"], q.get("qtype", ""))
            except Exception as exc:  # noqa: BLE001
                print(f"  {q['qid']} failed: {str(exc)[:90]}")
                continue
            dt = time.perf_counter() - t0
            times.append(dt)
            print(f"  {q['qid']:<9} {q['qtype']:<12} {dt:6.2f}s", flush=True)
        if times:
            report[name] = {
                "n": len(times),
                "mean_seconds": round(statistics.mean(times), 2),
                "median_seconds": round(statistics.median(times), 2),
                "min_seconds": round(min(times), 2),
                "max_seconds": round(max(times), 2),
            }
            print(f"  -> mean {report[name]['mean_seconds']}s, median {report[name]['median_seconds']}s")

    out = ROOT / "artifacts" / "latency.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{'pipeline':<12}{'mean s':>9}{'median s':>10}")
    for name, r in report.items():
        print(f"{name:<12}{r['mean_seconds']:>9}{r['median_seconds']:>10}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
