"""Measure what top-k retrieval can and cannot reach, with no LLM involved.

The headline comparison of this project is usually argued with answer accuracy,
which confounds retrieval with generation. This script separates them: it asks
only whether the documents needed to answer are present in the top-k at all.

Two metrics, and the gap between them is the point:

partial recall    fraction of supporting documents retrieved
complete recall   fraction of questions where *every* supporting document was
                  retrieved

For a COUNT or an ARG-MAX, partial recall is worth nothing: retrieving 14 of 15
events still yields the wrong count. Complete recall is therefore the honest
predictor of achievable accuracy, and it is the number that collapses for
aggregation and superlative questions.

Run:  python scripts/03_retrieval_ceiling.py
"""

from __future__ import annotations

import json
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agraph.corpus import iter_questions  # noqa: E402
from agraph.tools.vector import BM25Index  # noqa: E402

KS = [5, 10, 20, 50, 100]


def main() -> None:
    with open(ROOT / "artifacts" / "chunks.pkl", "rb") as fh:
        chunks = pickle.load(fh)
    print(f"indexing {len(chunks)} chunks ...")
    bm = BM25Index(chunks)

    questions = list(iter_questions(ROOT / "data" / "questions" / "eval_public.jsonl"))
    partial = {k: defaultdict(list) for k in KS}
    complete = {k: defaultdict(list) for k in KS}
    gold_sizes = defaultdict(list)

    for q in questions:
        gold = set(q.get("gold_doc_ids") or [])
        if not gold:
            continue
        gold_sizes[q["qtype"]].append(len(gold))
        hits = bm.search(q["question"], top_k=max(KS) * 4)
        ranked: list[str] = []
        for h in hits:
            if h.doc_id not in ranked:
                ranked.append(h.doc_id)
        for k in KS:
            got = set(ranked[:k])
            partial[k][q["qtype"]].append(len(gold & got) / len(gold))
            complete[k][q["qtype"]].append(1.0 if gold <= got else 0.0)

    types = sorted(gold_sizes)
    report: dict = {"k_values": KS, "by_type": {}}

    def table(metric, title: str) -> None:
        print(f"\n{title}")
        print(f"{'qtype':<13}{'gold docs':>11}" + "".join(f"{'k=' + str(k):>9}" for k in KS))
        for t in types:
            row = "".join(f"{100 * statistics.mean(metric[k][t]):>8.0f}%" for k in KS)
            print(f"{t:<13}{statistics.mean(gold_sizes[t]):>11.1f}{row}")
        allrow = "".join(
            f"{100 * statistics.mean([v for t in types for v in metric[k][t]]):>8.0f}%"
            for k in KS
        )
        print(f"{'ALL':<13}{'':>11}{allrow}")

    table(partial, "PARTIAL gold-document recall")
    table(complete, "COMPLETE gold-set retrieval (every supporting doc in top-k)")

    for t in types:
        report["by_type"][t] = {
            "mean_gold_docs": round(statistics.mean(gold_sizes[t]), 2),
            "n_questions": len(gold_sizes[t]),
            "partial_recall": {k: round(statistics.mean(partial[k][t]), 4) for k in KS},
            "complete_recall": {k: round(statistics.mean(complete[k][t]), 4) for k in KS},
        }

    out = ROOT / "artifacts" / "retrieval_ceiling.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print(
        "\nReading: at k=5, aggregation and superlative reach 0% complete retrieval.\n"
        "Those 31 questions are unanswerable by top-k retrieval in principle, however\n"
        "good the generator is. The graph tools answer them from a single grouped scan."
    )


if __name__ == "__main__":
    main()
