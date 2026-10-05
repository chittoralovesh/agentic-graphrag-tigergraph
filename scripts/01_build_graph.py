"""Build the graph tables from the corpus and report coverage.

Run:  python scripts/01_build_graph.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agraph.corpus import load_corpus  # noqa: E402
from agraph.graphmodel import build_graph  # noqa: E402

CORPUS = ROOT / "data" / "corpus" / "corpus.jsonl"
OUT = ROOT / "artifacts"


def main() -> None:
    docs = load_corpus(CORPUS)
    print(f"loaded {len(docs)} documents")
    g = build_graph(docs)
    print(json.dumps(g.stats(), indent=2))

    # Coverage of the fields the benchmark depends on.
    ev = list(g.events.values())
    def pct(n: int) -> str:
        return f"{n}/{len(ev)} ({100*n/len(ev):.1f}%)"

    print("\nfield coverage")
    print("  competitors :", pct(sum(1 for e in ev if e.competitors is not None)))
    print("  nations     :", pct(sum(1 for e in ev if e.nations is not None)))
    print("  venue       :", pct(sum(1 for e in ev if e.venue)))
    print("  date parsed :", pct(sum(1 for e in ev if e.date and e.date.month)))
    print("  gold        :", pct(sum(1 for e in ev if e.gold_raw)))
    print("  prev edge   :", pct(sum(1 for e in ev if e.prev_qid)))
    print("  next edge   :", pct(sum(1 for e in ev if e.next_qid)))
    unresolved = sum(1 for e in ev if e.prev_year and not e.prev_qid)
    print(f"  prev hints that did not resolve: {unresolved}")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "graph.pkl", "wb") as fh:
        pickle.dump(g, fh)
    print(f"\nwrote {OUT / 'graph.pkl'}")


if __name__ == "__main__":
    main()
