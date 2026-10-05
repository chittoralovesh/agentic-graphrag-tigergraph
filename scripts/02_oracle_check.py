"""Upper-bound check: can the graph tools answer the 100 public questions?

This is not a pipeline. It routes each question to the right tool using the
known question templates, which isolates *retrieval capability* from *LLM
reasoning*. If a type scores poorly here, the graph layer is at fault; if it
scores well here but poorly in the pipeline, the agent is at fault.

Run:  python scripts/02_oracle_check.py
"""

from __future__ import annotations

import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agraph.corpus import iter_questions, norm  # noqa: E402
from agraph.tools import structured as st  # noqa: E402

PUB = ROOT / "data" / "questions" / "eval_public.jsonl"


def answer(g, q: str, qtype: str):
    """Template-route one question to the structured tools."""
    if qtype == "lookup":
        m = re.search(r"How many nations competed in (.+?)\?$", q)
        if not m:
            return None, []
        ev = st.link_event_by_title(g, m.group(1))
        if not ev.value:
            return None, []
        got = st.get_event_attribute(g, ev.value, "nations")
        return (str(got.value) if got.value is not None else None), got.doc_ids

    if qtype == "aggregation":
        m = re.search(
            r"how many (.+?) events at the (.+?) Olympics had more than (\d+) (competitors|nations)",
            q, re.I,
        )
        if not m:
            return None, []
        ev = st.count_events_where(g, m.group(1), m.group(2), m.group(4), ">", int(m.group(3)))
        return (str(ev.value) if ev.value is not None else None), ev.doc_ids

    if qtype == "superlative":
        m = re.search(
            r"which (.+?) event at the (.+?) Olympics had the (highest|lowest) number of (competitors|nations)",
            q, re.I,
        )
        if not m:
            return None, []
        ev = st.argmax_event(g, m.group(1), m.group(2), m.group(4))
        return ev.value, ev.doc_ids

    if qtype == "multi_hop":
        m = re.search(r"held at (.+?) on (.+?)(?: at the (.+?) Olympics)?\?$", q)
        if not m:
            return None, []
        ev = st.find_event_by_venue_date(g, m.group(1), m.group(2), m.group(3))
        if not ev.value:
            return None, ev.doc_ids
        got = st.get_event_attribute(g, ev.value, "gold_raw")
        return got.value, list(dict.fromkeys(ev.doc_ids + got.doc_ids))

    if qtype == "temporal":
        m = re.search(
            r"gold medal in the (.+?) event at the (Summer|Winter) Olympics held immediately before (\d{4})",
            q,
        )
        if not m:
            return None, []
        phrase, season, year = m.group(1), m.group(2), int(m.group(3))
        sport = None
        for s in sorted(g.sports, key=len, reverse=True):
            if norm(s) in norm(phrase):
                sport = s
                break
        discipline = phrase
        if sport:
            discipline = re.sub(re.escape(sport), "", phrase, flags=re.I).strip()
        ev = st.edition_before_year(g, sport, discipline, year, season)
        if not ev.value:
            return None, []
        got = st.get_event_attribute(g, ev.value, "gold_raw")
        return got.value, list(dict.fromkeys(ev.doc_ids + got.doc_ids))

    return None, []


def main() -> None:
    with open(ROOT / "artifacts" / "graph.pkl", "rb") as fh:
        g = pickle.load(fh)

    by_type: dict[str, list[bool]] = defaultdict(list)
    recall: dict[str, list[float]] = defaultdict(list)
    misses = []

    for q in iter_questions(PUB):
        got, docs = answer(g, q["question"], q["qtype"])
        gold = [str(a) for a in q["answer"]]
        ok = got is not None and any(norm(got) == norm(a) for a in gold)
        by_type[q["qtype"]].append(ok)

        gold_docs = set(q.get("gold_doc_ids") or [])
        if gold_docs:
            recall[q["qtype"]].append(len(gold_docs & set(docs)) / len(gold_docs))
        if not ok:
            misses.append((q["qid"], q["qtype"], q["question"][:95], gold[0][:45], str(got)[:45]))

    total = sum(len(v) for v in by_type.values())
    correct = sum(sum(v) for v in by_type.values())
    print(f"ORACLE ACCURACY  {correct}/{total} = {100*correct/total:.1f}%\n")
    print(f"{'type':<13}{'acc':>12}{'gold-doc recall':>18}")
    for t in sorted(by_type):
        v = by_type[t]
        r = recall.get(t, [])
        rr = f"{100*sum(r)/len(r):.1f}%" if r else "-"
        print(f"{t:<13}{sum(v):>5}/{len(v):<6}{rr:>18}")

    if misses:
        print(f"\n{len(misses)} misses:")
        for qid, t, q, gold, got in misses:
            print(f"  [{t}] {qid}: {q}")
            print(f"       want={gold!r}  got={got!r}")


if __name__ == "__main__":
    main()
