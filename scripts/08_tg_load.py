"""Load the corpus, the graph projection and the chunks into TigerGraph.

Upserts in batches through the REST endpoint. Re-running is safe: upserts are
idempotent on primary id, so a partial load can simply be repeated.

Run:  python scripts/08_tg_load.py [--skip-chunks] [--batch 500]
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agraph.corpus import norm  # noqa: E402
from agraph.llm import load_env  # noqa: E402
from agraph.store.tigergraph import connect  # noqa: E402

GRAPH = "OlympicGraphRAG"


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def upsert_vertices(conn, vtype: str, rows: dict, batch: int) -> int:
    total = 0
    items = list(rows.items())
    for chunk in batched(items, batch):
        total += conn.upsertVertices(vtype, chunk)
    return total


def upsert_edges(conn, src: str, etype: str, tgt: str, pairs: list, batch: int) -> int:
    total = 0
    rows = [(a, b, {}) for a, b in pairs]
    for chunk in batched(rows, batch):
        total += conn.upsertEdges(src, etype, tgt, chunk)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--skip-chunks", action="store_true")
    ap.add_argument("--skip-docs", action="store_true")
    args = ap.parse_args()

    load_env()
    conn = connect(graphname=GRAPH, verify=False)
    conn.graphname = GRAPH

    with open(ROOT / "artifacts" / "graph.pkl", "rb") as fh:
        g = pickle.load(fh)

    t0 = time.perf_counter()

    if not args.skip_docs:
        docs = {
            d.doc_id: {
                "title": d.title, "url": d.url, "wikidata_qid": d.wikidata_qid,
                "infobox_type": d.infobox_type or "", "approx_tokens": d.approx_tokens,
            }
            for d in g.documents
        }
        print(f"Document   {upsert_vertices(conn, 'Document', docs, args.batch)}")

    events = {}
    for e in g.events.values():
        events[e.qid] = {
            "title": e.title, "sport": e.sport, "event_name": e.event_name,
            "event_key": e.event_key, "games": e.games or "", "year": e.year or 0,
            "season": e.season or "", "venue": e.venue or "",
            "competitors": e.competitors if e.competitors is not None else -1,
            "nations": e.nations if e.nations is not None else -1,
            "win_value": e.win_value or "", "date_raw": e.date_raw or "",
            "date_month": e.date_month or 0,
            "date_day_start": e.date_day_start or 0,
            "date_day_end": (e.date.day_end or 0) if e.date else 0,
            "gold_raw": e.gold_raw or "", "silver_raw": e.silver_raw or "",
            "bronze_raw": e.bronze_raw or "", "gold_noc": e.gold_noc or "",
            "silver_noc": e.silver_noc or "", "bronze_noc": e.bronze_noc or "",
            "url": e.url,
        }
    print(f"OlympicEvent {upsert_vertices(conn, 'OlympicEvent', events, args.batch)}")

    print(f"Games      {upsert_vertices(conn, 'Games', {k: {'name': v['name'], 'year': v['year'] or 0, 'season': v['season'] or ''} for k, v in g.games.items()}, args.batch)}")
    print(f"Sport      {upsert_vertices(conn, 'Sport', {s: {'name': s} for s in g.sports}, args.batch)}")
    print(f"Venue      {upsert_vertices(conn, 'Venue', {k: {'name': v} for k, v in g.venues.items()}, args.batch)}")
    print(f"Athlete    {upsert_vertices(conn, 'Athlete', {k: {'name': v} for k, v in g.athletes.items()}, args.batch)}")
    print(f"NOC        {upsert_vertices(conn, 'NOC', {c: {} for c in g.nocs}, args.batch)}")

    edge_spec = {
        "HAS_EVENT": ("Games", "OlympicEvent"),
        "OF_SPORT": ("OlympicEvent", "Sport"),
        "AT_VENUE": ("OlympicEvent", "Venue"),
        "WON_GOLD": ("OlympicEvent", "Athlete"),
        "WON_SILVER": ("OlympicEvent", "Athlete"),
        "WON_BRONZE": ("OlympicEvent", "Athlete"),
        "REPRESENTS": ("Athlete", "NOC"),
        "PREV_EDITION": ("OlympicEvent", "OlympicEvent"),
        "NEXT_EDITION": ("OlympicEvent", "OlympicEvent"),
        "SOURCED_FROM": ("OlympicEvent", "Document"),
    }
    for etype, (src, tgt) in edge_spec.items():
        pairs = g.edges.get(etype, [])
        if not pairs:
            continue
        n = upsert_edges(conn, src, etype, tgt, pairs, args.batch)
        print(f"{etype:<14} {n}")

    if not args.skip_chunks:
        with open(ROOT / "artifacts" / "chunks.pkl", "rb") as fh:
            chunks = pickle.load(fh)
        rows = {
            c.chunk_id: {
                "doc_id": c.doc_id, "title": c.title,
                # Keep the stored passage bounded: TigerGraph string attributes
                # are fine with this, and retrieval only ever shows an excerpt.
                "text": c.text[:6000], "ordinal": c.ordinal,
            }
            for c in chunks
        }
        print(f"Chunk      {upsert_vertices(conn, 'Chunk', rows, args.batch)}")
        pairs = [(c.doc_id, c.chunk_id) for c in chunks]
        print(f"HAS_CHUNK      {upsert_edges(conn, 'Document', 'HAS_CHUNK', 'Chunk', pairs, args.batch)}")

    print(f"\nloaded in {time.perf_counter() - t0:.1f}s")
    for v in ("Document", "OlympicEvent", "Games", "Sport", "Venue", "Athlete", "NOC", "Chunk"):
        try:
            print(f"  {v:<14}{conn.getVertexCount(v)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {v:<14}? {str(exc)[:80]}")


if __name__ == "__main__":
    main()
