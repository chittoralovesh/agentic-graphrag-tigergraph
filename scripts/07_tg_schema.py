"""Create the TigerGraph schema for the Olympic GraphRAG graph.

Mirrors src/agraph/graphmodel.py so the structured tools mean the same thing
whether they run in-process or as GSQL against TigerGraph.

Run:  python scripts/07_tg_schema.py [--drop]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agraph.llm import load_env  # noqa: E402
from agraph.store.tigergraph import connect  # noqa: E402

GRAPH = "OlympicGraphRAG"

SCHEMA_JOB = """USE GRAPH {g}
CREATE SCHEMA_CHANGE JOB build_schema FOR GRAPH {g} {{
  ADD VERTEX Document (PRIMARY_ID doc_id STRING, title STRING, url STRING,
      wikidata_qid STRING, infobox_type STRING, approx_tokens INT)
      WITH primary_id_as_attribute="true";
  ADD VERTEX OlympicEvent (PRIMARY_ID qid STRING, title STRING, sport STRING,
      event_name STRING, event_key STRING, games STRING, year INT, season STRING,
      venue STRING, competitors INT, nations INT, win_value STRING,
      date_raw STRING, date_month INT, date_day_start INT, date_day_end INT,
      gold_raw STRING, silver_raw STRING, bronze_raw STRING,
      gold_noc STRING, silver_noc STRING, bronze_noc STRING, url STRING)
      WITH primary_id_as_attribute="true";
  ADD VERTEX Games (PRIMARY_ID name STRING, year INT, season STRING)
      WITH primary_id_as_attribute="true";
  ADD VERTEX Sport (PRIMARY_ID name STRING) WITH primary_id_as_attribute="true";
  ADD VERTEX Venue (PRIMARY_ID vkey STRING, name STRING)
      WITH primary_id_as_attribute="true";
  ADD VERTEX Athlete (PRIMARY_ID akey STRING, name STRING)
      WITH primary_id_as_attribute="true";
  ADD VERTEX NOC (PRIMARY_ID code STRING) WITH primary_id_as_attribute="true";
  ADD VERTEX Chunk (PRIMARY_ID chunk_id STRING, doc_id STRING, title STRING,
      text STRING, ordinal INT) WITH primary_id_as_attribute="true";

  ADD DIRECTED EDGE HAS_EVENT (FROM Games, TO OlympicEvent) WITH REVERSE_EDGE="EVENT_OF";
  ADD DIRECTED EDGE OF_SPORT (FROM OlympicEvent, TO Sport) WITH REVERSE_EDGE="SPORT_OF";
  ADD DIRECTED EDGE AT_VENUE (FROM OlympicEvent, TO Venue) WITH REVERSE_EDGE="VENUE_OF";
  ADD DIRECTED EDGE WON_GOLD (FROM OlympicEvent, TO Athlete) WITH REVERSE_EDGE="GOLD_IN";
  ADD DIRECTED EDGE WON_SILVER (FROM OlympicEvent, TO Athlete) WITH REVERSE_EDGE="SILVER_IN";
  ADD DIRECTED EDGE WON_BRONZE (FROM OlympicEvent, TO Athlete) WITH REVERSE_EDGE="BRONZE_IN";
  ADD DIRECTED EDGE REPRESENTS (FROM Athlete, TO NOC) WITH REVERSE_EDGE="REPRESENTED_BY";
  ADD DIRECTED EDGE PREV_EDITION (FROM OlympicEvent, TO OlympicEvent) WITH REVERSE_EDGE="IS_NEXT_OF";
  ADD DIRECTED EDGE NEXT_EDITION (FROM OlympicEvent, TO OlympicEvent) WITH REVERSE_EDGE="IS_PREV_OF";
  ADD DIRECTED EDGE SOURCED_FROM (FROM OlympicEvent, TO Document) WITH REVERSE_EDGE="SOURCE_OF";
  ADD DIRECTED EDGE HAS_CHUNK (FROM Document, TO Chunk) WITH REVERSE_EDGE="CHUNK_OF";
}}
RUN SCHEMA_CHANGE JOB build_schema
DROP JOB build_schema"""

# TigerVector lives on its own statement: the embedding dimension must match
# whatever produced the vectors (768 here).
VECTOR_ATTR = """USE GRAPH {g}
CREATE SCHEMA_CHANGE JOB add_vec FOR GRAPH {g} {{
  ADD VECTOR ATTRIBUTE Chunk.embedding(dimension=768, metric="COSINE");
}}
RUN SCHEMA_CHANGE JOB add_vec
DROP JOB add_vec"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", action="store_true", help="drop and recreate the graph")
    args = ap.parse_args()

    load_env()
    conn = connect(graphname=GRAPH, verify=False)

    if args.drop:
        print("dropping graph ...")
        print(_run(conn, f"DROP GRAPH {GRAPH}")[:200])
        print(_run(conn, f"CREATE GRAPH {GRAPH}()")[:200])

    print("creating vertices and edges ...")
    print(_run(conn, SCHEMA_JOB.format(g=GRAPH))[-1200:])

    print("\nadding the TigerVector attribute ...")
    print(_run(conn, VECTOR_ATTR.format(g=GRAPH))[-600:])

    print("\nschema now:")
    conn.graphname = GRAPH
    try:
        print("  vertices:", conn.getVertexTypes())
        print("  edges   :", conn.getEdgeTypes())
    except Exception as exc:  # noqa: BLE001
        print("  (could not list:", str(exc)[:160], ")")


def _run(conn, stmt: str) -> str:
    try:
        return str(conn.gsql(stmt))
    except Exception as exc:  # noqa: BLE001
        return f"[gsql error] {str(exc)[:600]}"


if __name__ == "__main__":
    main()
