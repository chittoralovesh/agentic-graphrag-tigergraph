"""TigerGraph schema, loading and connection handling.

Supports both auth styles Savanna exposes: a database secret (preferred - what
the Database Secrets page issues) and classic username/password for a local
Community Edition. Connection details come from the environment so the same
code runs against either.

The schema mirrors ``graphmodel.py`` exactly, so the structured tools have the
same semantics whether they run in-process or as GSQL.
"""

from __future__ import annotations

import os
from typing import Any

VERTICES = """
CREATE VERTEX Document (
    PRIMARY_ID doc_id STRING,
    title STRING, url STRING, wikidata_qid STRING,
    infobox_type STRING, approx_tokens INT
) WITH primary_id_as_attribute="true"

CREATE VERTEX OlympicEvent (
    PRIMARY_ID qid STRING,
    title STRING, sport STRING, event_name STRING, event_key STRING,
    games STRING, year INT, season STRING, venue STRING,
    competitors INT, nations INT, win_value STRING,
    date_raw STRING, date_month INT, date_day_start INT, date_day_end INT,
    gold_raw STRING, silver_raw STRING, bronze_raw STRING,
    gold_noc STRING, silver_noc STRING, bronze_noc STRING,
    url STRING
) WITH primary_id_as_attribute="true"

CREATE VERTEX Games (PRIMARY_ID name STRING, year INT, season STRING)
    WITH primary_id_as_attribute="true"
CREATE VERTEX Sport (PRIMARY_ID name STRING) WITH primary_id_as_attribute="true"
CREATE VERTEX Venue (PRIMARY_ID vkey STRING, name STRING)
    WITH primary_id_as_attribute="true"
CREATE VERTEX Athlete (PRIMARY_ID akey STRING, name STRING)
    WITH primary_id_as_attribute="true"
CREATE VERTEX NOC (PRIMARY_ID code STRING) WITH primary_id_as_attribute="true"

CREATE VERTEX Chunk (
    PRIMARY_ID chunk_id STRING,
    doc_id STRING, title STRING, text STRING, ordinal INT,
    embedding LIST<DOUBLE>
) WITH primary_id_as_attribute="true"
"""

EDGES = """
CREATE DIRECTED EDGE HAS_EVENT (FROM Games, TO OlympicEvent) WITH REVERSE_EDGE="EVENT_OF"
CREATE DIRECTED EDGE OF_SPORT (FROM OlympicEvent, TO Sport) WITH REVERSE_EDGE="SPORT_OF"
CREATE DIRECTED EDGE AT_VENUE (FROM OlympicEvent, TO Venue) WITH REVERSE_EDGE="VENUE_OF"
CREATE DIRECTED EDGE WON_GOLD (FROM OlympicEvent, TO Athlete) WITH REVERSE_EDGE="GOLD_IN"
CREATE DIRECTED EDGE WON_SILVER (FROM OlympicEvent, TO Athlete) WITH REVERSE_EDGE="SILVER_IN"
CREATE DIRECTED EDGE WON_BRONZE (FROM OlympicEvent, TO Athlete) WITH REVERSE_EDGE="BRONZE_IN"
CREATE DIRECTED EDGE REPRESENTS (FROM Athlete, TO NOC) WITH REVERSE_EDGE="REPRESENTED_BY"
CREATE DIRECTED EDGE PREV_EDITION (FROM OlympicEvent, TO OlympicEvent) WITH REVERSE_EDGE="NEXT_EDITION_OF"
CREATE DIRECTED EDGE NEXT_EDITION (FROM OlympicEvent, TO OlympicEvent) WITH REVERSE_EDGE="PREV_EDITION_OF"
CREATE DIRECTED EDGE SOURCED_FROM (FROM OlympicEvent, TO Document) WITH REVERSE_EDGE="SOURCE_OF"
CREATE DIRECTED EDGE HAS_CHUNK (FROM Document, TO Chunk) WITH REVERSE_EDGE="CHUNK_OF"
"""


def connect(graphname: str | None = None, verify: bool = True):
    """Open a connection using whichever credentials the environment provides."""
    import pyTigerGraph as tg

    host = os.environ.get("TG_HOST", "http://127.0.0.1")
    graph = graphname or os.environ.get("TG_GRAPHNAME", "OlympicGraphRAG")
    secret = os.environ.get("TG_SECRET", "").strip()
    username = os.environ.get("TG_USERNAME", "tigergraph")
    password = os.environ.get("TG_PASSWORD", "tigergraph")

    kwargs: dict[str, Any] = {"host": host, "graphname": graph}
    # Savanna exposes 443 for both REST++ and GSQL through its gateway.
    if "tgcloud.io" in host or host.startswith("https://"):
        kwargs["restppPort"] = os.environ.get("TG_SSL_PORT", "443")
        kwargs["gsPort"] = os.environ.get("TG_SSL_PORT", "443")
    else:
        kwargs["restppPort"] = os.environ.get("TG_RESTPP_PORT", "9000")
        kwargs["gsPort"] = os.environ.get("TG_GS_PORT", "14240")

    if secret:
        conn = tg.TigerGraphConnection(gsqlSecret=secret, **kwargs)
        try:
            conn.getToken(secret)
        except Exception:
            pass  # some versions authenticate implicitly from the secret
    else:
        conn = tg.TigerGraphConnection(username=username, password=password, **kwargs)
        try:
            conn.getToken(conn.createSecret())
        except Exception:
            pass

    if verify:
        conn.echo()
    return conn


def create_schema(conn, graphname: str, drop_first: bool = False) -> str:
    """Create the graph, vertex and edge types. Idempotent when drop_first."""
    out = []
    if drop_first:
        out.append(_gsql(conn, f"DROP GRAPH {graphname}", graph=None))
    out.append(_gsql(conn, f"CREATE GRAPH {graphname}()", graph=None))
    schema = "\n".join(
        [f"USE GRAPH {graphname}", "BEGIN", VERTICES.strip(), EDGES.strip(), "END"]
    )
    # Vertex/edge creation runs as a schema change job in 4.x.
    job = f"""USE GRAPH {graphname}
CREATE SCHEMA_CHANGE JOB build_{graphname} FOR GRAPH {graphname} {{
{_indent(VERTICES)}
{_indent(EDGES)}
}}
RUN SCHEMA_CHANGE JOB build_{graphname}
DROP JOB build_{graphname}"""
    out.append(_gsql(conn, job, graph=None))
    return "\n".join(o for o in out if o)


def _indent(block: str) -> str:
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    merged: list[str] = []
    buf = ""
    for ln in lines:
        if ln.startswith("CREATE") and buf:
            merged.append(buf)
            buf = ln
        else:
            buf = f"{buf} {ln}".strip()
    if buf:
        merged.append(buf)
    return "\n".join(f"  ADD {m[len('CREATE '):]};" if m.startswith("CREATE") else f"  {m};" for m in merged)


def _gsql(conn, statement: str, graph: str | None = None) -> str:
    try:
        return conn.gsql(statement, graph=graph) if graph else conn.gsql(statement)
    except Exception as exc:  # noqa: BLE001
        return f"[gsql error] {str(exc)[:400]}"
