"""Pipeline 2 - GraphRAG with a fixed retrieval plan.

Structure is used, but the control flow is not adaptive: every question walks
the same four stages.

    1. classify the question into a retrieval shape (one LLM call)
    2. link the mentioned entities onto graph vertices
    3. execute the retrieval for that shape
    4. synthesise

This is the honest middle term of the experiment. It has the *same* graph
access as the agentic pipeline, so any difference between pipelines 2 and 3 is
attributable to adaptive control - re-planning, evaluating intermediate
results, recovering from a bad first move - rather than to graph access itself.
Without this control, "agentic beats RAG" would conflate two separate causes.
"""

from __future__ import annotations

import json

from ..graphmodel import GraphTables
from ..llm import GeminiClient
from ..tools import structured as st
from ..trace import Trace
from .base import Answer, format_hits, synthesise

ROUTER_PROMPT = """Classify this question about the Olympics into one retrieval shape and
extract its parameters. Reply with JSON only.

Shapes:
  "attribute"   one named event, read a field. e.g. "How many nations competed in X?"
  "count"       count events of a sport at a Games passing a numeric test
  "argmax"      which event of a sport at a Games has the highest/lowest value
  "venue_date"  identify the event held at a venue on a date, then read a field
  "edition"     an event at the Games immediately before/after a year
  "semantic"    anything else; fall back to passage retrieval

JSON fields (include only those relevant):
  shape, event_title, sport, games, attribute, op, threshold,
  venue, date, discipline, year, season, direction

attribute must be one of: nations, competitors, gold_raw, silver_raw, bronze_raw, win_value
op must be one of: >, >=, <, <=, ==
season must be "Summer" or "Winter".

Question: {question}

JSON:"""


class GraphRAGPipeline:
    name = "graphrag"

    def __init__(self, graph: GraphTables, index, llm: GeminiClient, top_k: int = 5):
        self.g = graph
        self.index = index
        self.llm = llm
        self.top_k = top_k

    def run(self, qid: str, question: str, qtype: str = "") -> Answer:
        trace = Trace(qid=qid, pipeline=self.name, question=question, qtype=qtype)
        plan = self._route(trace, question)
        shape = (plan.get("shape") or "semantic").lower()

        ev = self._execute(trace, shape, plan)
        if ev is None:
            ev = self._semantic(trace, question)

        answer = synthesise(self.llm, trace, question, ev)
        return Answer(answer, trace.finish(answer, stop_reason=f"fixed plan: {shape}"))

    # -- stage 1 -----------------------------------------------------------
    def _route(self, trace: Trace, question: str) -> dict:
        step = trace.step("classify", "map the question onto one fixed retrieval shape")
        try:
            reply = self.llm.generate(ROUTER_PROMPT.format(question=question))
            step.tokens = reply.tokens
            plan = reply.json()
            if not isinstance(plan, dict):
                raise ValueError("router did not return an object")
        except Exception as exc:  # noqa: BLE001
            step.error = str(exc)[:200]
            return {"shape": "semantic"}
        step.result_brief = json.dumps(plan)[:200]
        return plan

    # -- stage 2 and 3 -----------------------------------------------------
    def _execute(self, trace: Trace, shape: str, plan: dict) -> str | None:
        g = self.g
        try:
            if shape == "attribute":
                step = trace.step("entity_link", "resolve the event title to a vertex")
                ev = st.link_event_by_title(g, plan.get("event_title") or "")
                step.doc_ids, step.result_brief = ev.doc_ids, ev.brief()
                if not ev.value:
                    return None
                step = trace.step("read_attribute", "read the requested field")
                got = st.get_event_attribute(g, ev.value, plan.get("attribute") or "nations")
                step.doc_ids, step.result_brief = got.doc_ids, got.brief()
                return _render(got)

            if shape == "count":
                step = trace.step("aggregate", "filtered count over the whole group")
                got = st.count_events_where(
                    g,
                    plan.get("sport") or "",
                    plan.get("games") or "",
                    plan.get("attribute") or "competitors",
                    plan.get("op") or ">",
                    int(plan.get("threshold") or 0),
                )
                step.doc_ids, step.result_brief = got.doc_ids, got.brief()
                return _render(got) if got.value is not None else None

            if shape == "argmax":
                step = trace.step("argmax", "rank the whole group on a numeric field")
                got = st.argmax_event(
                    g, plan.get("sport") or "", plan.get("games") or "",
                    plan.get("attribute") or "competitors",
                )
                step.doc_ids, step.result_brief = got.doc_ids, got.brief()
                return _render(got) if got.value else None

            if shape == "venue_date":
                step = trace.step("venue_date_lookup", "identify the event from venue and date")
                ev = st.find_event_by_venue_date(
                    g, plan.get("venue") or "", plan.get("date") or "", plan.get("games")
                )
                step.doc_ids, step.result_brief = ev.doc_ids, ev.brief()
                if not ev.value:
                    return None
                step = trace.step("read_attribute", "read the medallist field")
                got = st.get_event_attribute(g, ev.value, plan.get("attribute") or "gold_raw")
                step.doc_ids, step.result_brief = got.doc_ids, got.brief()
                return _render(got, extra=ev)

            if shape == "edition":
                step = trace.step("temporal_hop", "walk to the preceding edition")
                ev = st.edition_before_year(
                    g, plan.get("sport") or "", plan.get("discipline") or "",
                    int(plan.get("year") or 0), plan.get("season"),
                )
                step.doc_ids, step.result_brief = ev.doc_ids, ev.brief()
                if not ev.value:
                    return None
                step = trace.step("read_attribute", "read the medallist field")
                got = st.get_event_attribute(g, ev.value, plan.get("attribute") or "gold_raw")
                step.doc_ids, step.result_brief = got.doc_ids, got.brief()
                return _render(got, extra=ev)
        except Exception as exc:  # noqa: BLE001
            trace.steps[-1].error = str(exc)[:200]
            return None
        return None

    def _semantic(self, trace: Trace, question: str) -> str:
        step = trace.step("vector_search", "fall back to passage retrieval", top_k=self.top_k)
        hits = self.index.search(question, top_k=self.top_k)
        step.doc_ids = list(dict.fromkeys(h.doc_id for h in hits))
        step.result_brief = f"{len(hits)} chunks"
        return format_hits(hits)


def _render(ev, extra=None) -> str:
    """Render structured evidence as a compact, citable block."""
    lines = []
    if extra is not None and extra.detail:
        lines.append(f"Resolved event: {json.dumps(extra.detail, ensure_ascii=False, default=str)}")
    lines.append(f"{ev.kind} = {ev.value}")
    if ev.detail:
        lines.append(json.dumps(ev.detail, ensure_ascii=False, default=str)[:2500])
    if ev.note:
        lines.append(f"note: {ev.note}")
    lines.append(f"supporting documents: {', '.join(ev.doc_ids[:40])}")
    return "\n".join(lines)
