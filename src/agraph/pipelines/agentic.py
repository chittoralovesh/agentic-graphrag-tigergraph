"""Pipeline 3 - Agentic GraphRAG.

An orchestrator chooses one action at a time from a tool catalogue, looks at
what came back, and decides whether to answer, retry differently, or dig
further. There is no fixed sequence: the next move is a function of the
question, the graph, the evidence already gathered, and what is still missing.

The parts the hackathon scores explicitly:

harness          ``AgentState`` carries the evidence ledger, the budget and the
                 stopping decision; every step is traced with tokens and latency
orchestrator     one LLM call per iteration picks the next tool and says why
specialists      entity linking, graph traversal, similarity search, document
                 fetch, aggregation, temporal hop, evidence evaluation
critic           after each observation, decides sufficient / insufficient and
                 names what is still missing - this is what drives re-planning
stopping         answers as soon as the evidence is sufficient; a question that
                 needs one hop costs one hop

Efficiency is a scored outcome, not an afterthought: burning ten steps on a
lookup is a failure even when the answer is right, so the orchestrator is
instructed to stop the moment the evidence settles the question.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..graphmodel import GraphTables
from ..llm import GeminiClient
from ..tools import structured as st
from ..trace import Trace
from .base import Answer, format_hits, synthesise

TOOL_CATALOGUE = """
link_event_by_title(title)
    Resolve a full event title to a graph vertex. Use when the question names
    an event almost verbatim.

read_attribute(event_id, attribute)
    Read one field of a resolved event. attribute is one of:
    nations, competitors, gold_raw, silver_raw, bronze_raw, win_value, venue,
    date_raw, games, title.

count_events(sport, games, attribute, op, threshold)
    Count events of a sport at a Games whose numeric field passes a test.
    op is one of > >= < <= ==. Scans the ENTIRE group, so it is exact.
    Use for "how many ... had more than N ...".

argmax_event(sport, games, attribute, direction)
    Rank every event of a sport at a Games on a numeric field and return the
    extreme one. direction is "highest" or "lowest".
    Use for "which ... had the highest/lowest number of ...".

find_event_by_venue_date(venue, date, games)
    Identify the event held at a venue on a date. Returns alternatives and
    flags ambiguity when several events share both.

edition_before_year(sport, discipline, year, season)
    The edition of a discipline at the Games immediately before a year.
    Use for "... held immediately before YYYY". season is Summer or Winter.

vector_search(query, top_k)
    Similarity search over document passages. Use when the question is not a
    clean structural shape, or to discover an entity name you do not yet know.

answer(value)
    Emit the final answer. Call this as soon as the evidence settles the
    question - extra steps cost tokens and are penalised.
"""

ORCHESTRATOR_PROMPT = """You are the orchestrator of a graph investigation over an Olympics corpus.

Choose the SINGLE next action. Prefer the cheapest action that can settle the
question. Call answer() as soon as the evidence is sufficient; do not gather
more than you need.

Counting and ranking questions must use count_events / argmax_event, which scan
the whole group. Passage retrieval cannot answer them: it returns a handful of
passages and the true supporting set is often 15 or more documents.

TOOLS
{catalogue}

QUESTION
{question}

EVIDENCE SO FAR
{evidence}

{guidance}
Reply with JSON only:
{{"tool": "<name>", "args": {{...}}, "rationale": "<one short sentence>"}}"""

# Tools whose successful result is self-validating: they either scan an entire
# group (count/argmax) or read a named field off an already-resolved vertex.
# Asking a critic "is this enough?" after one of these spends a call to confirm
# what the tool's own semantics already guarantee, so the agent skips straight
# to answering. Ambiguous or partial retrievals (venue+date, vector search) are
# still reviewed.
DEFINITIVE_TOOLS = {"count_events", "argmax_event", "read_attribute"}

CRITIC_PROMPT = """Decide whether the evidence answers the question.

QUESTION
{question}

EVIDENCE
{evidence}

Reply with JSON only:
{{"sufficient": true|false, "missing": "<what is still needed, or empty>"}}"""


@dataclass
class AgentState:
    """Everything the orchestrator may condition its next move on."""

    question: str
    evidence: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    resolved_event: str | None = None

    def ledger(self) -> str:
        if not self.evidence:
            return "(nothing gathered yet)"
        return "\n\n".join(self.evidence)

    def add(self, text: str, doc_ids: list[str]) -> None:
        self.evidence.append(text)
        for d in doc_ids:
            if d not in self.doc_ids:
                self.doc_ids.append(d)


class AgenticGraphRAG:
    name = "agentic"

    def __init__(
        self,
        graph: GraphTables,
        index,
        llm: GeminiClient,
        max_steps: int = 6,
        use_critic: bool = True,
        top_k: int = 5,
    ):
        self.g = graph
        self.index = index
        self.llm = llm
        self.max_steps = max_steps
        self.use_critic = use_critic
        self.top_k = top_k

    # -- main loop ---------------------------------------------------------
    def run(self, qid: str, question: str, qtype: str = "") -> Answer:
        trace = Trace(qid=qid, pipeline=self.name, question=question, qtype=qtype)
        state = AgentState(question=question)
        guidance = ""
        answer: str | None = None
        stop = "budget exhausted"

        for _ in range(self.max_steps):
            action = self._decide(trace, state, guidance)
            guidance = ""
            if action is None:
                stop = "orchestrator failed to produce an action"
                break

            tool = action.get("tool", "")
            args = action.get("args") or {}
            rationale = str(action.get("rationale", ""))[:200]

            if tool == "answer":
                answer = str(args.get("value", "")).strip() or None
                stop = "orchestrator judged the evidence sufficient"
                break

            ok = self._dispatch(trace, state, tool, args, rationale)
            if not ok:
                # A failed tool is information: tell the orchestrator so its
                # next choice differs instead of repeating the same move.
                state.failures.append(tool)
                guidance = (
                    f"The previous action ({tool}) returned nothing useful. "
                    "Choose a DIFFERENT tool or different arguments."
                )
                trace.strategy_changes += 1
                continue

            if tool in DEFINITIVE_TOOLS:
                stop = f"{tool} returned a complete result; no review needed"
                break

            if self.use_critic:
                verdict = self._criticise(trace, state)
                if verdict.get("sufficient"):
                    stop = "evidence critic judged the evidence sufficient"
                    break
                missing = str(verdict.get("missing", ""))[:200]
                if missing:
                    guidance = f"Still missing: {missing}"

        if answer is None:
            answer = synthesise(self.llm, trace, question, state.ledger())
            if stop == "budget exhausted" and answer:
                stop = "answered after evidence evaluation"

        trace_obj = trace.finish(answer, stop_reason=stop)
        # Citations come from the evidence ledger, not just the final step.
        for d in state.doc_ids:
            if d not in trace_obj.citations:
                trace_obj.citations.append(d)
        return Answer(answer, trace_obj)

    # -- orchestrator ------------------------------------------------------
    def _decide(self, trace: Trace, state: AgentState, guidance: str) -> dict | None:
        step = trace.step("orchestrate", "choose the next action")
        prompt = ORCHESTRATOR_PROMPT.format(
            catalogue=TOOL_CATALOGUE,
            question=state.question,
            evidence=state.ledger(),
            guidance=(guidance + "\n") if guidance else "",
        )
        try:
            reply = self.llm.generate(prompt, temperature=0.0)
            step.tokens = reply.tokens
            action = reply.json()
        except Exception as exc:  # noqa: BLE001
            step.error = str(exc)[:200]
            return None
        if not isinstance(action, dict) or "tool" not in action:
            step.error = "malformed action"
            return None
        step.result_brief = f"{action.get('tool')} - {str(action.get('rationale',''))[:90]}"
        return action

    # -- critic ------------------------------------------------------------
    def _criticise(self, trace: Trace, state: AgentState) -> dict:
        step = trace.step("evidence_critic", "is the evidence sufficient yet?")
        try:
            reply = self.llm.generate(
                CRITIC_PROMPT.format(question=state.question, evidence=state.ledger())
            )
            step.tokens = reply.tokens
            verdict = reply.json()
            if not isinstance(verdict, dict):
                raise ValueError("critic returned a non-object")
        except Exception as exc:  # noqa: BLE001
            step.error = str(exc)[:200]
            return {"sufficient": False, "missing": ""}
        step.result_brief = json.dumps(verdict)[:160]
        return verdict

    # -- specialist dispatch ----------------------------------------------
    def _dispatch(
        self, trace: Trace, state: AgentState, tool: str, args: dict, rationale: str
    ) -> bool:
        g = self.g
        step = trace.step(tool, rationale, **_safe_args(args))
        try:
            if tool == "link_event_by_title":
                ev = st.link_event_by_title(g, str(args.get("title", "")))
                if ev.value:
                    state.resolved_event = ev.value
            elif tool == "read_attribute":
                target = str(args.get("event_id") or state.resolved_event or "")
                ev = st.get_event_attribute(g, target, str(args.get("attribute", "nations")))
            elif tool == "count_events":
                ev = st.count_events_where(
                    g,
                    str(args.get("sport", "")),
                    str(args.get("games", "")),
                    str(args.get("attribute", "competitors")),
                    str(args.get("op", ">")),
                    int(args.get("threshold") or 0),
                )
            elif tool == "argmax_event":
                ev = st.argmax_event(
                    g,
                    str(args.get("sport", "")),
                    str(args.get("games", "")),
                    str(args.get("attribute", "competitors")),
                )
            elif tool == "find_event_by_venue_date":
                ev = st.find_event_by_venue_date(
                    g, str(args.get("venue", "")), str(args.get("date", "")), args.get("games")
                )
                if ev.value:
                    state.resolved_event = ev.value
            elif tool == "edition_before_year":
                ev = st.edition_before_year(
                    g,
                    str(args.get("sport", "")),
                    str(args.get("discipline", "")),
                    int(args.get("year") or 0),
                    args.get("season"),
                )
                if ev.value:
                    state.resolved_event = ev.value
            elif tool == "vector_search":
                k = int(args.get("top_k") or self.top_k)
                hits = self.index.search(str(args.get("query", state.question)), top_k=k)
                doc_ids = list(dict.fromkeys(h.doc_id for h in hits))
                step.doc_ids = doc_ids
                step.result_brief = f"{len(hits)} chunks"
                state.tools_called.append(tool)
                state.add(format_hits(hits), doc_ids)
                return bool(hits)
            else:
                step.error = f"unknown tool {tool!r}"
                return False
        except Exception as exc:  # noqa: BLE001
            step.error = str(exc)[:200]
            return False

        step.doc_ids = ev.doc_ids
        step.result_brief = ev.brief()
        state.tools_called.append(tool)
        if ev.value is None:
            state.add(f"{tool}: no result. {ev.note}", ev.doc_ids)
            return False
        state.add(_render_evidence(tool, ev), ev.doc_ids)
        return True


def _render_evidence(tool: str, ev) -> str:
    out = [f"{tool} -> {ev.kind} = {ev.value}"]
    if ev.detail:
        out.append(json.dumps(ev.detail, ensure_ascii=False, default=str)[:2500])
    if ev.note:
        out.append(f"note: {ev.note}")
    if ev.doc_ids:
        out.append(f"documents: {', '.join(ev.doc_ids[:40])}")
    return "\n".join(out)


def _safe_args(args: dict) -> dict[str, Any]:
    return {k: v for k, v in list(args.items())[:8] if isinstance(v, (str, int, float, bool))}
