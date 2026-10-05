"""Shared plumbing for the three pipelines.

All three answer the same questions, record the same ``Trace`` shape and are
given the same final-answer instructions. The only thing that differs is how
evidence is gathered - which is precisely the variable under test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..llm import GeminiClient
from ..trace import Step, Trace

ANSWER_RULES = """You answer questions about the Olympic Games using ONLY the evidence provided.

Rules:
- The evidence is the single source of truth. If it disagrees with what you
  remember about the real world, the evidence wins.
- Answer with the shortest exact form: a number ("23"), a person's name
  ("Chen Ding"), or a full event title. No sentences, no units, no explanation.
- For team events the medallist field may run names together
  ("Dani KingLaura Trott"). Reproduce it exactly as written.
- If the evidence is insufficient, reply exactly: INSUFFICIENT
"""


@dataclass
class Answer:
    text: str | None
    trace: Trace


class Pipeline(Protocol):
    name: str

    def run(self, qid: str, question: str, qtype: str = "") -> Answer: ...


def synthesise(
    llm: GeminiClient,
    trace: Trace,
    question: str,
    evidence_block: str,
    step_name: str = "synthesize",
    rationale: str = "compose the final answer from gathered evidence",
) -> str | None:
    """Final answer step, shared so generation is identical across pipelines."""
    step = trace.step(step_name, rationale)
    prompt = f"{ANSWER_RULES}\n\nEVIDENCE\n{evidence_block}\n\nQUESTION\n{question}\n\nANSWER:"
    try:
        reply = llm.generate(prompt, temperature=0.0)
    except Exception as exc:  # noqa: BLE001
        step.error = str(exc)[:300]
        return None
    step.tokens = reply.tokens
    text = (reply.text or "").strip()
    step.result_brief = text[:160]
    return None if text.upper().startswith("INSUFFICIENT") else text


def format_hits(hits, max_chars: int = 1400) -> str:
    """Render retrieved chunks as a numbered, citable evidence block."""
    parts = []
    for i, h in enumerate(hits, start=1):
        body = h.text if len(h.text) <= max_chars else h.text[:max_chars] + " ..."
        parts.append(f"[{i}] ({h.doc_id}) {h.title}\n{body}")
    return "\n\n".join(parts) if parts else "(no evidence retrieved)"
