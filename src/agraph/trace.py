"""Execution traces and token accounting.

The hackathon scores *whether the extra reasoning was worth the extra tokens*,
so cost is a first-class measurement rather than a log line. Every pipeline
records the same ``Trace`` shape, which makes the three-way comparison exact:
same question, same units, same accounting.

Recorded per step: tool, rationale, latency, tokens in/out, documents touched.
Recorded per run: totals, step count, retrieval methods used, stop reason, and
whether the agent changed strategy mid-investigation.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TokenUsage:
    prompt: int = 0
    completion: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion

    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        self.prompt += other.prompt
        self.completion += other.completion
        return self


@dataclass
class Step:
    """One action inside an investigation."""

    index: int
    tool: str
    rationale: str = ""
    args: dict = field(default_factory=dict)
    result_brief: str = ""
    doc_ids: list[str] = field(default_factory=list)
    tokens: TokenUsage = field(default_factory=TokenUsage)
    seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tokens"] = {"prompt": self.tokens.prompt, "completion": self.tokens.completion,
                       "total": self.tokens.total}
        return d


@dataclass
class Trace:
    """The full record of answering one question with one pipeline."""

    qid: str
    pipeline: str
    question: str
    qtype: str = ""
    answer: str | None = None
    steps: list[Step] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    stop_reason: str = ""
    strategy_changes: int = 0
    seconds: float = 0.0
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    # -- recording ---------------------------------------------------------
    def step(self, tool: str, rationale: str = "", **args: Any) -> Step:
        s = Step(index=len(self.steps) + 1, tool=tool, rationale=rationale, args=args)
        self.steps.append(s)
        return s

    def finish(self, answer: str | None, stop_reason: str = "") -> "Trace":
        self.answer = answer
        self.stop_reason = stop_reason or self.stop_reason
        self.seconds = round(time.perf_counter() - self._t0, 3)
        seen: dict[str, None] = {}
        for s in self.steps:
            for d in s.doc_ids:
                seen.setdefault(d, None)
        self.citations = list(seen)
        return self

    # -- derived metrics ---------------------------------------------------
    @property
    def tokens(self) -> TokenUsage:
        total = TokenUsage()
        for s in self.steps:
            total += s.tokens
        return total

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def n_llm_calls(self) -> int:
        return sum(1 for s in self.steps if s.tokens.total > 0)

    @property
    def tools_used(self) -> list[str]:
        return list(dict.fromkeys(s.tool for s in self.steps))

    def to_dict(self) -> dict:
        t = self.tokens
        return {
            "qid": self.qid,
            "pipeline": self.pipeline,
            "question": self.question,
            "qtype": self.qtype,
            "answer": self.answer,
            "citations": self.citations,
            "n_citations": len(self.citations),
            "n_steps": self.n_steps,
            "n_llm_calls": self.n_llm_calls,
            "tools_used": self.tools_used,
            "stop_reason": self.stop_reason,
            "strategy_changes": self.strategy_changes,
            "seconds": self.seconds,
            "tokens": {"prompt": t.prompt, "completion": t.completion, "total": t.total},
            "steps": [s.to_dict() for s in self.steps],
        }


class TraceWriter:
    """Append traces to a JSONL file as they complete."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, trace: Trace) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")


def load_traces(path: str | Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out
