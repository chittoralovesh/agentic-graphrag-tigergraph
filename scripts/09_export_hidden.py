"""Export the 50 hidden-question answers in submission form.

The organisers score these against held-out ground truth, and ask for the raw
outputs: tokens used, the answer generated, and the agentic trace. This writes
one self-contained JSONL where each line carries the answer, the full token
accounting, the ordered reasoning steps and the cited documents.

Run:  python scripts/09_export_hidden.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agraph.trace import load_traces  # noqa: E402

RUN = ROOT / "artifacts" / "runs" / "hidden"
OUT = ROOT / "artifacts" / "submission"


def main() -> None:
    traces_path = RUN / "traces_agentic.jsonl"
    if not traces_path.exists():
        raise SystemExit(
            "no hidden run found. First:\n"
            "  python scripts/05_run_benchmark.py --split hidden --pipelines agentic"
        )
    traces = load_traces(traces_path)
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for t in traces:
        rows.append(
            {
                "qid": t["qid"],
                "question": t["question"],
                "answer": t["answer"],
                "pipeline": "agentic_graphrag",
                "tokens": t["tokens"],
                "n_steps": t["n_steps"],
                "n_llm_calls": t["n_llm_calls"],
                "seconds": t["seconds"],
                "stop_reason": t["stop_reason"],
                "strategy_changes": t["strategy_changes"],
                "citations": t["citations"],
                "n_citations": t["n_citations"],
                "trace": [
                    {
                        "step": s["index"],
                        "tool": s["tool"],
                        "rationale": s["rationale"],
                        "args": s["args"],
                        "result": s["result_brief"],
                        "documents": s["doc_ids"],
                        "tokens": s["tokens"]["total"],
                        "seconds": s["seconds"],
                        "error": s["error"],
                    }
                    for s in t["steps"]
                ],
            }
        )

    answers = OUT / "hidden_answers.jsonl"
    answers.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )

    total = sum(r["tokens"]["total"] for r in rows)
    answered = sum(1 for r in rows if r["answer"])
    summary = {
        "n_questions": len(rows),
        "answered": answered,
        "unanswered": len(rows) - answered,
        "total_tokens": total,
        "mean_tokens_per_question": round(total / max(len(rows), 1), 1),
        "mean_steps": round(sum(r["n_steps"] for r in rows) / max(len(rows), 1), 2),
        "mean_citations": round(sum(r["n_citations"] for r in rows) / max(len(rows), 1), 2),
    }
    (OUT / "hidden_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {answers}")
    print(f"wrote {OUT / 'hidden_summary.json'}")


if __name__ == "__main__":
    main()
