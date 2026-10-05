"""Aggregate benchmark runs into the comparison metrics and a dashboard.

Produces artifacts/metrics.json and dashboard/index.html.

The dashboard answers the question the hackathon actually poses - not "which
pipeline wins" but "where does the extra reasoning pay for itself". So every
view is per question type, and cost sits beside accuracy rather than in a
separate table. The headline number is tokens-per-correct-answer: a pipeline
that is 2% better for 5x the tokens has not earned it.

Run:  python scripts/06_metrics.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "runs" / "public"
ORDER = ["rag", "graphrag", "agentic"]
LABEL = {"rag": "RAG", "graphrag": "GraphRAG", "agentic": "Agentic GraphRAG"}
QTYPES = ["lookup", "multi_hop", "temporal", "aggregation", "superlative"]


def load(name: str) -> list[dict]:
    p = RUNS / f"results_{name}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {}
    correct = sum(1 for r in rows if r.get("correct"))
    tokens = sum(r.get("tokens", 0) for r in rows)
    out = {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "total_tokens": tokens,
        "mean_tokens": tokens / n,
        "mean_steps": statistics.mean(r.get("n_steps", 0) for r in rows),
        "mean_seconds": statistics.mean(r.get("seconds", 0) for r in rows),
        "mean_citations": statistics.mean(r.get("n_citations", 0) for r in rows),
        # The efficiency number that decides whether the complexity was worth it.
        "tokens_per_correct": tokens / correct if correct else None,
        "by_type": {},
    }
    recalls = [r["doc_recall"] for r in rows if "doc_recall" in r]
    if recalls:
        out["mean_doc_recall"] = statistics.mean(recalls)
        out["complete_doc_recall"] = statistics.mean(
            1.0 if r.get("doc_recall_complete") else 0.0 for r in rows if "doc_recall" in r
        )
    by = defaultdict(list)
    for r in rows:
        by[r.get("qtype", "")].append(r)
    for t, rs in by.items():
        c = sum(1 for r in rs if r.get("correct"))
        out["by_type"][t] = {
            "n": len(rs),
            "correct": c,
            "accuracy": c / len(rs),
            "mean_tokens": statistics.mean(r.get("tokens", 0) for r in rs),
            "mean_steps": statistics.mean(r.get("n_steps", 0) for r in rs),
        }
    return out


def agentic_behaviour(rows: list[dict]) -> dict:
    """Trace statistics the hackathon asks for explicitly."""
    if not rows:
        return {}
    tools = defaultdict(int)
    stops = defaultdict(int)
    steps_by_type = defaultdict(list)
    for r in rows:
        for t in r.get("tools_used", []):
            tools[t] += 1
        stops[r.get("stop_reason", "")] += 1
        steps_by_type[r.get("qtype", "")].append(r.get("n_steps", 0))
    return {
        "tool_usage": dict(sorted(tools.items(), key=lambda kv: -kv[1])),
        "stop_reasons": dict(sorted(stops.items(), key=lambda kv: -kv[1])),
        "mean_steps_by_type": {
            t: round(statistics.mean(v), 2) for t, v in sorted(steps_by_type.items())
        },
        "strategy_changes": sum(r.get("strategy_changes", 0) for r in rows),
    }


def render_html(metrics: dict, ceiling: dict) -> str:
    """A self-contained dashboard: no build step, no CDN, opens from disk."""
    live = [n for n in ORDER if metrics.get(n)]
    payload = json.dumps(
        {"metrics": metrics, "ceiling": ceiling, "order": live, "label": LABEL, "qtypes": QTYPES}
    )
    return (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agentic GraphRAG Benchmark</title>
<style>
:root{--bg:#fbfaf9;--fg:#1c1a17;--muted:#6b6560;--line:#e5e1dc;--card:#fff;
--rag:#b4654a;--graphrag:#4a7ab4;--agentic:#3f8f6a;--warn:#c0392b}
:root:not([data-theme="light"]){@media (prefers-color-scheme:dark){
:root{--bg:#17161a;--fg:#ece9e4;--muted:#9c958d;--line:#2e2b30;--card:#201e23}}}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#17161a;--fg:#ece9e4;--muted:#9c958d;--line:#2e2b30;--card:#201e23}}
:root[data-theme="dark"]{--bg:#17161a;--fg:#ece9e4;--muted:#9c958d;--line:#2e2b30;--card:#201e23}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1060px;margin:0 auto;padding:40px 16px 80px}
h1{font-size:28px;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:18px;margin:38px 0 12px;letter-spacing:-.01em}
.sub{color:var(--muted);margin:0 0 28px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
.card .name{font-size:13px;color:var(--muted);margin-bottom:8px}
.big{font-size:30px;font-weight:600;letter-spacing:-.02em}
.meta{font-size:13px;color:var(--muted);margin-top:6px}
table{width:100%;border-collapse:collapse;background:var(--card);
border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:14px}
th,td{padding:9px 12px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
thead th{background:transparent;color:var(--muted);font-weight:500;font-size:13px}
tbody tr:last-child td{border-bottom:none}
.bar{height:9px;border-radius:5px;background:var(--line);overflow:hidden;min-width:70px}
.bar>i{display:block;height:100%;border-radius:5px}
.zero{color:var(--warn);font-weight:600}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--agentic);
border-radius:8px;padding:14px 16px;margin:18px 0;font-size:14px}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}
</style></head><body><div class="wrap">
<h1>Agentic GraphRAG &mdash; benchmark</h1>
<p class="sub">Three pipelines, one corpus, 100 evaluation questions. Accuracy beside token cost.</p>
<div id="app"></div></div>
<script>const DATA = """
        + payload
        + """;
const C={rag:'var(--rag)',graphrag:'var(--graphrag)',agentic:'var(--agentic)'};
const pct=x=>(100*x).toFixed(0)+'%';
// Returns a fragment, not firstElementChild: most sections are an <h2> plus a
// <table>, and taking only the first element silently drops every table.
const el=(h)=>{const t=document.createElement('template');t.innerHTML=h.trim();return t.content};
const app=document.getElementById('app');
const M=DATA.metrics, order=DATA.order;

// headline cards
let cards='<div class="cards">';
for(const n of order){const m=M[n];
  cards+=`<div class="card"><div class="name">${DATA.label[n]}</div>
  <div class="big" style="color:${C[n]}">${pct(m.accuracy)}</div>
  <div class="meta">${m.correct}/${m.n} correct<br>${Math.round(m.mean_tokens)} tokens/question
  ${m.tokens_per_correct?`<br>${Math.round(m.tokens_per_correct)} tokens per correct answer`:''}</div></div>`;}
cards+='</div>';
app.appendChild(el(cards));

// accuracy by question type
let t='<h2>Accuracy by question type</h2><table><thead><tr><th>question type</th>';
for(const n of order) t+=`<th>${DATA.label[n]}</th>`;
t+='</tr></thead><tbody>';
for(const q of DATA.qtypes){
  const any=order.find(n=>M[n].by_type[q]); if(!any) continue;
  t+=`<tr><td>${q}</td>`;
  for(const n of order){const b=M[n].by_type[q];
    if(!b){t+='<td>&mdash;</td>';continue}
    const z=b.correct===0?' class="zero"':'';
    t+=`<td${z}>${b.correct}/${b.n} <span style="color:var(--muted)">(${pct(b.accuracy)})</span></td>`;}
  t+='</tr>';}
t+='</tbody></table>';
app.appendChild(el(t));

// cost
let c='<h2>Cost of being right</h2><table><thead><tr><th>pipeline</th><th>tokens / question</th>'
 +'<th>steps</th><th>tokens / correct answer</th></tr></thead><tbody>';
const maxT=Math.max(...order.map(n=>M[n].tokens_per_correct||0));
for(const n of order){const m=M[n];const w=maxT?100*(m.tokens_per_correct||0)/maxT:0;
  c+=`<tr><td>${DATA.label[n]}</td><td>${Math.round(m.mean_tokens)}</td>
  <td>${m.mean_steps.toFixed(1)}</td><td><div style="display:flex;gap:8px;align-items:center;justify-content:flex-end">
  <div class="bar" style="flex:1"><i style="width:${w}%;background:${C[n]}"></i></div>
  <span>${m.tokens_per_correct?Math.round(m.tokens_per_correct):'—'}</span></div></td></tr>`;}
c+='</tbody></table>';
app.appendChild(el(c));

// retrieval ceiling
if(DATA.ceiling&&DATA.ceiling.by_type){
 let r='<h2>Why top-k retrieval cannot close the gap</h2>'
  +'<div class="note">Measured with no LLM involved. <b>Complete</b> retrieval means every '
  +'supporting document reached the top-k. For a count or an arg-max, partial retrieval is '
  +'worth nothing: 14 of 15 events still yields the wrong number.</div>'
  +'<table><thead><tr><th>question type</th><th>supporting docs</th>'
  +'<th>complete @5</th><th>@20</th><th>@100</th></tr></thead><tbody>';
 for(const q of DATA.qtypes){const b=DATA.ceiling.by_type[q]; if(!b) continue;
  const z=v=>v===0?' class="zero"':'';
  r+=`<tr><td>${q}</td><td>${b.mean_gold_docs}</td>
  <td${z(b.complete_recall['5'])}>${pct(b.complete_recall['5'])}</td>
  <td>${pct(b.complete_recall['20'])}</td><td>${pct(b.complete_recall['100'])}</td></tr>`;}
 r+='</tbody></table>';
 app.appendChild(el(r));}

// agentic behaviour
const beh=M.agentic&&M.agentic.behaviour;
if(beh){
 let b='<h2>Agentic behaviour</h2><table><thead><tr><th>tool</th><th>questions used on</th>'
  +'</tr></thead><tbody>';
 for(const [k,v] of Object.entries(beh.tool_usage)) b+=`<tr><td><code>${k}</code></td><td>${v}</td></tr>`;
 b+='</tbody></table>';
 let s='<h2>Steps taken, by question type</h2><table><thead><tr><th>question type</th>'
  +'<th>mean steps</th></tr></thead><tbody>';
 for(const [k,v] of Object.entries(beh.mean_steps_by_type)) s+=`<tr><td>${k}</td><td>${v}</td></tr>`;
 s+='</tbody></table>';
 app.appendChild(el(b)); app.appendChild(el(s));
 let st='<h2>Why the agent stopped</h2><table><thead><tr><th>stop reason</th><th>questions</th>'
  +'</tr></thead><tbody>';
 for(const [k,v] of Object.entries(beh.stop_reasons)) st+=`<tr><td>${k||'—'}</td><td>${v}</td></tr>`;
 st+='</tbody></table>'; app.appendChild(el(st));}
</script></body></html>"""
    )


def main() -> None:
    data = {name: load(name) for name in ORDER}
    metrics = {name: summarise(rows) for name, rows in data.items() if rows}
    if data.get("agentic"):
        metrics.setdefault("agentic", {})["behaviour"] = agentic_behaviour(data["agentic"])

    ceiling_path = ROOT / "artifacts" / "retrieval_ceiling.json"
    ceiling = json.loads(ceiling_path.read_text(encoding="utf-8")) if ceiling_path.exists() else {}

    out = {"pipelines": metrics, "retrieval_ceiling": ceiling}
    (ROOT / "artifacts" / "metrics.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    # ---- console report ----
    print(f"{'pipeline':<20}{'acc':>8}{'tokens/q':>11}{'steps':>8}{'tok/correct':>13}")
    for name in ORDER:
        m = metrics.get(name)
        if not m:
            continue
        tpc = m["tokens_per_correct"]
        print(
            f"{LABEL[name]:<20}{100*m['accuracy']:>7.0f}%{m['mean_tokens']:>11.0f}"
            f"{m['mean_steps']:>8.1f}{(f'{tpc:.0f}' if tpc else '-'):>13}"
        )

    live = [n for n in ORDER if metrics.get(n)]
    print(f"\n{'accuracy by type':<20}" + "".join(f"{LABEL[n][:12]:>14}" for n in live))
    for t in QTYPES:
        row = f"{t:<20}"
        for n in live:
            bt = metrics[n]["by_type"].get(t)
            cell = f"{bt['correct']}/{bt['n']}" if bt else "-"
            row += f"{cell:>14}"
        print(row)

    html = render_html(metrics, ceiling)
    (ROOT / "dashboard").mkdir(exist_ok=True)
    (ROOT / "dashboard" / "index.html").write_text(html, encoding="utf-8")
    print(f"\nwrote artifacts/metrics.json and dashboard/index.html")


if __name__ == "__main__":
    main()
