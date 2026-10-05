# Demo video script (3–5 minutes)

Record at 1080p. Terminal font large enough to read on a phone. Have the
dashboard open in a browser tab and the TigerGraph Savanna GraphStudio page in
another before you start.

The through-line: **we did not set out to prove the agent wins. We set out to
find where it does, and we measured the mechanism.**

---

## 0:00–0:30 — The question

> "RAG retrieves text. GraphRAG adds structure. Agentic GraphRAG plans its own
> investigation. The question this hackathon asks is not which is best — it's
> when the extra reasoning is worth its token cost. We answered that with a
> measurement, not a leaderboard."

Show: the README's opening table on screen.

---

## 0:30–1:15 — The dataset is not what it looks like

> "The corpus is 2,951 Wikipedia documents. 2,162 are Olympic events. The other
> 789 — mostly films — are distractors. Every document leads with a
> machine-readable infobox."

Run:

```bash
head -c 700 data/corpus/corpus.jsonl
```

> "That infobox is the whole game. Venue, date, competitors, nations,
> medallists, and `prev`/`next` pointers to the same event at the previous and
> next Games. That last pair becomes a temporal spine in the graph."

---

## 1:15–2:15 — The finding, measured without an LLM

```bash
python scripts/03_retrieval_ceiling.py
```

> "Before involving any language model, we measured what top-k retrieval can
> physically reach. Complete retrieval means *every* supporting document made
> the top-k — the only standard that matters for a count, because retrieving 14
> of 15 events still gives the wrong number."

Point at the zeros:

> "At k=5, aggregation and superlative are at zero percent. Not poor — zero.
> Those 31 questions are unanswerable by top-k retrieval in principle, however
> good the generator is. Aggregation questions need 15 supporting documents on
> average. Superlative, 14."

Then:

> "And this predicted the accuracy before we ran it. RAG scored 0 out of 21 on
> aggregation. The ceiling said it would."

---

## 2:15–3:00 — The graph, live on TigerGraph

Switch to Savanna / GraphStudio.

> "Everything runs on TigerGraph 4.2.5 on Savanna. 2,162 event vertices,
> 5,981 athletes, 11 edge types, and 16,528 chunks behind a TigerVector HNSW
> index."

Show the schema, then run a structured query against it.

> "An aggregation question becomes one grouped scan over the whole set —
> exactly the operation top-k cannot do."

---

## 3:00–4:00 — The agent choosing its own path

```bash
python scripts/05_run_benchmark.py --limit 3 --pipelines agentic
```

> "The orchestrator picks one action at a time, looks at what came back, and
> decides whether to answer or dig further. Nothing here is a fixed sequence."

Show a trace:

> "This aggregation question settles in three steps — the orchestrator picks
> the grouped count, and the agent stops immediately, because a tool that
> scanned the entire group is self-validating. Asking a critic 'is that enough?'
> would spend a call to confirm what the tool already guarantees."

> "A venue-and-date question takes longer: the first lookup is ambiguous, the
> critic says so, and the agent re-plans. That's where being agentic actually
> earns its keep."

---

## 4:00–4:45 — Results

Switch to `dashboard/index.html`.

> "Three pipelines, same questions, same synthesis step — the only variable is
> how evidence is gathered."

Walk the table, then land the honest point:

> "GraphRAG is both more accurate and cheaper than RAG — structured retrieval
> returns a small exact answer where passage-stuffing returns five chunks of
> prose. And on lookup questions, plain RAG is already at ninety percent. The
> agent adds nothing there and we don't pretend otherwise."

> "The agent earns its cost on the questions where the first retrieval can be
> wrong — ambiguous venues, edition walks — and it is the wrong tool for a
> question a single lookup already answers."

---

## 4:45–5:00 — Close

> "The answer to 'when does agentic reasoning help' is: when the evidence set is
> larger than a context window, or when the first retrieval can be wrong. We
> measured both, and we can tell you which questions are which."

---

## Before you record

```bash
python scripts/01_build_graph.py
python scripts/02_oracle_check.py
python scripts/06_metrics.py
```

Have ready: a clean terminal, the dashboard, Savanna open and logged in.
Record the ceiling study and the agent trace live — those two land hardest.
