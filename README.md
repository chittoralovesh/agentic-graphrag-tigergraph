# Agentic GraphRAG on TigerGraph

**When is agentic reasoning worth its token cost, and when is it overkill?**

Three pipelines answer the same 100 questions over the same corpus — classical
RAG, GraphRAG with a fixed plan, and an Agentic GraphRAG that plans its own
investigation. The interesting result is not that the agent wins. It is
*where* it wins, *where it does not*, and the measured reason why.

---

## The headline

Before any LLM is involved, we measured what top-k retrieval can physically
reach. **Complete** retrieval means every supporting document landed in the
top-k — the only standard that matters for a count, since retrieving 14 of 15
events still produces the wrong number.

| question type | supporting docs | complete @5 | @20 | @100 |
|---|---|---|---|---|
| lookup | 1.0 | 100% | 100% | 100% |
| multi_hop | 1.0 | 89% | 96% | 100% |
| temporal | 2.0 | 59% | 91% | 100% |
| aggregation | 15.1 | **0%** | 24% | 86% |
| superlative | 13.8 | **0%** | 10% | 80% |

At k=5, aggregation and superlative reach **zero**. Those 31 questions are
unanswerable by top-k retrieval in principle, however good the generator is.

That measurement *predicted* the accuracy before we ran it:

| question type | predicted complete recall @5 | actual RAG accuracy |
|---|---|---|
| lookup | 100% | 17/19 (89%) |
| multi_hop | 89% | 23/28 (82%) |
| temporal | 59% | 16/22 (73%) |
| superlative | **0%** | **2/10 (20%)** |
| aggregation | **0%** | **0/21 (0%)** |

This is the project's central claim: the gap is **structural, not a tuning
problem**. No amount of prompt engineering or reranking fixes a retriever that
cannot fit the evidence set into its context window.

---

## Architecture

```
                         ┌─────────────────────────────────┐
                         │   Benchmark harness             │
                         │  150 Q x 3 pipelines -> traces  │
                         └───────────────┬─────────────────┘
          ┌──────────────────────────────┼──────────────────────────────┐
          v                              v                              v
  ┌───────────────┐            ┌──────────────────┐         ┌──────────────────────┐
  │ P1: RAG       │            │ P2: GraphRAG     │         │ P3: Agentic GraphRAG │
  │ vector/BM25   │            │ classify -> link │         │  ORCHESTRATOR LOOP   │
  │ top-k -> LLM  │            │ -> retrieve      │         │ plan -> act -> judge │
  │ single shot   │            │ fixed 4 stages   │         │ -> re-plan -> stop   │
  └───────┬───────┘            └────────┬─────────┘         └──────────┬───────────┘
          │                             │                              │
          │                             │              ┌───────────────┴────────────────┐
          │                             │              │        Specialist tools        │
          │                             │              │ entity_link · graph_traverse   │
          │                             │              │ vector_search · doc_fetch      │
          │                             │              │ aggregate(count/argmax)        │
          │                             │              │ temporal_hop · evidence_critic │
          │                             │              └───────────────┬────────────────┘
          └─────────────────────────────┴──────────────────────────────┘
                                        v
                    ┌───────────────────────────────────────────┐
                    │  TigerGraph 4.2.5 (Savanna)               │
                    │  Games ─HAS_EVENT→ OlympicEvent           │
                    │     OlympicEvent ─AT_VENUE→ Venue         │
                    │     OlympicEvent ─OF_SPORT→ Sport         │
                    │     OlympicEvent ─WON_GOLD→ Athlete →NOC  │
                    │     OlympicEvent ⇄ PREV/NEXT_EDITION      │  temporal spine
                    │     OlympicEvent ─SOURCED_FROM→ Document  │  citations
                    │     Document ─HAS_CHUNK→ Chunk[vector]    │  TigerVector HNSW
                    └───────────────────────────────────────────┘
```

### Why the middle pipeline exists

GraphRAG has **the same graph access** as the agentic pipeline but a fixed
control flow. Without it, "agentic beats RAG" would conflate two separate
causes — having a graph, and being adaptive. The three-way split attributes
each gain to the right mechanism.

---

## The dataset, as it actually is

The corpus is 2,951 Wikipedia documents, each led by a machine-readable
infobox. **2,162 are Olympic events; the other 789 — 546 films, plus
politicians, companies and scientists — are distractors.** Every one of the 547
gold documents across the public questions is an Olympic event.

```
[Infobox Olympic event]
  event: Men's canoe sprint K-2 1,000 metres
  games: 2012 Summer      venue: Eton Dorney      date: 6 to 8 August
  competitors: 24         nations: 12
  gold: Rudolf DombiRoland Kokeny                 goldNOC: HUN
  prev: 2008              next: 2016
```

The distractors are indexed too, and infoboxes are kept inside the chunks. A
baseline that never has to discriminate is not a baseline.

Five question types reduce to five query shapes over those fields: attribute
read, filtered count, arg-max, venue+date resolution, and an edition walk along
`prev`/`next`.

---

## Results

Run on the 100 public questions.

| pipeline | accuracy | tokens / question | tokens per correct answer |
|---|---|---|---|
| RAG | 58/100 | 2,002 | 3,452 |
| GraphRAG | 85/100 | 1,467 | 1,726 |
| Agentic GraphRAG | see `artifacts/metrics.json` | | |

GraphRAG is both **more accurate and cheaper** than RAG: structured retrieval
returns a small exact answer where passage stuffing returns 5 chunks of prose.

The open dashboard (`dashboard/index.html`) carries accuracy by type, cost per
correct answer, the retrieval-ceiling table, tool usage, steps per question
type, and why the agent stopped.

### Retrieval ceiling vs. pipelines

`scripts/02_oracle_check.py` routes questions straight to the graph tools with
no LLM, and scores **100/100 with 100% gold-document recall**. That is the
ceiling. It separates blame cleanly: a pipeline that misses a question the
oracle answers has an orchestration bug, not a data problem. Every accuracy fix
in this repo was found that way.

---

## Efficiency, deliberately

Burning ten steps on a lookup is a failure even when the answer is right, so:

- the agent **skips the evidence-critic call** after any tool that scans a
  complete group (`count_events`, `argmax_event`) or reads a resolved
  attribute — those results are self-validating, and reviewing them spends a
  call to confirm what the tool's own semantics already guarantee
- aggregation and superlative questions settle in **3 steps**; multi-hop and
  temporal, which need disambiguation, take longer
- answers are graded by exact match first, and the LLM judge is consulted only
  when exact match fails, so grading cost tracks disagreement rather than
  dataset size

---

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env     # add GEMINI_API_KEY and TigerGraph credentials
```

```bash
python scripts/01_build_graph.py        # corpus -> property graph
python scripts/02_oracle_check.py       # retrieval ceiling (no LLM)
python scripts/03_retrieval_ceiling.py  # what top-k can reach
python scripts/04_embed_chunks.py       # chunk embeddings
python scripts/07_tg_schema.py          # TigerGraph schema + TigerVector
python scripts/08_tg_load.py            # load into TigerGraph
python scripts/05_run_benchmark.py      # all three pipelines
python scripts/06_metrics.py            # metrics.json + dashboard
```

The corpus is not in the repo (23 MB); see `data/README.md`.

---

## Layout

```
src/agraph/
  corpus.py          infobox parsing, date spans, name splitting
  graphmodel.py      typed graph projection + edition linking
  chunking.py        passage chunks (infobox kept in chunk 0)
  llm.py             Gemini client, token accounting, disk cache
  trace.py           per-step tokens / latency / tools / citations
  tools/
    structured.py    count, arg-max, venue+date, edition walk
    vector.py        dense, BM25, RRF hybrid, TigerVector
  pipelines/
    rag.py           single-shot retrieval
    graphrag.py      fixed four-stage plan
    agentic.py       orchestrator loop + evidence critic
  store/
    tigergraph.py    connection, schema, loading
  eval/judge.py      exact match, then LLM judge
scripts/             numbered, run in order
dashboard/           self-contained metrics dashboard
```

---

## Honest limitations

- The vector lane runs on **BM25** in the reported numbers. Gemini's free
  embedding tier is 1,000 requests/day, and `gemini-embedding-2` silently
  returns one vector per request regardless of batch size, so dense embeddings
  for 16,528 chunks did not fit the budget. Dense and RRF-hybrid backends are
  implemented and swap in behind the same interface.
- `prev`/`next` infobox hints resolve to edges for 77% of events; the rest fall
  back to a nearest-earlier-year lookup on the same discipline key.
- Two public questions are genuinely ambiguous from venue+date alone — two
  events really did share a venue and a day. The tie is broken by matching the
  sport against the venue name.
- The judge is an LLM for the cases exact match rejects; it is not infallible,
  and its verdicts are recorded per question in the results files.
