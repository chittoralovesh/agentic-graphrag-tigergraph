# When does agentic reasoning actually help?

**Agentic GraphRAG Hackathon — TigerGraph**

---

## The claim

Agentic retrieval is not generally better. It is better on a **specific,
identifiable class of question**, and that class can be predicted in advance
from a property of the question rather than discovered by trial.

The property is the **size of the evidence set**:

- When the answer depends on **one document**, a single retrieval is enough and
  anything more is wasted tokens.
- When the answer depends on **the whole group** — a count, a maximum —
  top-k retrieval cannot answer it at any k that fits a context window, and
  structure is required.
- When the **first retrieval can be wrong** — an ambiguous venue, an edition
  that must be walked to — adaptivity is what recovers it, and that is the only
  place the agent loop genuinely earns its cost.

---

## How we established it

### 1. Measure retrieval before involving a language model

The usual way to compare these pipelines is end-to-end accuracy, which
confounds retrieval with generation. We separated them.

`scripts/03_retrieval_ceiling.py` asks only: did the documents needed to answer
appear in the top-k at all? Two metrics, and the gap between them is the point:

- **partial recall** — the fraction of supporting documents retrieved
- **complete recall** — the fraction of questions where *every* supporting
  document was retrieved

For a count, partial recall is worthless. Retrieving 14 of 15 events still
yields the wrong number. Complete recall is therefore the honest predictor of
achievable accuracy.

| question type | supporting docs | complete @5 | @20 | @100 |
|---|---|---|---|---|
| lookup | 1.0 | 100% | 100% | 100% |
| multi_hop | 1.0 | 89% | 96% | 100% |
| temporal | 2.0 | 59% | 91% | 100% |
| aggregation | 15.1 | **0%** | 24% | 86% |
| superlative | 13.8 | **0%** | 10% | 80% |

At k=5, aggregation and superlative reach **zero percent**. 31 of 100 questions
are unanswerable by top-k retrieval in principle.

### 2. The ceiling predicted the accuracy

| question type | complete recall @5 | RAG accuracy |
|---|---|---|
| lookup | 100% | 17/19 (89%) |
| multi_hop | 89% | 23/28 (82%) |
| temporal | 59% | 16/22 (73%) |
| superlative | **0%** | **2/10 (20%)** |
| aggregation | **0%** | **0/21 (0%)** |

RAG scored zero on aggregation. The ceiling said it would, before the pipeline
was run. This is the difference between "our system scored higher" and "here is
the mechanism, measured independently, and here is the accuracy it forecasts."

### 3. Separate "has a graph" from "is adaptive"

GraphRAG (pipeline 2) has **the same graph access** as the agentic pipeline but
a fixed four-stage plan. Without that middle term, any win would conflate two
different causes. The three-way split attributes each gain to the right
mechanism:

- RAG → GraphRAG isolates the value of **structure**
- GraphRAG → Agentic isolates the value of **adaptivity**

### 4. Establish a ceiling to separate retrieval bugs from reasoning bugs

`scripts/02_oracle_check.py` routes questions directly to the graph tools with
no LLM, and scores **100/100 with 100% gold-document recall**.

This turned out to be the single most useful artefact in the project. Any
question the oracle answers but a pipeline misses is an *orchestration* fault,
not a data fault. Every accuracy fix below was found that way.

---

## Results

100 public questions. Same corpus, same synthesis step; the only variable is
how evidence is gathered.

| pipeline | accuracy | tokens / question | tokens per correct answer |
|---|---|---|---|
| RAG | 58/100 | 2,002 | 3,452 |
| **GraphRAG** | **100/100** | **832** | **832** |
| **Agentic GraphRAG** | **100/100** | 1,878 | 1,878 |

| question type | RAG | GraphRAG | Agentic |
|---|---|---|---|
| lookup | 17/19 | 19/19 | 19/19 |
| multi_hop | 23/28 | 28/28 | 28/28 |
| temporal | 16/22 | 22/22 | 22/22 |
| aggregation | 0/21 | 21/21 | 21/21 |
| superlative | 2/10 | 10/10 | 10/10 |

Two results here are worth stating plainly, because neither is the convenient
one.

**GraphRAG is more accurate *and* cheaper than RAG.** Structured retrieval
returns a small exact answer where passage-stuffing returns five chunks of
prose, so the better pipeline is also the thriftier one. "Agentic costs more"
is not a law; it depends entirely on what retrieval returns.

**The agent matches the fixed plan's accuracy and costs 2.3x more to do it.**
Both reach 100/100; GraphRAG spends 832 tokens per question and the agent
1,878. On a benchmark whose question shapes can be enumerated in advance,
adaptivity buys no accuracy and is pure overhead.

An earlier agent run scored 91/100, and the gap was instructive: all three of
its genuine misses came from the orchestrator paraphrasing a literal — it
respaced a venue, truncated the compound date "21 September 2000 (slow)22
September 2000 (fast)" to its first half, and shortened a discipline to
"Greco-Roman". One of those, an ambiguous venue, cost **13 steps and 29,310
tokens** before giving up. Restoring the literals from the question text closed
all three.

### Was RAG simply starved of context?

The obvious objection to a 58% baseline is that k=5 was too small. So we swept
it, measuring for each k the fraction of questions whose **complete** supporting
set fits in the top-k — an upper bound on accuracy, since no generator can be
right about a count whose evidence it never saw.

| k | ceiling | context tokens / question | aggregation | superlative |
|---|---|---|---|---|
| 5 | 57% | 1,515 | 0% | 0% |
| 10 | 64% | 3,064 | 5% | 0% |
| 20 | 72% | 6,212 | 24% | 10% |
| 50 | 87% | 15,637 | 57% | 60% |
| 100 | 94% | 31,666 | 81% | 80% |

Two things fall out of this table.

**RAG was not starved — it was already at its ceiling.** Measured accuracy at
k=5 was **58%** against a retrieval ceiling of **57%**. The generator is
performing perfectly; every point of the gap to 100 is retrieval, not
reasoning. That is as clean a separation of the two failure modes as this
dataset allows.

**Buying more context does not close the gap.** At k=100 — **21x the context
budget, 31,666 tokens per question** — the ceiling is still 94%, and
aggregation still tops out at 81%. GraphRAG reaches **100% on 832 total tokens
per question**, roughly **38x cheaper** than RAG at k=100 and more accurate,
because a grouped scan returns one exact number instead of fifteen documents to
read.

This is the quantitative form of the project's claim. The failure is not a
tuning problem that a bigger window fixes; it is structural.

### But the comparison is not as flattering to the fixed plan as it looks

GraphRAG only reaches 100/100 **after three rounds of hand-written, per-shape
parameter repair** — the fixes catalogued below. The agent reached 21/21 on
aggregation *with none of them*:

```
pub-001  agent: count_events(sport="biathlon", games="2018 Winter Olympics", ...) -> 5   ✓ 3 steps
pub-001  fixed plan: {"sport":"Biathlon","year":2018,"season":"Winter"}  -> count 0      ✗
```

The router split the Games into `year` and `season` with no `games` field,
selecting an empty group and counting zero. That cost the fixed pipeline two
questions until we wrote a deterministic repair for it. The orchestrator simply
passed the right argument.

So the honest conclusion is three-tier:

1. **Structure** is where the enormous win is: 58 → 100, at lower cost.
2. **Adaptivity** did not add accuracy over a fixed plan we had already
   engineered against this exact question distribution, and cost 2.5x.
3. **Adaptivity substitutes for that engineering.** The agent approached the
   same accuracy without bespoke per-shape repairs, because it corrects its own
   parameters from the evidence instead of needing a human to anticipate each
   failure.

Which means the real trade is not accuracy against tokens. It is **tokens
against the cost of enumerating your question shapes in advance.** On a
templated benchmark you can enumerate them, and the fixed plan wins. On an open
domain you cannot, and that is what the agent's 2.5x buys you.

---

## What we got wrong, and what it taught us

Most of our accuracy came from fixing bugs, not from better prompting. The
oracle made each one findable.

**1. Silent data corruption in the parser.** `event_suffix` split titles on
ASCII hyphens as well as the en-dash separator, so `"Men's cross-country"`
became `"country"` and `"Greco-Roman 96 kg"` became `"roman 96 kg"`. Separately,
normalisation stripped `+`, collapsing `"Men's +80 kg"` and `"Men's 80 kg"` into
one key — two different events with two different gold medallists. Both capped
every pipeline's accuracy invisibly.

**2. The router reads the right shape but fills the wrong fields.** Asked who
won the *gold medal*, it requested `win_value` — the winning *time*. That reads
a real field and returns a confidently wrong answer, which is worse than
failing. It also put an edition's discipline under `event_title` while the code
read `discipline`.

**3. The model paraphrases strings that must match the corpus byte-for-byte.**
It respaced `"TechnologyUniversity"` and dropped `", Barcelona"` from venue
names. Exact lookup then missed, same-day siblings tied, and the wrong athlete
was returned. It also split `"2018 Winter Olympics"` into `year` and `season`
with no `games` field, which selected an empty group and counted **0** instead
of 5.

The general lesson, and the one we would carry to any agentic system:

> **Let the model choose the shape of the retrieval. Do not let it retype the
> literals.** Anything that must match stored data exactly should be extracted
> deterministically from the question. An LLM's instinct to tidy a string is a
> silent correctness bug when that string is a key.

**4. A correct answer was discarded by a network blip.** An aggregation
returned the right count, then synthesis hit a connection reset and the answer
was lost. Exact results from the graph are now retained if synthesis fails.

---

## Efficiency as a design goal

Spending ten steps on a lookup is a failure even when the answer is right.

- The agent **skips the evidence-critic call** after any tool that scans a
  complete group (`count_events`, `argmax_event`) or reads a resolved
  attribute. Those results are self-validating; asking "is this enough?" spends
  a call to confirm what the tool's semantics already guarantee. This roughly
  halved the agent's LLM calls.
- Grading uses exact match first and consults the LLM judge only when exact
  match fails, so evaluation cost tracks disagreement rather than dataset size.
- Answers are cached on disk keyed by prompt, so re-runs cost nothing.

---

## Engineering notes

Everything runs on **TigerGraph 4.2.5 (Savanna)**: 2,951 Document, 2,162
OlympicEvent, 5,981 Athlete, 298 Venue, 133 NOC, 16,528 Chunk vertices and 11
edge types, with `Chunk.embedding` as a TigerVector HNSW/COSINE attribute.

The `PREV_EDITION` / `NEXT_EDITION` spine is recovered from the infobox's
`prev`/`next` year hints, which resolve for 77% of events; the rest fall back
to the nearest earlier year on the same discipline key. That spine turns "the
Games held immediately before 2016" into a single hop instead of a search.

**Free-tier quotas shaped the engineering more than anything else.** Each Gemini
model carries its own daily cap — 20/day on `gemini-3.8-flash`, 500/day on
`flash-lite`, 1,000/day for embeddings — and `gemini-embedding-2` silently
returns one vector per request regardless of batch size, discarding the rest. A
spent cap stalled runs in backoff against a counter that only resets the next
day, so the client now retires an exhausted model and continues on the next,
recording which model served each call.

---

## Limitations

- The reported numbers use **BM25** for the vector lane. Dense and RRF-hybrid
  backends are implemented behind the same interface, but embedding 16,528
  chunks did not fit the free daily quota.
- Two public questions are genuinely ambiguous from venue and date alone — two
  events really did share a venue and a day. The tie is broken by matching the
  sport against the venue name, which is a heuristic, not a guarantee.
- The LLM judge handles cases exact match rejects. It is not infallible; its
  verdict is recorded per question so any disagreement is auditable.

---

## What we would do next

- Push the structured tools down into **GSQL** so aggregation runs inside
  TigerGraph rather than in the client, and measure the latency difference.
- **Round 2 (reasoning over time):** the `PREV_EDITION` spine is already a
  temporal backbone. Conflicting facts would need `valid_from`/`valid_to` on
  the edges, a source-authority weight on `Document`, and a supersession edge —
  then the evidence critic becomes a conflict resolver rather than a
  sufficiency check.
- Let the orchestrator **learn its stopping rule** from the trace corpus
  instead of following a prompt instruction.
