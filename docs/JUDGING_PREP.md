# 20-minute judging round — prep

Assume roughly 6–8 minutes of you talking and 12–14 minutes of questions. The
questions are where this is won. Judges have seen a dozen teams claim "agentic
is better"; almost nobody will have *measured why*, and nobody will volunteer
where their agent loses. You can do both.

---

## Numbers to know cold

| pipeline | accuracy | tokens/q | latency | steps |
|---|---|---|---|---|
| RAG | 58/100 | 2,002 | 2.3s | 2.0 |
| GraphRAG | **100/100** | **832** | 4.0s | 3.7 |
| Agentic GraphRAG | **100/100** | 1,878 | 15.7s (median 8.3) | 5.1 |

| question type | RAG | GraphRAG | Agentic |
|---|---|---|---|
| lookup | 17/19 | 19/19 | 19/19 |
| multi_hop | 23/28 | 28/28 | 28/28 |
| temporal | 16/22 | 22/22 | 22/22 |
| aggregation | **0/21** | 21/21 | 21/21 |
| superlative | 2/10 | 10/10 | 10/10 |

**Retrieval ceiling (no LLM involved) — complete gold-set retrieval:**

| qtype | docs needed | k=5 | k=20 | k=100 |
|---|---|---|---|---|
| aggregation | 15.1 | **0%** | 24% | 86% |
| superlative | 13.8 | **0%** | 10% | 80% |
| lookup | 1.0 | 100% | 100% | 100% |

Hidden set: 50/50 answered, 2,094 tokens/q, 4.64 steps, 7.98 citations.

TigerGraph: 2,951 Documents · 2,162 OlympicEvents · 5,981 Athletes · 16,528
Chunks · 11 edge types · 1,665 PREV_EDITION edges · TigerVector HNSW/cosine.

---

## The opening (60 seconds — memorise the shape, not the words)

> "The brief asked when agentic reasoning is worth its token cost. Most answers
> to that are a leaderboard. Ours is a measurement.
>
> Before we ran a single LLM call, we measured what top-k retrieval can
> physically reach — the fraction of questions where *every* supporting
> document fits in the context. For a count, partial retrieval is worthless:
> retrieve 14 of 15 events and your number is still wrong.
>
> At k=5, aggregation and superlative questions sit at zero percent. Thirty-one
> of a hundred questions are unanswerable by RAG in principle, however good the
> generator is.
>
> That measurement predicted the result. RAG scored 0 out of 21 on aggregation,
> exactly as forecast — and its overall 58% sits against a measured ceiling of
> 57%, which means the generator was already perfect and the entire gap was
> retrieval.
>
> Then the uncomfortable part: our agent reaches 100%, but so does our
> non-agentic GraphRAG, at 2.3× less cost. So I can tell you precisely where
> the agent earns its keep, and where it doesn't."

That last line is your differentiator. Say it early and deliberately.

---

## The three things to get across

**1. We separated retrieval failure from generation failure.**
Everyone else reports end-to-end accuracy, which confounds the two. We measured
retrieval alone, with no LLM, and it *predicted* the accuracy. That's the
difference between "our system scored higher" and "here is the mechanism."

**2. We built a control that isolates the right cause.**
GraphRAG (pipeline 2) has identical graph access to the agent but a fixed plan.
RAG → GraphRAG measures the value of *structure*. GraphRAG → Agentic measures
the value of *adaptivity*. Without that middle term, "agentic wins" conflates
two different mechanisms, and nearly every team will have conflated them.

**3. We built an oracle that bounds the system.**
Routing questions straight to the graph tools, with no LLM, scores 100/100 with
100% document recall. So any pipeline miss is provably an *orchestration* bug,
not a data limitation. Every accuracy fix we made was found that way. It is the
single most useful artefact in the repo.

---

## Hard questions, with honest answers

### "Your agent is no more accurate than GraphRAG and costs 2.3× more. Why build it?"

This is *the* question. Do not get defensive — you have a real answer.

> "On this benchmark, it isn't worth it, and we say so in the writeup. But the
> comparison flatters the fixed plan for a reason worth examining: GraphRAG only
> reaches 100 after three rounds of hand-written, per-question-shape parameter
> repair that we wrote *after* reading its failures. The agent reached 21/21 on
> aggregation with none of them — it corrected its own arguments from the
> evidence.
>
> One concrete case: the router split '2018 Winter Olympics' into year and
> season with no games field, selected an empty group, and confidently answered
> zero instead of five. We had to hand-write a repair for that. The agent just
> passed the right argument.
>
> So the trade isn't accuracy against tokens. It's tokens against the cost of
> enumerating your question shapes in advance. This dataset has five templates,
> so enumeration is cheap and the fixed plan wins. On an open domain where you
> can't enumerate them, that 2.3× is what you're buying."

### "Did you use the same LLM across all three pipelines?"

Be straight. This is the weakest methodological point and they may probe it.

> "Primarily gemini-flash-lite-latest. Free-tier daily caps are per model —
> some as low as 20 requests a day — so we built automatic rotation across four
> flash models and three API keys to finish the runs. That means parts of the
> comparison were served by different models in the flash family, which is a
> confound we'd remove with a paid key. What it does *not* explain is the
> headline result: RAG's gap is retrieval, not generation, and we proved that
> without any LLM in the loop."

That last sentence is important — it shows the confound can't touch your
central claim.

### "You used BM25, not vectors. Isn't this a vector/graph hackathon?"

> "The graph work is all TigerGraph — 2,162 event vertices, 11 edge types, a
> temporal spine of 1,665 PREV_EDITION edges. TigerVector is provisioned with an
> HNSW cosine index and populated at document granularity.
>
> The reported vector lane runs on BM25 because free-tier embedding quota is
> 1,000 requests a day and we had 16,528 chunks — and because one Gemini
> embedding model silently returns a single vector per request regardless of
> batch size, which we only caught because a batch of 64 wrote exactly 16 rows.
> Dense and RRF-hybrid backends are implemented behind the same interface and
> swap in with a flag.
>
> It also doesn't change the conclusion: the ceiling study is a property of
> top-k retrieval, not of which retriever. A perfect dense retriever still
> cannot fit 15 documents into k=5."

### "How do we know you didn't overfit to the public questions?"

> "The hidden 50 — we answered all of them, 2,094 tokens each, with full traces
> submitted. We also never tuned against answers: the fixes we made were found
> by the oracle, which compares tool output to the graph, not to gold answers.
> And the dataset's own structure is the check — the five question types are
> templates, and we solve them by shape, not by memorising instances."

### "Your hand-written repairs look like overfitting to the benchmark."

> "Fair challenge, and it's why the agent result matters. The repairs are all
> one principle: the model chooses the *shape* of a retrieval; it must not
> retype a *literal* that has to match stored data. It respaced
> 'TechnologyUniversity', dropped ', Barcelona' from a venue, and truncated the
> compound date '21 September 2000 (slow)22 September 2000 (fast)' to its first
> half. Each edit resolved a real but *different* event — a confidently wrong
> answer, which is worse than a failure.
>
> That generalises well beyond Olympics data. Any agent calling a tool against
> a keyed store has this bug."

### "Why is RAG so bad — did you weaken it?"

> "We deliberately didn't. Infoboxes are inside the indexed chunks, so the
> numeric facts are retrievable. All 2,951 documents are indexed including 789
> distractors — mostly films — so retrieval has to discriminate. And we swept k:
> at k=100, twenty-one times the context budget and 31,666 tokens per question,
> RAG's ceiling is still 94% and aggregation still tops out at 81%. GraphRAG is
> perfect on 832 tokens — roughly 38× cheaper."

### "15.7 seconds per question is slow."

> "Median is 8.3. The spread is the interesting part: 4.5 seconds when the first
> tool call settles the question, 41 seconds when the agent has to re-plan.
> Most of that is free-tier API latency and a 10 requests-per-minute self-imposed
> rate limit, not computation — our graph tools answer in milliseconds. The
> oracle runs all 100 questions with no LLM at all."

### "Round 2 is reasoning over evolving and conflicting facts. How would you extend?"

> "The temporal spine is already there — PREV_EDITION / NEXT_EDITION edges make
> 'the Games immediately before 2016' a single hop instead of a search. For
> conflicting facts we'd add valid_from / valid_to on the edges, a
> source-authority weight on Document, and a supersession edge. The evidence
> critic then changes job: today it asks 'is this sufficient?', and it would ask
> 'which of these conflicting claims supersedes the others, and on what
> authority?' That's a prompt and a scoring function on top of the same harness,
> not a rebuild."

### "What would you do differently?"

> "Three things. Push the structured tools down into GSQL so aggregation runs
> inside TigerGraph instead of the client. Use a paid key so the model is held
> fixed across pipelines. And record the model on every trace row — we made the
> cache model-independent to save quota, which cost us the ability to
> reconstruct which model answered which question."

Admitting that last one unprompted buys a lot of credibility.

---

## Demo checklist — do this 15 minutes before the call

1. **Resume the Savanna workspace** at tgcloud.io. It auto-suspends when idle and
   every endpoint returns 500 until it wakes. *This is the single most likely
   thing to embarrass you live.* Verify:
   ```
   python -c "import sys;sys.path.insert(0,'src');from agraph.llm import load_env;load_env();from agraph.store.tigergraph import connect;c=connect(graphname='OlympicGraphRAG',verify=False);c.graphname='OlympicGraphRAG';print(c.echo(), c.getVertexCount('OlympicEvent'))"
   ```
2. Open `dashboard/index.html` in a browser tab.
3. Open `docs/architecture.svg` in another tab.
4. Have a terminal ready in the project directory, font large.
5. Pre-run these so nothing is cold:
   ```
   python scripts/02_oracle_check.py
   python scripts/03_retrieval_ceiling.py
   ```

**If you demo one thing live, demo the ceiling study.** It takes 40 seconds, uses
no API, cannot fail on quota, and it is the most persuasive artefact you have.

Do **not** demo a live agentic run unless quota is confirmed — a 429 mid-demo is
a bad look. Show a saved trace from `artifacts/runs/public/traces_agentic.jsonl`
instead.

---

## Things not to say

- Don't claim the agent beats GraphRAG. It doesn't, and the data is in your repo.
- Don't claim full chunk-level TigerVector population. It's document-level.
- Don't claim a single model across all pipelines without the rotation caveat.
- Don't say "we got 100%" as the headline. Two pipelines at 100% is the *least*
  interesting thing you found. Lead with why RAG structurally can't.

---

## If you only remember one line

> "We can tell you which questions need an agent, which don't, and what each one
> costs — because we measured the retrieval ceiling before we wrote a single
> prompt."
