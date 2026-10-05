"""Pipeline 1 - classical RAG.

Retrieve once by similarity, stuff the chunks into the prompt, generate. No
graph, no iteration, no planning.

This baseline is built to be as strong as a single retrieval step honestly
allows: infoboxes are inside the indexed chunks, retrieval can be dense, BM25
or an RRF hybrid, and top_k is configurable. The point of the project is to
find where this is *enough* - a weak baseline would manufacture the conclusion
instead of testing it.
"""

from __future__ import annotations

from ..llm import GeminiClient
from ..trace import Trace
from .base import Answer, format_hits, synthesise


class RAGPipeline:
    name = "rag"

    def __init__(self, index, llm: GeminiClient, top_k: int = 5):
        self.index = index
        self.llm = llm
        self.top_k = top_k

    def run(self, qid: str, question: str, qtype: str = "") -> Answer:
        trace = Trace(qid=qid, pipeline=self.name, question=question, qtype=qtype)

        step = trace.step(
            "vector_search",
            f"single similarity retrieval, top_k={self.top_k}",
            query=question,
            top_k=self.top_k,
        )
        hits = self.index.search(question, top_k=self.top_k)
        step.doc_ids = list(dict.fromkeys(h.doc_id for h in hits))
        step.result_brief = f"{len(hits)} chunks from {len(step.doc_ids)} documents"

        answer = synthesise(self.llm, trace, question, format_hits(hits))
        return Answer(answer, trace.finish(answer, stop_reason="single-shot retrieval"))
