"""Chunk corpus documents for the vector lane.

Design note that matters for a fair benchmark: the infobox is prepended to the
*first* chunk of every document rather than discarded. Infoboxes hold the
numeric facts (competitors, nations, medallists) that most questions turn on,
so dropping them would hobble the RAG baseline and make the agentic pipeline
look better than it is. All 2,951 documents are chunked, including the 789
film/politician distractors, so retrieval has to actually discriminate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    ordinal: int
    approx_tokens: int

    @property
    def embed_text(self) -> str:
        """Title-prefixed text: the title carries sport/Games/discipline."""
        return f"{self.title}\n\n{self.text}"


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_document(
    doc: Document, target_tokens: int = 420, overlap_tokens: int = 60
) -> list[Chunk]:
    """Pack paragraphs into ~target_tokens chunks with a small overlap.

    Paragraph boundaries are respected where possible; an oversized paragraph
    (long results tables) is split on sentence boundaries instead.
    """
    header = doc.infobox_text.strip()
    body = doc.prose.strip()
    paras = split_paragraphs(body)
    if header:
        paras.insert(0, header)
    if not paras:
        paras = [doc.title]

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf, buf_tokens
        if not buf:
            return
        text = "\n\n".join(buf).strip()
        if text:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{len(chunks)}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=text,
                    ordinal=len(chunks),
                    approx_tokens=_approx_tokens(text),
                )
            )
        buf, buf_tokens = [], 0

    for para in paras:
        ptok = _approx_tokens(para)
        if ptok > target_tokens * 1.6:
            flush()
            for piece in _split_long(para, target_tokens):
                buf = [piece]
                buf_tokens = _approx_tokens(piece)
                flush()
            continue
        if buf_tokens + ptok > target_tokens and buf:
            tail = buf[-1] if overlap_tokens else ""
            flush()
            if tail and _approx_tokens(tail) <= overlap_tokens:
                buf, buf_tokens = [tail], _approx_tokens(tail)
        buf.append(para)
        buf_tokens += ptok

    flush()
    return chunks


def _split_long(para: str, target_tokens: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", para)
    out, buf, tok = [], [], 0
    for s in sentences:
        st = _approx_tokens(s)
        if tok + st > target_tokens and buf:
            out.append(" ".join(buf))
            buf, tok = [], 0
        buf.append(s)
        tok += st
    if buf:
        out.append(" ".join(buf))
    return out or [para]


def chunk_corpus(docs: list[Document], **kw) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, **kw))
    return out
