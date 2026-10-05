"""Scoring answers against ground truth.

Two layers, deliberately:

``exact_match``   normalised string comparison. Deterministic, free, and right
                  for the numeric and title answers that dominate this dataset.
``llm_judge``     a PASS/FAIL grader for the cases exact match would wrongly
                  fail - "Naim Suleymanoglu" vs "Naim Süleymanoğlu", or a team
                  medallist string with different name separators.

The judge is only consulted when exact match fails, so grading cost stays
proportional to disagreement rather than to dataset size, and a judge outage
can never silently downgrade an already-correct answer.
"""

from __future__ import annotations

import re
import unicodedata

from ..llm import GeminiClient
from ..trace import TokenUsage

JUDGE_PROMPT = """You grade one answer against the reference. Reply with JSON only.

Equivalent answers are PASS, even when written differently:
- accents or transliteration ("Suleymanoglu" = "Süleymanoğlu")
- team medallists listed with or without separators
- a number written as digits or words ("5" = "five")
- an event title with or without its "Sport at the YYYY Games -" prefix

Different facts are FAIL: a different person, a different number, a different event.
An answer of INSUFFICIENT or an empty answer is FAIL.

QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}

{{"verdict": "PASS"|"FAIL", "why": "<short>"}}"""


def normalise(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def exact_match(answer: str | None, references: list[str]) -> bool:
    if not answer:
        return False
    a = normalise(answer)
    if not a:
        return False
    for ref in references:
        r = normalise(str(ref))
        if a == r:
            return True
        # Titles are often returned with or without the "Sport at the ..." prefix.
        if r and (a.endswith(r) or r.endswith(a)) and min(len(a), len(r)) >= 8:
            return True
    return False


def llm_judge(
    llm: GeminiClient, question: str, answer: str | None, references: list[str]
) -> tuple[bool, str, TokenUsage]:
    if not answer:
        return False, "no answer produced", TokenUsage()
    try:
        reply = llm.generate(
            JUDGE_PROMPT.format(
                question=question,
                reference=" | ".join(str(r) for r in references),
                answer=answer,
            )
        )
        verdict = reply.json()
        passed = str(verdict.get("verdict", "")).upper() == "PASS"
        return passed, str(verdict.get("why", ""))[:160], reply.tokens
    except Exception as exc:  # noqa: BLE001
        return False, f"judge error: {str(exc)[:120]}", TokenUsage()


def grade(
    llm: GeminiClient | None, question: str, answer: str | None, references: list[str]
) -> dict:
    """Exact match first; consult the judge only when it fails."""
    if exact_match(answer, references):
        return {"correct": True, "method": "exact", "why": "", "judge_tokens": 0}
    if llm is None:
        return {"correct": False, "method": "exact", "why": "no match", "judge_tokens": 0}
    passed, why, tokens = llm_judge(llm, question, answer, references)
    return {
        "correct": passed,
        "method": "llm_judge",
        "why": why,
        "judge_tokens": tokens.total,
    }
