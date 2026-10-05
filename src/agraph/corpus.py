"""Parse the hackathon corpus into prose + typed infobox records.

Every document in ``corpus.jsonl`` begins with a machine-readable infobox::

    [Infobox Olympic event]
      event: Men's canoe sprint K-2 1,000 metres
      games: 2012 Summer
      venue: Eton Dorney
      date: 6 to 8 August
      competitors: 24
      gold: Rudolf DombiRoland Kokeny
      prev: 2008
      next: 2016

2,162 of the 2,951 documents are ``Olympic event``; the rest (films, politicians,
companies) are distractors that only ever reach the vector lane.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

INFOBOX_RE = re.compile(r"\[Infobox ([^\]]+)\]\n")
OLYMPIC = "Olympic event"

MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})

# En/em dashes and friends all mean "to" inside a date range.
DASHES = "‐‑‒–—―−-"

# Titles separate sport from discipline with an en dash: "Cycling at the 2012
# Summer Olympics - Men's cross-country". The ASCII hyphen is deliberately
# excluded: it binds names like "cross-country" and "Greco-Roman", and
# splitting on it silently truncates the discipline key.
TITLE_SEP = re.compile(r"[‒–—―−]")


@dataclass
class Document:
    """One corpus document: raw text plus its parsed infobox."""

    doc_id: str
    title: str
    url: str
    wikidata_qid: str
    wikipedia_pageid: int
    approx_tokens: int
    text: str
    infobox_type: str | None
    fields: dict[str, str] = field(default_factory=dict)

    @property
    def is_olympic_event(self) -> bool:
        return self.infobox_type == OLYMPIC

    @property
    def prose(self) -> str:
        """Document text with the infobox block stripped off."""
        m = INFOBOX_RE.match(self.text)
        if not m:
            return self.text
        rest = self.text[m.end() :]
        lines = rest.split("\n")
        i = 0
        while i < len(lines) and (lines[i].startswith("  ") or not lines[i].strip()):
            i += 1
        return "\n".join(lines[i:]).strip()

    @property
    def infobox_text(self) -> str:
        """The infobox rendered back as text, for embedding alongside prose."""
        if not self.fields:
            return ""
        body = "\n".join(f"{k}: {v}" for k, v in self.fields.items())
        return f"{self.title}\n{body}"


def parse_infobox(text: str) -> tuple[str | None, dict[str, str]]:
    m = INFOBOX_RE.match(text)
    if not m:
        return None, {}
    fields: dict[str, str] = {}
    for line in text[m.end() :].split("\n"):
        if line.startswith("  ") and ":" in line:
            k, v = line.strip().split(":", 1)
            k, v = k.strip(), v.strip()
            if k and v and k not in fields:
                fields[k] = v
        elif not line.strip():
            continue
        else:
            break  # first prose line ends the infobox
    return m.group(1), fields


def load_corpus(path: str | Path) -> list[Document]:
    docs: list[Document] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            ib_type, fields = parse_infobox(raw["text"])
            docs.append(
                Document(
                    doc_id=raw["doc_id"],
                    title=raw["title"],
                    url=raw["url"],
                    wikidata_qid=raw.get("wikidata_qid", ""),
                    wikipedia_pageid=raw.get("wikipedia_pageid", 0),
                    approx_tokens=raw.get("approx_tokens", 0),
                    text=raw["text"],
                    infobox_type=ib_type,
                    fields=fields,
                )
            )
    return docs


def iter_questions(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


# --------------------------------------------------------------------------
# Normalisation helpers
# --------------------------------------------------------------------------

def norm(s: str) -> str:
    """Casefold, strip accents and punctuation. For fuzzy key matching."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def parse_int(value: str | None) -> int | None:
    """``"24"`` -> 24, ``"1,024 (est.)"`` -> 1024, junk -> None."""
    if not value:
        return None
    m = re.search(r"\d[\d,]*", value)
    if not m:
        return None
    try:
        return int(m.group(0).replace(",", ""))
    except ValueError:
        return None


def split_games(games: str | None) -> tuple[int | None, str | None]:
    """``"2012 Summer"`` -> ``(2012, "Summer")``."""
    if not games:
        return None, None
    m = re.match(r"\s*(\d{4})\s+(Summer|Winter)", games)
    if not m:
        y = parse_int(games)
        return y, None
    return int(m.group(1)), m.group(2)


def sport_of(title: str) -> str:
    """``"Canoeing at the 2012 Summer Olympics - Men's K-2"`` -> ``"Canoeing"``."""
    return title.split(" at the ")[0].strip()


def event_suffix(title: str) -> str:
    """The discipline part after the en-dash: ``"Men's K-2 1000 metres"``.

    This is the key that links the same event across Games, which is what
    ``prev``/``next`` edges and temporal questions depend on.
    """
    parts = TITLE_SEP.split(title)
    return parts[-1].strip() if len(parts) > 1 else title.strip()


def norm_event(s: str) -> str:
    """Normalise a discipline name, preserving weight-class polarity.

    ``norm`` would turn both "Men's +80 kg" and "Men's 80 kg" into the same
    key, which silently merges two different events (and two different gold
    medallists). Spell the signs out before punctuation is stripped.
    """
    s = (s or "").replace("+", " plus ")
    s = re.sub(r"(?<![a-zA-Z0-9])[−-](?=\s*\d)", " minus ", s)
    return norm(s)


# --------------------------------------------------------------------------
# Date parsing
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DateSpan:
    """A resolved (start, end) calendar span. Month/day may be absent."""

    year: int | None
    month: int | None
    day_start: int | None
    day_end: int | None
    month_end: int | None = None
    raw: str = ""

    def key(self) -> tuple:
        return (self.year, self.month, self.day_start, self.day_end)

    def overlaps(self, other: "DateSpan") -> bool:
        """True when the two spans could describe the same event."""
        if self.year and other.year and self.year != other.year:
            return False
        if self.month and other.month and self.month != other.month:
            return False
        if self.day_start and other.day_start and self.day_start != other.day_start:
            return False
        return True


def parse_date(value: str | None, year_hint: int | None = None) -> DateSpan | None:
    """Parse the many date shapes the corpus and questions use.

    Handles ``6 to 8 August``, ``August 12, 2008``, ``15 to 22 August 2004``,
    ``28 July - 4 August 2012``, ``16-18 August 2016``, ``22 September 2000``
    and ``August 14, 2004 (heats & final)``.
    """
    if not value:
        return None
    s = value.strip()
    s = re.sub(r"\([^)]*\)", " ", s)  # drop "(heats & final)" asides
    s = re.sub(f"[{DASHES}]", " to ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None

    year = None
    ym = re.search(r"\b(1[89]\d{2}|20\d{2})\b", s)
    if ym:
        year = int(ym.group(1))
        s = (s[: ym.start()] + " " + s[ym.end() :]).strip()
    elif year_hint:
        year = year_hint

    s = s.replace(",", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # Collect month/day tokens in order of appearance.
    tokens = re.findall(r"[A-Za-z]+|\d{1,2}", s)
    months: list[tuple[int, int]] = []  # (position, month)
    days: list[tuple[int, int]] = []  # (position, day)
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low in MONTHS:
            months.append((i, MONTHS[low]))
        elif tok.isdigit():
            d = int(tok)
            if 1 <= d <= 31:
                days.append((i, d))

    if not months and not days:
        return None

    month = months[0][1] if months else None
    month_end = months[-1][1] if len(months) > 1 else None
    day_start = days[0][1] if days else None
    day_end = days[-1][1] if len(days) > 1 else None

    # "28 July to 4 August": the first day belongs to the first month, so when
    # months differ the second day pairs with month_end, not month.
    if month_end and month_end != month and len(days) > 1:
        pass  # day_start/month, day_end/month_end - already correct

    return DateSpan(
        year=year,
        month=month,
        day_start=day_start,
        day_end=day_end,
        month_end=month_end,
        raw=value,
    )


def doc_date(fields: dict[str, str], year_hint: int | None) -> DateSpan | None:
    """Prefer ``date``, fall back to ``dates``."""
    return parse_date(fields.get("date") or fields.get("dates"), year_hint)


# --------------------------------------------------------------------------
# Athlete name splitting
# --------------------------------------------------------------------------

def split_athletes(raw: str | None) -> list[str]:
    """Split run-together medallist strings into individual names.

    The corpus concatenates team members without a separator
    (``"Dani KingLaura TrottJoanna Rowsell"``). Split on a lower-to-upper
    transition, which is the only boundary signal available.

    Note: callers answering medal questions should use the *raw* string, since
    the dataset's gold answers keep the names concatenated.
    """
    if not raw:
        return []
    s = raw.strip()
    if not s:
        return []
    parts = re.split(r"(?<=[a-zß-ÿ])(?=[A-ZÀ-Þ])", s)
    return [p.strip() for p in parts if p.strip()]
