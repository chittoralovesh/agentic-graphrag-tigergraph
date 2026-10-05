"""Structured retrieval over the graph.

These are the operations a vector index cannot perform: exact attribute reads,
filtered counts, arg-max over a group, and edge walks. Each returns both an
answer and the ``doc_ids`` that support it, so every claim stays citable.

The same functions back the GraphRAG pipeline (fixed call order) and the
agentic pipeline (orchestrator picks the call), which keeps the comparison
honest: identical retrieval power, different control flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import re

from ..corpus import DASHES, DateSpan, norm, norm_event, parse_date
from ..graphmodel import EventNode, GraphTables


@dataclass
class Evidence:
    """A retrieval result plus its provenance."""

    kind: str
    value: Any
    doc_ids: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    note: str = ""

    def brief(self, limit: int = 12) -> str:
        v = self.value
        if isinstance(v, list):
            shown = v[:limit]
            more = f" (+{len(v)-limit} more)" if len(v) > limit else ""
            return f"{self.kind}: {shown}{more}"
        return f"{self.kind}: {v}"


# --------------------------------------------------------------------------
# Entity linking
# --------------------------------------------------------------------------

def link_event_by_title(g: GraphTables, title: str) -> Evidence:
    """Exact-then-fuzzy match of a full event title."""
    key = norm(title)
    qid = g.by_title.get(key)
    if qid:
        return Evidence("event", qid, [qid], {"title": g.events[qid].title, "match": "exact"})

    want = set(key.split())
    best, score = None, 0.0
    for k, cand in g.by_title.items():
        have = set(k.split())
        s = len(want & have) / len(want | have) if have else 0
        if s > score:
            best, score = cand, s
    if best and score >= 0.55:
        return Evidence(
            "event", best, [best],
            {"title": g.events[best].title, "match": "fuzzy", "score": round(score, 3)},
        )
    return Evidence("event", None, [], {}, note=f"no event matched {title!r}")


def resolve_sport(g: GraphTables, name: str) -> str | None:
    """Map a free-text sport ("cross-country skiing") onto a graph Sport."""
    n = norm(name)
    for s in g.sports:
        if norm(s) == n:
            return s
    for s in g.sports:
        if n and (n in norm(s) or norm(s) in n):
            return s
    return None


def resolve_games(g: GraphTables, text: str) -> str | None:
    """"2008 Summer Olympics" / "2008 Summer" -> the Games vertex key."""
    n = norm(text)
    for games in g.games:
        gn = norm(games)
        if gn == n or gn in n:
            return games
    return None


# --------------------------------------------------------------------------
# Attribute reads  (lookup questions)
# --------------------------------------------------------------------------

def get_event_attribute(g: GraphTables, qid: str, attr: str) -> Evidence:
    ev = g.events.get(qid)
    if not ev:
        return Evidence(attr, None, [], note=f"unknown event {qid}")
    value = getattr(ev, attr, None)
    return Evidence(
        attr, value, [qid],
        {"title": ev.title, "games": ev.games, "venue": ev.venue},
    )


# --------------------------------------------------------------------------
# Group filter / count / arg-max  (aggregation + superlative questions)
# --------------------------------------------------------------------------

def events_in(g: GraphTables, sport: str, games: str) -> list[EventNode]:
    sp = resolve_sport(g, sport)
    gm = resolve_games(g, games)
    if not sp or not gm:
        return []
    return [g.events[q] for q in g.by_sport_games.get((norm(sp), gm), [])]


def count_events_where(
    g: GraphTables, sport: str, games: str, attr: str, op: str, threshold: int
) -> Evidence:
    """COUNT events of a sport at a Games whose numeric attr passes a test.

    This is the operation top-k vector retrieval structurally cannot do: the
    answer depends on the whole group, which routinely runs to 40+ documents.
    """
    rows = events_in(g, sport, games)
    ops = {
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
    }
    test = ops.get(op)
    if test is None:
        return Evidence("count", None, [], note=f"unsupported operator {op!r}")

    hits = [e for e in rows if getattr(e, attr, None) is not None and test(getattr(e, attr), threshold)]
    # Cite every event examined, not only the matches: the count is only
    # defensible if the whole group was seen, and that is exactly the evidence
    # a top-k retriever cannot assemble.
    return Evidence(
        "count",
        len(hits),
        [e.qid for e in rows],
        {
            "sport": sport,
            "games": games,
            "predicate": f"{attr} {op} {threshold}",
            "group_size": len(rows),
            "matches": [{"title": e.title, attr: getattr(e, attr)} for e in hits],
        },
        note=f"scanned {len(rows)} events in {sport} at {games}",
    )


def argmax_event(g: GraphTables, sport: str, games: str, attr: str = "competitors") -> Evidence:
    rows = [e for e in events_in(g, sport, games) if getattr(e, attr, None) is not None]
    if not rows:
        return Evidence("argmax", None, [], note=f"no {sport} events at {games} with {attr}")
    best = max(rows, key=lambda e: getattr(e, attr))
    ranked = sorted(rows, key=lambda e: getattr(e, attr), reverse=True)[:5]
    return Evidence(
        "argmax",
        best.title,
        [e.qid for e in rows],
        {
            "winner": {"title": best.title, attr: getattr(best, attr)},
            "runners_up": [{"title": e.title, attr: getattr(e, attr)} for e in ranked[1:]],
            "group_size": len(rows),
        },
        note=f"compared {len(rows)} events",
    )


# --------------------------------------------------------------------------
# Venue + date lookup  (multi-hop questions)
# --------------------------------------------------------------------------

def find_event_by_venue_date(
    g: GraphTables, venue: str, date_text: str, games: str | None = None
) -> Evidence:
    """Find the event held at a venue on a date.

    Venue names repeat across Games ("Olympic Stadium"), so the date is what
    disambiguates. Candidates are scored rather than filtered, so the caller
    can see ties instead of silently getting the wrong one.
    """
    vkey = norm(venue)
    cands = g.by_venue.get(vkey, [])
    if not cands:  # fall back to substring match on venue names
        for k, qids in g.by_venue.items():
            if vkey and (vkey in k or k in vkey):
                cands.extend(qids)

    gm = resolve_games(g, games) if games else None
    want = parse_date(date_text)
    scored: list[tuple[float, EventNode]] = []
    for qid in dict.fromkeys(cands):
        ev = g.events[qid]
        if gm and ev.games != gm:
            continue
        if want and want.year and ev.year and want.year != ev.year:
            continue
        s = _date_score(want, ev.date)
        # The questions quote the infobox `date` field verbatim, so string
        # equality is decisive and must outrank any fuzzy overlap. Tiered:
        # a literal match beats a year-tolerant one, which is what separates
        # "30 July" (the answer) from its "30 July 2012" siblings.
        s += 20.0 if _same_date_text(date_text, ev.date_raw, None) else (
            10.0 if _same_date_text(date_text, ev.date_raw, ev.year) else 0.0
        )
        # Venues are often named after their sport ("Laura Biathlon & Ski
        # Complex"), which breaks ties when two events really did share a
        # venue and a day.
        if norm(ev.sport) and norm(ev.sport) in norm(venue):
            s += 0.5
        if s > 0:
            scored.append((s, ev))

    scored.sort(key=lambda t: -t[0])
    if not scored:
        return Evidence(
            "event", None, [],
            {"venue": venue, "date": date_text, "candidates": len(cands)},
            note="no event matched venue+date",
        )

    top = scored[0][1]
    tie = len(scored) > 1 and scored[1][0] == scored[0][0]
    return Evidence(
        "event",
        top.qid,
        [e.qid for _, e in scored[:4]],
        {
            "title": top.title,
            "venue": top.venue,
            "date": top.date_raw,
            "games": top.games,
            "ambiguous": tie,
            "alternatives": [
                {"title": e.title, "date": e.date_raw, "score": round(s, 2)}
                for s, e in scored[1:4]
            ],
        },
        note="ambiguous: several events share this venue and date" if tie else "",
    )


def _same_date_text(asked: str, stored: str | None, year: int | None) -> bool:
    """True when the question's date string is the infobox field verbatim.

    Tolerates the year being present on one side only ("30 July" vs
    "30 July 2012") and any dash variant inside a range.
    """
    if not asked or not stored:
        return False
    def canon(s: str) -> str:
        s = re.sub(f"[{DASHES}]", " to ", s)
        return norm(s)
    a, b = canon(asked), canon(stored)
    if a == b:
        return True
    if year:
        y = str(year)
        return a == f"{b} {y}".strip() or f"{a} {y}".strip() == b
    return False


def _date_score(want: DateSpan | None, have: DateSpan | None) -> float:
    """How well two date spans agree. 0 means incompatible."""
    if want is None or have is None:
        return 0.1  # no date signal either way - weak keep
    score = 0.0
    if want.month and have.month:
        if want.month == have.month or want.month == have.month_end or (
            have.month_end and want.month_end == have.month_end
        ):
            score += 2.0
        elif want.month_end and want.month_end == have.month:
            score += 1.5
        else:
            return 0.0
    if want.day_start and have.day_start:
        if want.day_start == have.day_start:
            score += 2.0
        elif have.day_start <= want.day_start <= (have.day_end or have.day_start):
            score += 1.0
        else:
            score -= 0.5
    if want.day_end and have.day_end and want.day_end == have.day_end:
        score += 1.0
    if want.year and have.year and want.year == have.year:
        score += 1.0
    return max(score, 0.0)


# --------------------------------------------------------------------------
# Temporal edge walk  (temporal questions)
# --------------------------------------------------------------------------

def previous_edition(g: GraphTables, qid: str) -> Evidence:
    """Hop to the same event at the preceding Games.

    Prefers the ``PREV_EDITION`` edge built from the infobox hint; when that
    hint is missing or unresolvable (~18% of events) it falls back to the
    nearest earlier year for the same discipline, which is what the edge was
    standing in for anyway.
    """
    ev = g.events.get(qid)
    if not ev:
        return Evidence("event", None, [], note=f"unknown event {qid}")
    if ev.prev_qid:
        p = g.events[ev.prev_qid]
        return Evidence("event", p.qid, [p.qid], {"title": p.title, "games": p.games, "via": "PREV_EDITION edge"})

    editions = g.by_event_key.get(ev.event_key, {})
    earlier = [y for y in editions if ev.year and y < ev.year]
    if earlier:
        y = max(earlier)
        p = g.events[editions[y]]
        return Evidence("event", p.qid, [p.qid], {"title": p.title, "games": p.games, "via": f"year fallback ({y})"})
    return Evidence("event", None, [], note=f"no earlier edition of {ev.title}")


def edition_before_year(g: GraphTables, sport: str, discipline: str, year: int, season: str | None = None) -> Evidence:
    """The edition of a discipline at the Games immediately before ``year``.

    Answers "...at the Summer Olympics held immediately before 2016" directly,
    without first needing to locate the 2016 edition.
    """
    sp = resolve_sport(g, sport) if sport else None
    want = set(norm_event(discipline).split())
    best: tuple[float, int, EventNode] | None = None

    for ev in g.events.values():
        if not ev.year or ev.year >= year:
            continue
        if season and ev.season != season:
            continue
        if sp and norm(ev.sport) != norm(sp):
            continue
        have = set(ev.event_key.split())
        if not have or not want:
            continue
        score = len(want & have) / len(want | have)
        if score < 0.6:
            continue
        cand = (score, ev.year, ev)
        if best is None or (cand[1], cand[0]) > (best[1], best[0]):
            best = cand

    if not best:
        return Evidence("event", None, [], note=f"no edition of {discipline!r} before {year}")
    ev = best[2]
    # Cite the anchor edition too: the claim is "this is the edition before
    # <year>", which is only checkable against the event at <year> itself.
    cites = [ev.qid]
    anchor = g.by_event_key.get(ev.event_key, {}).get(year)
    if anchor:
        cites.append(anchor)
    return Evidence(
        "event", ev.qid, cites,
        {"title": ev.title, "games": ev.games, "year": ev.year, "match_score": round(best[0], 3)},
    )
