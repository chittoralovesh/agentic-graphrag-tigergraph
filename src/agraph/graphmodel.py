"""Project the corpus into a typed property graph.

Vertices
    Document      every corpus doc, including distractors (provenance + citations)
    OlympicEvent  one per "Infobox Olympic event" doc, with the numeric fields
                  the benchmark actually asks about
    Games         "2012 Summer"
    Sport         "Canoeing"
    Venue         "Eton Dorney"
    Athlete       individual medallists
    NOC           "HUN"

Edges
    Games   -HAS_EVENT->      OlympicEvent
    OlympicEvent -OF_SPORT->  Sport
    OlympicEvent -AT_VENUE->  Venue
    OlympicEvent -WON_GOLD/SILVER/BRONZE-> Athlete
    Athlete -REPRESENTS->     NOC
    OlympicEvent -PREV_EDITION/NEXT_EDITION-> OlympicEvent   (temporal spine)
    OlympicEvent -SOURCED_FROM-> Document
    Document -HAS_CHUNK->     Chunk

The ``PREV_EDITION`` spine is what makes "the Games held immediately before
2016" a single hop instead of a search.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .corpus import (
    Document,
    DateSpan,
    doc_date,
    event_suffix,
    norm,
    norm_event,
    parse_int,
    split_athletes,
    split_games,
    sport_of,
)


@dataclass
class EventNode:
    qid: str
    title: str
    sport: str
    event_name: str
    event_key: str  # normalised discipline, stable across Games
    games: str | None
    year: int | None
    season: str | None
    venue: str | None
    competitors: int | None
    nations: int | None
    win_value: str | None
    date_raw: str | None
    date: DateSpan | None
    gold_raw: str | None
    silver_raw: str | None
    bronze_raw: str | None
    gold_noc: str | None
    silver_noc: str | None
    bronze_noc: str | None
    prev_year: int | None
    next_year: int | None
    url: str = ""
    prev_qid: str | None = None
    next_qid: str | None = None

    @property
    def date_month(self) -> int | None:
        return self.date.month if self.date else None

    @property
    def date_day_start(self) -> int | None:
        return self.date.day_start if self.date else None


@dataclass
class GraphTables:
    events: dict[str, EventNode] = field(default_factory=dict)
    documents: list[Document] = field(default_factory=list)
    games: dict[str, dict] = field(default_factory=dict)
    sports: set[str] = field(default_factory=set)
    venues: dict[str, str] = field(default_factory=dict)  # norm -> display
    athletes: dict[str, str] = field(default_factory=dict)
    nocs: set[str] = field(default_factory=set)
    edges: dict[str, list[tuple]] = field(default_factory=lambda: defaultdict(list))

    # ---- indexes used by the retrieval tools -----------------------------
    by_title: dict[str, str] = field(default_factory=dict)
    by_sport_games: dict[tuple[str, str], list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    by_venue: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    by_event_key: dict[str, dict[int, str]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    def stats(self) -> dict:
        return {
            "documents": len(self.documents),
            "olympic_events": len(self.events),
            "games": len(self.games),
            "sports": len(self.sports),
            "venues": len(self.venues),
            "athletes": len(self.athletes),
            "nocs": len(self.nocs),
            "edges": {k: len(v) for k, v in sorted(self.edges.items())},
        }


def build_graph(docs: Iterable[Document]) -> GraphTables:
    g = GraphTables()
    g.documents = list(docs)

    for d in g.documents:
        if not d.is_olympic_event:
            continue
        f = d.fields
        year, season = split_games(f.get("games"))
        date = doc_date(f, year)
        sport = sport_of(d.title)
        suffix = event_suffix(d.title)

        ev = EventNode(
            qid=d.doc_id,
            title=d.title,
            sport=sport,
            event_name=f.get("event") or suffix,
            event_key=norm_event(suffix),
            games=f.get("games"),
            year=year,
            season=season,
            venue=f.get("venue"),
            competitors=parse_int(f.get("competitors")),
            nations=parse_int(f.get("nations")),
            win_value=f.get("win_value"),
            date_raw=f.get("date") or f.get("dates"),
            date=date,
            gold_raw=f.get("gold"),
            silver_raw=f.get("silver"),
            bronze_raw=f.get("bronze"),
            gold_noc=f.get("goldNOC"),
            silver_noc=f.get("silverNOC"),
            bronze_noc=f.get("bronzeNOC"),
            prev_year=parse_int(f.get("prev")),
            next_year=parse_int(f.get("next")),
            url=d.url,
        )
        g.events[ev.qid] = ev

        # vertices
        if ev.games:
            g.games.setdefault(ev.games, {"name": ev.games, "year": year, "season": season})
            g.edges["HAS_EVENT"].append((ev.games, ev.qid))
        g.sports.add(sport)
        g.edges["OF_SPORT"].append((ev.qid, sport))
        if ev.venue:
            g.venues.setdefault(norm(ev.venue), ev.venue)
            g.edges["AT_VENUE"].append((ev.qid, norm(ev.venue)))
            g.by_venue[norm(ev.venue)].append(ev.qid)

        for medal, raw, noc in (
            ("WON_GOLD", ev.gold_raw, ev.gold_noc),
            ("WON_SILVER", ev.silver_raw, ev.silver_noc),
            ("WON_BRONZE", ev.bronze_raw, ev.bronze_noc),
        ):
            for name in split_athletes(raw):
                g.athletes.setdefault(norm(name), name)
                g.edges[medal].append((ev.qid, norm(name)))
                if noc:
                    g.nocs.add(noc)
                    g.edges["REPRESENTS"].append((norm(name), noc))

        g.edges["SOURCED_FROM"].append((ev.qid, d.doc_id))

        # indexes
        g.by_title[norm(d.title)] = ev.qid
        if ev.games:
            g.by_sport_games[(norm(sport), ev.games)].append(ev.qid)
        if year:
            g.by_event_key[ev.event_key][year] = ev.qid

    _link_editions(g)
    return g


def _link_editions(g: GraphTables) -> None:
    """Resolve ``prev``/``next`` year hints into real edges.

    The infobox only records a year ("prev: 2008"), so the edge is recovered by
    looking up the same discipline key in that year. Falls back to a
    season-matched search when the discipline was renamed between Games.
    """
    for ev in g.events.values():
        for attr, target_year, edge in (
            ("prev_qid", ev.prev_year, "PREV_EDITION"),
            ("next_qid", ev.next_year, "NEXT_EDITION"),
        ):
            if not target_year:
                continue
            cand = g.by_event_key.get(ev.event_key, {}).get(target_year)
            if cand is None:
                cand = _fuzzy_edition(g, ev, target_year)
            if cand and cand != ev.qid:
                setattr(ev, attr, cand)
                g.edges[edge].append((ev.qid, cand))


def _fuzzy_edition(g: GraphTables, ev: EventNode, year: int) -> str | None:
    """Same sport + season + year, with the closest discipline name."""
    best, best_score = None, 0.0
    want = set(ev.event_key.split())
    if not want:
        return None
    for qid, other in g.events.items():
        if other.year != year or other.season != ev.season:
            continue
        if norm(other.sport) != norm(ev.sport):
            continue
        have = set(other.event_key.split())
        if not have:
            continue
        score = len(want & have) / len(want | have)
        if score > best_score:
            best, best_score = qid, score
    return best if best_score >= 0.6 else None
