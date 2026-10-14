"""Retrieve candidate notes for a flagged route-week, then verify each one strictly.

Retrieval only *ranks*. Whether a note may justify a rise is decided by `verify`, which needs ALL of:
  1. scope     - note is for this route, or for "All Routes"
  2. effect    - note actually describes a cost increase (negations / unknowns never pass)
  3. window    - note's effective window covers >= min_window_overlap_days of the week
  4. coherence - an "All Routes" note cannot explain a route that is pulling away from its peers
No note passing all four  ->  unexplained.  (No guessing, no "closest match" justifications.)
"""
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from .config import CONFIG, Config
from .notes import ALL_ROUTES, Note, NoteIndex

MATCH = "MATCH"
WRONG_ROUTE = "WRONG_ROUTE"
NO_COST_INCREASE = "NO_COST_INCREASE"
EFFECT_UNKNOWN = "EFFECT_UNKNOWN"
OUTSIDE_WINDOW = "OUTSIDE_WINDOW"
NETWORKWIDE_OUTLIER = "NETWORKWIDE_BUT_ROUTE_IS_OUTLIER"


@dataclass
class Candidate:
    note: Note
    tfidf: float
    scope_match: bool
    gap_days: int  # 0 when the note's window overlaps the week, else days between them
    overlap_days: int
    score: float
    code: str = ""


@dataclass
class Verdict:
    justified: bool
    matched: Optional[Candidate]
    considered: list = field(default_factory=list)  # every retrieved candidate, with its code
    caveat: str = ""


def week_span(week_of) -> tuple:
    start = pd.Timestamp(week_of).date()
    return start, start + timedelta(days=6)


def overlap_and_gap(note: Note, week_start: date, week_end: date) -> tuple:
    lo, hi = max(note.window_start, week_start), min(note.window_end, week_end)
    overlap = max(0, (hi - lo).days + 1)
    if overlap:
        return overlap, 0
    gap = (note.window_start - week_end).days if note.window_start > week_end else (week_start - note.window_end).days
    return 0, gap


def build_query(route: str) -> str:
    cities = route.replace("-", " ")
    return f"{cities} freight cost increase surcharge fuel diesel toll flood disruption festival"


def retrieve(index: NoteIndex, route: str, week_of, cfg: Config = CONFIG) -> list:
    """Rank all notes for this route-week: text similarity + scope match + closeness in time."""
    ws, we = week_span(week_of)
    sims = index.similarities(build_query(route))
    cands = []
    for note, sim in zip(index.notes, sims):
        scope = note.applies_to in (route, ALL_ROUTES)
        overlap, gap = overlap_and_gap(note, ws, we)
        recency = max(0.0, 1.0 - gap / cfg.recency_scale_days)
        cands.append(Candidate(note, float(sim), scope, gap, overlap, float(sim) + float(scope) + recency))
    cands.sort(key=lambda c: (-c.score, c.note.note_id))  # note_id breaks ties deterministically
    return cands[: cfg.retrieval_top_k]


def verify(c: Candidate, peer_excess_pp: float, cfg: Config = CONFIG) -> str:
    n = c.note
    if not c.scope_match:
        return WRONG_ROUTE
    if n.effect == "no_increase":
        return NO_COST_INCREASE
    if n.effect != "increase":
        return EFFECT_UNKNOWN
    if c.overlap_days < cfg.min_window_overlap_days:
        return OUTSIDE_WINDOW
    if n.applies_to == ALL_ROUTES and pd.notna(peer_excess_pp) and peer_excess_pp >= cfg.peer_moderate_pp:
        return NETWORKWIDE_OUTLIER
    return MATCH


def decide(index: NoteIndex, row, cfg: Config = CONFIG) -> Verdict:
    """`row` needs: route, week_of, own_pct, peer_excess_pp."""
    cands = retrieve(index, row["route"], row["week_of"], cfg)
    for c in cands:
        c.code = verify(c, row["peer_excess_pp"], cfg)
    matched = next((c for c in cands if c.code == MATCH), None)
    caveat = ""
    if matched and matched.note.magnitude:
        lo, hi = matched.note.magnitude
        if row["own_pct"] > hi + cfg.magnitude_tolerance_pp:
            caveat = (f"The observed rise (+{row['own_pct']:.1f}%) is larger than the {lo:g}-{hi:g}% the note "
                      f"quotes, so part of it may be unexplained.")
    return Verdict(justified=matched is not None, matched=matched, considered=cands, caveat=caveat)
