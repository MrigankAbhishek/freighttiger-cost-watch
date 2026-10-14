"""Synthetic traps for the hallucination guard: every case here must end up UNEXPLAINED
(except the two 'control' cases, which must be justified so we know the guard is not just refusing everything)."""
from datetime import date

import pandas as pd

from freight_watch.config import CONFIG
from freight_watch.grounding import decide
from freight_watch.notes import NoteIndex, parse_note

WEEK = pd.Timestamp("2025-03-10")  # Monday; week = Mar 10 - Mar 16


def _row(route="Mumbai-Pune", week=WEEK, own=20.0, excess=3.0):
    return {"route": route, "week_of": week, "own_pct": own, "peer_excess_pp": excess}


def _verdict(note_args, row):
    note = parse_note(*note_args)
    return decide(NoteIndex([note, parse_note("X99", date(2024, 1, 1), "All Routes", "Filler note about parking.")]), row)


# (case_id, description, note_args = (id, note_date, applies_to, text), row, expect_justified)
CASES = [
    ("A01", "wrong route: flood note for another route, same dates",
     ("T1", date(2025, 3, 10), "Delhi-Jaipur", "Flooding caused higher trip costs this week."), _row(), False),
    ("A02", "wrong window: right route, window ended the day before the week starts",
     ("T2", date(2025, 3, 1), "Mumbai-Pune", "Flooding from Mar 1 to Mar 9 caused higher trip costs."), _row(), False),
    ("A03", "wrong window: only 2 days overlap (< 3 required)",
     ("T3", date(2025, 3, 1), "Mumbai-Pune", "Flooding from Mar 1 to Mar 11 caused higher trip costs."), _row(), False),
    ("A04", "negation: right route + window but note says costs were not affected",
     ("T4", date(2025, 3, 10), "Mumbai-Pune", "Roadworks this week; costs were not significantly affected."), _row(), False),
    ("A05", "no cost statement at all (road resurfaced)",
     ("T5", date(2025, 3, 10), "Mumbai-Pune", "Authorities announced improved road conditions after resurfacing."), _row(), False),
    ("A06", "all-routes shock cannot explain a route pulling away from its peers",
     ("T6", date(2025, 3, 10), "All Routes", "Diesel prices rose nationwide starting this week, pushing up costs."),
     _row(excess=25.0), False),
    ("A07", "note dated after the week (no look-ahead / future note)",
     ("T7", date(2025, 4, 14), "Mumbai-Pune", "A festival week surcharge was applied on this corridor."), _row(), False),
    ("A08", "note far in the past (stale single-week note)",
     ("T8", date(2024, 3, 11), "Mumbai-Pune", "A festival week surcharge was applied on this corridor."), _row(), False),
    ("A09", "unrelated all-routes note with a negation (toll plaza not in dataset)",
     ("T9", date(2025, 3, 10), "All Routes", "A new toll plaza opened, but the affected routes are not part of this dataset."), _row(), False),
    ("C01", "CONTROL: right route, right week, states higher costs -> must be justified",
     ("C1", date(2025, 3, 10), "Mumbai-Pune", "A festival week saw a surcharge applied by transporters on this corridor."), _row(), True),
    ("C02", "CONTROL: all-routes diesel note, ordinary route (small peer excess) -> justified",
     ("C2", date(2025, 3, 10), "All Routes", "Diesel prices rose nationwide starting this week, pushing up costs by roughly 5-7%."),
     _row(own=8.0, excess=1.0), True),
]

# Hand-labelled expected effect for each given note (read from the text by a human).
NOTE_EFFECT_LABELS = {
    "N001": "increase", "N002": "increase", "N003": "increase", "N004": "no_increase", "N005": "no_increase",
    "N006": "no_increase", "N007": "unknown", "N008": "no_increase", "N009": "no_increase", "N010": "no_increase",
}


def run_adversarial():
    """Return list of (case_id, passed, description, got_note)."""
    out = []
    for cid, desc, note_args, row, expect in CASES:
        v = _verdict(note_args, row)
        out.append((cid, v.justified == expect, desc, v.matched.note.note_id if v.matched else ""))
    return out
