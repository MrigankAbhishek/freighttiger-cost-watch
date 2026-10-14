"""Load context notes, turn each into structured facts, and index them for retrieval.

Each note becomes: scope (route or all routes), effect (does it describe a cost *increase*?),
and an effective time window. Everything is extracted by explicit, auditable rules - an
unrecognised note ends up with effect="unknown" and therefore can never justify anything.
"""
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import CONFIG, Config

ALL_ROUTES = "All Routes"

# Negation wins over increase cues: "costs were not significantly affected" must never justify.
NO_INCREASE_PATTERNS = [
    r"\bnot (?:significantly |materially )?affected\b",
    r"\bno (?:major|significant|material) (?:disruptions?|impact|change)\b",
    r"\bremained (?:stable|normal)\b",
    r"\breturned to normal\b",
    r"\bwithout a (?:rate|price|cost) (?:change|increase)\b",
    r"\bnot part of this dataset\b",
]
INCREASE_PATTERNS = [
    r"\bhigher\b[^.]*\bcosts?\b",
    r"\bsurcharge\b",
    r"\bpushing up\b",
    r"\bcosts? (?:rose|increased|went up)\b",
    r"\b(?:rate|toll|price) (?:hike|increase)\b",
]
STEP_CHANGE_PATTERN = r"\b(?:starting|beginning|from) this week\b"
RANGE_PATTERN = (
    r"\bfrom ([A-Z][a-z]{2})[a-z]* (\d{1,2}) (?:to|through|until) ([A-Z][a-z]{2})[a-z]* (\d{1,2})\b"
)
MAGNITUDE_PATTERN = r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*%"
MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}


@dataclass(frozen=True)
class Note:
    note_id: str
    date: date
    applies_to: str
    text: str
    effect: str  # "increase" | "no_increase" | "unknown"
    window_start: date
    window_end: date  # inclusive
    window_kind: str  # "explicit_range" | "step_change" | "single_week"
    magnitude: Optional[tuple]  # quoted % range, e.g. (5.0, 7.0)

    @property
    def first_sentence(self) -> str:
        return re.split(r"(?<=[.!?])\s+", self.text.strip())[0]


def monday_of(d: date) -> date:
    return d - timedelta(days=d.weekday())


def extract_effect(text: str) -> str:
    if any(re.search(p, text, re.I) for p in NO_INCREASE_PATTERNS):
        return "no_increase"
    if any(re.search(p, text, re.I) for p in INCREASE_PATTERNS):
        return "increase"
    return "unknown"


def extract_window(text: str, note_date: date, cfg: Config = CONFIG):
    """Return (start, end_inclusive, kind). Order: explicit range > step change > the note's own week."""
    m = re.search(RANGE_PATTERN, text)
    if m:
        m1, d1, m2, d2 = MONTHS[m.group(1)], int(m.group(2)), MONTHS[m.group(3)], int(m.group(4))
        # pick the year that puts the start closest to the note date
        start = min((date(y, m1, d1) for y in (note_date.year - 1, note_date.year, note_date.year + 1)),
                    key=lambda c: abs((c - note_date).days))
        end = date(start.year + (1 if (m2, d2) < (m1, d1) else 0), m2, d2)
        return start, end, "explicit_range"
    if re.search(STEP_CHANGE_PATTERN, text, re.I):
        start = monday_of(note_date)
        return start, start + timedelta(weeks=cfg.step_change_horizon_weeks) - timedelta(days=1), "step_change"
    start = monday_of(note_date)
    return start, start + timedelta(days=6), "single_week"


def parse_note(note_id: str, note_date: date, applies_to: str, text: str, cfg: Config = CONFIG) -> Note:
    start, end, kind = extract_window(text, note_date, cfg)
    mag = re.search(MAGNITUDE_PATTERN, text)
    return Note(note_id=note_id, date=note_date, applies_to=applies_to.strip(), text=text.strip(),
                effect=extract_effect(text), window_start=start, window_end=end, window_kind=kind,
                magnitude=(float(mag.group(1)), float(mag.group(2))) if mag else None)


def load_notes(path, cfg: Config = CONFIG) -> list:
    df = pd.read_csv(path)
    return [parse_note(r.note_id, pd.Timestamp(r.date).date(), r.applies_to, r.note, cfg)
            for r in df.itertuples(index=False)]


class NoteIndex:
    """TF-IDF vector index over the notes (deterministic, no model download, no randomness)."""

    def __init__(self, notes: list):
        self.notes = notes
        docs = [f"{n.applies_to} {n.text}" for n in notes]
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                                          sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(docs)  # rows are L2-normalised

    def similarities(self, query: str) -> np.ndarray:
        """Cosine similarity of `query` to every note, in the order of self.notes."""
        q = self.vectorizer.transform([query])
        return (self.matrix @ q.T).toarray().ravel()
