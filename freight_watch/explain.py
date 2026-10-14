"""Plain-English reasons, built only from the matched/rejected notes and the computed numbers."""
from .config import CONFIG, Config
from .grounding import (EFFECT_UNKNOWN, NETWORKWIDE_OUTLIER, NO_COST_INCREASE, OUTSIDE_WINDOW, Verdict,
                        week_span)
from .notes import Note

NO_NOTE_REASON = ("Unexplained. No matching note found for this route or date range. "
                  "Cost rise looks unexplained and worth a human review.")


def _trim(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def describe_window(note: Note) -> str:
    return f"{note.window_start} to {note.window_end}"


def window_caveat(note: Note, cfg: Config = CONFIG) -> str:
    """Extra sentence when the window was derived (not stated) - say so openly."""
    if note.window_kind == "step_change":
        return (f"The note says the change started that week, so it is treated as explaining rises for "
                f"{cfg.step_change_horizon_weeks} weeks (to {note.window_end}).")
    return ""


def justified_reason(verdict: Verdict, row, cfg: Config = CONFIG) -> str:
    n = verdict.matched.note
    ws, we = week_span(row["week_of"])
    scope = "applies to all routes" if n.applies_to == "All Routes" else f"is specific to {n.applies_to}"
    text = (f"Justified. Matches note {n.note_id} dated {n.date}: {n.first_sentence} "
            f"The note {scope}, and its window ({describe_window(n)}) covers the week of {ws} "
            f"({ws} to {we}). {window_caveat(n, cfg)} The cost rise has a clear explanation.")
    text = " ".join(text.split())  # tidy double spaces when there is no window caveat
    return f"{text} {verdict.caveat}".strip()


def _rejection_sentence(c, row, cfg: Config) -> str:
    n = c.note
    quote = f'{n.note_id} ({n.date}) says "{_trim(n.first_sentence)}"'
    if c.code == NO_COST_INCREASE:
        return f"{quote} - it does not describe a reason for a cost rise on this route"
    if c.code == EFFECT_UNKNOWN:
        return f"{quote} - it does not describe any cost increase"
    if c.code == OUTSIDE_WINDOW:
        return (f"{n.note_id} ({n.date}) describes a rise covering {describe_window(n)}, "
                f"which does not cover the week of {week_span(row['week_of'])[0]}")
    if c.code == NETWORKWIDE_OUTLIER:
        return (f"{n.note_id} ({n.date}) describes a rise across all routes, but this route moved "
                f"{row['peer_excess_pp']:.0f} pp further from similar routes than it usually sits, which an "
                f"all-routes shock does not explain")
    return f"{n.note_id} ({n.date}) does not apply"


def unexplained_reason(verdict: Verdict, row, cfg: Config = CONFIG) -> str:
    """Mention (at most two) nearby, in-scope notes and why each was rejected."""
    near = [c for c in verdict.considered if c.scope_match and c.gap_days <= cfg.proximity_days][:2]
    if not near:
        return NO_NOTE_REASON
    label = "Closest note checked:" if len(near) == 1 else "Closest notes checked:"
    parts = "; ".join(_rejection_sentence(c, row, cfg) for c in near)
    return f"Unexplained. {label} {parts}. No genuine justification found; flagged for review."


def build_reason(verdict: Verdict, row, cfg: Config = CONFIG) -> str:
    return justified_reason(verdict, row, cfg) if verdict.justified else unexplained_reason(verdict, row, cfg)
