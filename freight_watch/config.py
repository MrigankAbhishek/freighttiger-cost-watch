"""All tunable numbers live here so the walkthrough (and the README) can point at one place."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- baselines (fixed by the brief, do not change) ---------------------------------
    history_weeks: int = 8  # trailing window for "vs own history"

    # --- flagging rule (our choice; justified in README "Flag rule") --------------------
    min_history_weeks: int = 4  # need this many prior weeks before any flag is raised
    peer_baseline_min_weeks: int = 8  # prior weeks needed to know a route's usual gap to its peers
    own_strong_pct: float = 8.0  # rise vs own history that is unusual on its own (~2.4 sigma of weekly noise)
    peer_strong_pp: float = 15.0  # rise in gap-to-peers (percentage points) unusual on its own
    own_moderate_pct: float = 6.0  # moderate own-history rise ...
    peer_moderate_pp: float = 10.0  # ... that is only flagged when the peer gap moved too

    # --- grounding rules (what counts as a valid justification) -------------------------
    min_window_overlap_days: int = 3  # a note's window must cover >= 3 of the week's 7 days
    step_change_horizon_weeks: int = 8  # a "starting this week" note explains rises for this long
    magnitude_tolerance_pp: float = 5.0  # slack over a note's quoted range before we caveat it

    # --- retrieval ----------------------------------------------------------------------
    retrieval_top_k: int = 5
    proximity_days: int = 28  # an unexplained row only mentions notes this close in time
    recency_scale_days: int = 60  # time-proximity score decays to 0 over this many days


CONFIG = Config()
