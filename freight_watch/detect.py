"""Decide which route-weeks look out of the ordinary. Pure numbers, no notes involved."""
import pandas as pd

from .config import CONFIG, Config


def flag_candidates(metrics: pd.DataFrame, cfg: Config = CONFIG) -> pd.DataFrame:
    """Add `is_candidate` and `trigger` columns.

    A route-week is a candidate when cost is rising vs its own history AND at least one holds:
      strong_own   - own_pct >= own_strong_pct
      strong_peer  - peer_excess_pp >= peer_strong_pp   (pulling away from similar routes)
      both_moderate- own_pct >= own_moderate_pct AND peer_excess_pp >= peer_moderate_pp
    """
    m = metrics.copy()
    excess = m["peer_excess_pp"].fillna(float("-inf"))
    enough_history = m["n_prior_weeks"] >= cfg.min_history_weeks
    rising = (m["own_pct"] > 0) & enough_history

    strong_own = m["own_pct"] >= cfg.own_strong_pct
    strong_peer = excess >= cfg.peer_strong_pp
    both_moderate = (m["own_pct"] >= cfg.own_moderate_pct) & (excess >= cfg.peer_moderate_pp)

    m["is_candidate"] = rising & (strong_own | strong_peer | both_moderate)

    def label(i):
        parts = []
        if strong_own[i]:
            parts.append(f"own>={cfg.own_strong_pct:g}%")
        if strong_peer[i]:
            parts.append(f"peer_excess>={cfg.peer_strong_pp:g}pp")
        if both_moderate[i] and not parts:
            parts.append(f"own>={cfg.own_moderate_pct:g}%&peer_excess>={cfg.peer_moderate_pp:g}pp")
        return "+".join(parts)

    m["trigger"] = [label(i) if m.at[i, "is_candidate"] else "" for i in m.index]
    return m
