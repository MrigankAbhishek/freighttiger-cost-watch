"""Stretch goal: answer plain-English questions using ONLY the pipeline's own result files.

Nothing here calls an LLM and nothing is generated freehand: every sentence is assembled from rows in
flagged_routes.csv / weekly_metrics_all.csv, so answers cannot contradict (or invent beyond) the findings.
"""
import re
from pathlib import Path

import pandas as pd

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]


def _find_month(q: str):
    for i, name in enumerate(MONTHS, start=1):
        if name == "may":
            if re.search(r"\b(?:in|of|during|for|since) may\b|\bmay (?:20\d{2})\b", q):
                return i
        elif re.search(rf"\b{name}\b|\b{name[:3]}\b", q):
            return i
    return None


def _find_routes(q: str, routes: list) -> list:
    exact = [r for r in routes if all(city.lower() in q for city in r.split("-"))]
    if exact:
        return exact
    return [r for r in routes if any(city.lower() in q for city in r.split("-"))]


def answer(question: str, out_dir="outputs") -> str:
    out_dir = Path(out_dir)
    flagged = pd.read_csv(out_dir / "flagged_routes.csv", parse_dates=["week_of"]).fillna("")
    allw = pd.read_csv(out_dir / "weekly_metrics_all.csv", parse_dates=["week_of"])
    routes = sorted(allw["route"].unique())
    q = question.lower()

    found_routes = _find_routes(q, routes)
    month = _find_month(q)
    year_m = re.search(r"\b(20\d{2})\b", q)
    year = int(year_m.group(1)) if year_m else None
    date_m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", q)

    if not found_routes and month is None and year is None and not date_m:
        return ("I can answer questions about routes and periods in my findings. Try e.g. "
                "'why did Chennai-Bangalore get pricier in March 2025?'. Routes I track: " + ", ".join(routes) + ".")

    def in_period(df):
        mask = pd.Series(True, index=df.index)
        if date_m:
            wk = pd.Timestamp(date_m.group(1))
            wk = wk - pd.Timedelta(days=wk.weekday())
            mask &= df["week_of"] == wk
        if month is not None:  # a week counts if any of its 7 days falls in the month
            mask &= (df["week_of"].dt.month == month) | ((df["week_of"] + pd.Timedelta(days=6)).dt.month == month)
        if year is not None:
            mask &= df["week_of"].dt.year == year
        return mask

    rmask = flagged["route"].isin(found_routes) if found_routes else pd.Series(True, index=flagged.index)
    hits = flagged[rmask & in_period(flagged)]
    scope = ", ".join(found_routes) if found_routes else "all routes"
    period = " ".join(x for x in [MONTHS[month - 1].title() if month else "", str(year) if year else "",
                                  date_m.group(1) if date_m else ""] if x) or "the whole period"

    if hits.empty:
        sub = allw[allw["route"].isin(found_routes) if found_routes else True]
        sub = sub[in_period(sub)]
        if sub.empty:
            return f"I have no data for {scope} in {period}."
        return (f"No week was flagged for {scope} in {period}. Average cost over those weeks was "
                f"{sub['cost_per_tonne_km'].mean():.2f} INR per tonne-km, with no unusual rise against "
                f"the route's own history or similar routes.")

    lines = [f"{len(hits)} flagged week(s) for {scope} in {period} (a week counts if any of its days is in the month):"]
    for _, h in hits.iterrows():
        lines.append(f"- {h['route']}, week of {h['week_of'].date()}: {h['cost_per_tonne_km']} INR/tonne-km "
                     f"({h['vs_own_history']}; {h['vs_similar_routes']}). {h['reason']}")
    j = int((hits["matched_note_id"] != "").sum())
    lines.append(f"Summary: {j} justified by a cited note, {len(hits) - j} unexplained.")
    return "\n".join(lines)
