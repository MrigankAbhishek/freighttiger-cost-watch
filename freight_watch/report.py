"""Write the graded CSV (exact column contract) plus supporting files."""
import csv

import pandas as pd

OUTPUT_COLUMNS = ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes",
                  "flagged", "matched_note_id", "reason"]


def read_contract_columns(sample_path) -> list:
    with open(sample_path, newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def format_row(row, verdict, reason: str) -> dict:
    own = f"{row['own_pct']:+.1f}% vs this route's past average"
    peer = (f"{row['peer_pct']:+.1f}% vs similar-length routes this week"
            if pd.notna(row["peer_pct"]) else "n/a (no other route of this type this week)")
    return {
        "route": row["route"],
        "week_of": pd.Timestamp(row["week_of"]).date().isoformat(),
        "cost_per_tonne_km": f"{row['cost_per_tonne_km']:.2f}",
        "vs_own_history": own,
        "vs_similar_routes": peer,
        "flagged": "No (justified)" if verdict.justified else "Yes",
        "matched_note_id": verdict.matched.note.note_id if verdict.justified else "",
        "reason": reason,
    }


def write_flagged_csv(rows: list, path):
    rows = sorted(rows, key=lambda r: (r["week_of"], r["route"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def write_all_metrics(metrics: pd.DataFrame, path):
    out = metrics.copy()
    out["week_of"] = out["week_of"].dt.date.astype(str)
    for col, nd in [("total_tonne_km", 2), ("cost_per_tonne_km", 4), ("own_avg", 4), ("own_pct", 2),
                    ("peer_avg", 4), ("peer_pct", 2), ("peer_gap_baseline", 2), ("peer_excess_pp", 2)]:
        out[col] = out[col].round(nd)
    out.to_csv(path, index=False, lineterminator="\n")


def write_trace(trace_rows: list, path):
    pd.DataFrame(trace_rows).to_csv(path, index=False, lineterminator="\n", float_format="%.4f")
