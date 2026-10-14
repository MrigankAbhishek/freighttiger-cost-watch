"""End-to-end run: shipments -> weekly metrics -> candidates -> note retrieval/verification -> CSVs."""
import time
from pathlib import Path

from .config import CONFIG, Config
from .costlog import CostLog
from .detect import flag_candidates
from .explain import build_reason
from .grounding import decide
from .llm import Rewriter
from .metrics import load_shipments, weekly_metrics
from .notes import NoteIndex, load_notes
from .report import (OUTPUT_COLUMNS, format_row, read_contract_columns, write_all_metrics,
                     write_flagged_csv, write_trace)


def run_pipeline(data_dir="data", out_dir="outputs", cfg: Config = CONFIG, llm_mode="off", model="",
                 base_url="http://localhost:11434/v1", price_in=0.0, price_out=0.0, transport=None) -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    contract = read_contract_columns(data_dir / "sample_output_format_v2.csv")
    if contract != OUTPUT_COLUMNS:
        raise RuntimeError(f"output columns {OUTPUT_COLUMNS} do not match the contract {contract}")

    shipments = load_shipments(data_dir / "shipment_records.csv")
    metrics = flag_candidates(weekly_metrics(shipments, cfg), cfg)
    notes = load_notes(data_dir / "context_notes.csv", cfg)
    index = NoteIndex(notes)

    costlog = CostLog(price_in_per_million_usd=price_in, price_out_per_million_usd=price_out)
    rewriter = Rewriter(llm_mode, model or "none", costlog, base_url, transport) if llm_mode != "off" else None

    rows, trace = [], []
    metrics["verdict"], metrics["matched_note_id"] = "", ""
    for i, r in metrics[metrics["is_candidate"]].iterrows():
        verdict = decide(index, r, cfg)
        reason = build_reason(verdict, r, cfg)
        if rewriter:
            reason = rewriter.rewrite(reason)  # may only re-word; verdict / note id / numbers are fixed
        rows.append(format_row(r, verdict, reason))
        metrics.at[i, "verdict"] = "justified" if verdict.justified else "unexplained"
        metrics.at[i, "matched_note_id"] = verdict.matched.note.note_id if verdict.justified else ""
        for rank, c in enumerate(verdict.considered, start=1):
            trace.append({"route": r["route"], "week_of": r["week_of"].date().isoformat(), "rank": rank,
                          "note_id": c.note.note_id, "tfidf": c.tfidf, "scope_match": c.scope_match,
                          "gap_days": c.gap_days, "overlap_days": c.overlap_days, "score": c.score,
                          "decision": c.code})

    write_flagged_csv(rows, out_dir / "flagged_routes.csv")
    write_all_metrics(metrics, out_dir / "weekly_metrics_all.csv")
    write_trace(trace, out_dir / "evidence_trace.csv")
    cost = costlog.to_json(out_dir / "token_cost_log.json")
    return {"flagged_rows": len(rows), "justified": sum(r["flagged"] != "Yes" for r in rows),
            "unexplained": sum(r["flagged"] == "Yes" for r in rows), "route_weeks": len(metrics),
            "cost_log": cost, "seconds": round(time.time() - started, 2), "out_dir": str(out_dir)}
