#!/usr/bin/env python3
"""Evaluation harness - run after `python run.py`:   python eval/run_eval.py

Checks (each prints PASS/FAIL, exit code 1 if any fails):
  1. independent recomputation of every weekly number with plain Python (no pandas)
  2. the three example rows in sample_output_format_v2.csv are reproduced (numbers + verdict + note id)
  3. hand-labelled route-weeks (eval/labeled_set.csv): flagged?, verdict, cited note
  4. adversarial grounding traps (wrong route / wrong window / negated note / ...) never become "justified"
  5. note-effect classification vs hand labels for the 10 given notes
  6. fabrication audit: every justified row re-checked against hand-read note windows
"""
import csv
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from adversarial_cases import NOTE_EFFECT_LABELS, run_adversarial  # noqa: E402
from freight_watch.notes import load_notes  # noqa: E402

DATA, OUT = ROOT / "data", ROOT / "outputs"
# Windows as a human reads them from context_notes.csv (used ONLY to audit justified rows).
HUMAN_WINDOWS = {
    "N001": (date(2025, 2, 24), date(2025, 3, 8)),     # "from Feb 24 to Mar 8"
    "N002": (date(2025, 1, 20), date(2025, 1, 26)),    # "a regional festival week" dated 2025-01-20
    "N003": (date(2025, 5, 5), date(2099, 1, 1)),      # "starting this week" - open ended
}
lines, failures = [], []


def check(name, ok, detail=""):
    lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def monday(d):
    return d - timedelta(days=d.weekday())


# ---------------------------------------------------------------- 1. independent recomputation
def recompute():
    cost, tk = defaultdict(float), defaultdict(float)
    rtype = {}
    with open(DATA / "shipment_records.csv", newline="") as f:
        for r in csv.DictReader(f):
            d = monday(date.fromisoformat(r["shipment_date"]))
            route = f"{r['origin']}-{r['destination']}"
            rtype[route] = r["route_type"]
            cost[(route, d)] += float(r["freight_cost_inr"])
            tk[(route, d)] += float(r["quantity_tonnes"]) * float(r["distance_km"])
    val = {k: cost[k] / tk[k] for k in cost}
    weeks = defaultdict(list)
    for (route, d) in sorted(val):
        weeks[route].append(d)
    out = {}
    for route, ds in weeks.items():
        for i, d in enumerate(ds):
            prior = [val[(route, p)] for p in ds[max(0, i - 8):i]]
            own = (val[(route, d)] / (sum(prior) / len(prior)) - 1) * 100 if prior else None
            peers = [val[(o, d)] for o in weeks if o != route and rtype[o] == rtype[route] and (o, d) in val]
            peer = (val[(route, d)] / (sum(peers) / len(peers)) - 1) * 100 if peers else None
            out[(route, d)] = (val[(route, d)], own, peer)
    return out


ref = recompute()
pipe = {}
with open(OUT / "weekly_metrics_all.csv", newline="") as f:
    for r in csv.DictReader(f):
        pipe[(r["route"], date.fromisoformat(r["week_of"]))] = r
worst = 0.0
for key, (c, own, peer) in ref.items():
    p = pipe[key]
    worst = max(worst, abs(float(p["cost_per_tonne_km"]) - c))
    if own is not None:
        worst = max(worst, abs(float(p["own_pct"]) - own) / 100)
    if peer is not None:
        worst = max(worst, abs(float(p["peer_pct"]) - peer) / 100)
check("1. independent recomputation matches pipeline for every route-week",
      len(ref) == len(pipe) and worst < 1e-3, f"{len(ref)} route-weeks, worst relative diff {worst:.1e}")

# ---------------------------------------------------------------- 2. the brief's example rows
with open(OUT / "flagged_routes.csv", newline="") as f:
    flagged = {(r["route"], r["week_of"]): r for r in csv.DictReader(f)}
with open(DATA / "sample_output_format_v2.csv", newline="") as f:
    for s in csv.DictReader(f):
        got = flagged.get((s["route"], s["week_of"]))
        cols = ["cost_per_tonne_km", "vs_own_history", "vs_similar_routes", "flagged", "matched_note_id"]
        same = got is not None and all(got[c] == s[c] for c in cols)
        check(f"2. example row {s['route']} {s['week_of']} reproduced exactly (numbers, flag, note id)", same)

# ---------------------------------------------------------------- 3. hand-labelled set
tp = fn = tn = fp = verdict_ok = note_ok = n_pos = 0
with open(ROOT / "eval" / "labeled_set.csv", newline="") as f:
    for c in csv.DictReader(f):
        p = pipe[(c["route"], date.fromisoformat(c["week_of"]))]
        is_flag = p["is_candidate"] == "True"
        if c["expect_flagged"] == "yes":
            n_pos += 1
            tp += is_flag
            fn += not is_flag
            verdict_ok += is_flag and p["verdict"] == c["expect_verdict"]
            note_ok += is_flag and p["matched_note_id"] == c["expect_note_id"]
        else:
            tn += not is_flag
            fp += is_flag
        ok = (is_flag == (c["expect_flagged"] == "yes")) and (
            c["expect_flagged"] == "no" or (p["verdict"] == c["expect_verdict"] and p["matched_note_id"] == c["expect_note_id"]))
        check(f"3. {c['case_id']} {c['route']} {c['week_of']}: expect flagged={c['expect_flagged']} "
              f"{c['expect_verdict']} {c['expect_note_id']}".strip(), ok)
labeled = {(c["route"], c["week_of"]) for c in csv.DictReader(open(ROOT / "eval" / "labeled_set.csv"))}
extra = [k for k in flagged if k not in labeled]
check("3. labelled set: recall / specificity / verdict accuracy / note-id accuracy all 100%",
      fn == 0 and fp == 0 and verdict_ok == n_pos and note_ok == n_pos,
      f"recall {tp}/{n_pos}, specificity {tn}/{tn + fp}, verdict {verdict_ok}/{n_pos}, note id {note_ok}/{n_pos}")
lines.append(f"     (info) {len(extra)} flagged rows are NOT in the labelled set - eyeball them: "
             + ", ".join(f"{r} {w}" for r, w in sorted(extra, key=lambda x: (x[1], x[0]))))

# ---------------------------------------------------------------- 4. adversarial grounding traps
adv = run_adversarial()
for cid, ok, desc, got in adv:
    check(f"4. {cid} {desc}", ok, f"matched={got or 'none'}")

# ---------------------------------------------------------------- 5. note-effect labels
for n in load_notes(DATA / "context_notes.csv"):
    check(f"5. effect of {n.note_id} classified as '{NOTE_EFFECT_LABELS[n.note_id]}'", n.effect == NOTE_EFFECT_LABELS[n.note_id],
          f"got {n.effect}")

# ---------------------------------------------------------------- 6. fabrication audit of justified rows
notes = {n.note_id: n for n in load_notes(DATA / "context_notes.csv")}
justified = [r for r in flagged.values() if r["flagged"] != "Yes"]
for r in justified:
    nid, wk = r["matched_note_id"], date.fromisoformat(r["week_of"])
    n = notes.get(nid)
    win = HUMAN_WINDOWS.get(nid)
    problems = []
    if n is None or win is None:
        problems.append("cited note unknown / has no hand-read window")
    else:
        if n.applies_to not in (r["route"], "All Routes"):
            problems.append("wrong route")
        if NOTE_EFFECT_LABELS[nid] != "increase":
            problems.append("note does not describe a rise")
        overlap = (min(wk + timedelta(days=6), win[1]) - max(wk, win[0])).days + 1
        if overlap < 3:
            problems.append(f"week overlaps the note window by only {max(overlap, 0)} day(s)")
        if nid not in r["reason"] or str(n.date) not in r["reason"]:
            problems.append("reason text does not name the note and its date")
    check(f"6. audit justified row {r['route']} {r['week_of']} -> {nid}", not problems, "; ".join(problems))
for r in flagged.values():
    if r["flagged"] == "Yes" and r["matched_note_id"]:
        check(f"6. unexplained row {r['route']} {r['week_of']} has blank matched_note_id", False)

lines.append("")
lines.append(f"SUMMARY: {len(lines) - lines.count('') - sum(1 for l in lines if l.startswith('     ')) - len(failures)} passed, "
             f"{len(failures)} failed")
report = "\n".join(lines)
print(report)
(OUT / "eval_report.txt").write_text(report + "\n", encoding="utf-8")
sys.exit(1 if failures else 0)
