

# Freight Cost Watch - FreightTiger AI Intern case study
Author: <Mrigank Abhishek> 

An assistant that watches cost per tonne-km on every route, flags rises that look out of the ordinary,
and checks `context_notes.csv` for a **real, applicable** reason - citing the exact note, or admitting it found none.

**Result on the supplied data:** 7 routes x 104 weeks = 728 route-weeks -> **20 flagged: 4 justified (cited note), 16 unexplained.**
Default run: **0 LLM calls, 0 tokens, $0, ~0.1 s, fully deterministic.**

## Run it

```bash
pip install -r requirements.txt          # pandas, numpy, scikit-learn
python run.py                            # writes outputs/ (see below)
python eval/run_eval.py                  # 49 checks: labelled set, traps, independent recomputation, audit
python -m unittest discover tests        # 14 unit tests
python scripts/check_reproducibility.py  # 3 separate processes, different hash seeds, diffed
python run.py --ask "why did Chennai-Bangalore get pricier in March 2025?"   # stretch goal
```

| Output file | What it is |
|---|---|
| `outputs/flagged_routes.csv` | **The graded file.** Columns/order identical to `sample_output_format_v2.csv` (checked at runtime). |
| `outputs/weekly_metrics_all.csv` | Every route-week with all numbers, flag trigger and verdict (the audit trail). |
| `outputs/evidence_trace.csv` | For each flagged row: the retrieved notes, their scores and why each was accepted/rejected. |
| `outputs/token_cost_log.json` | Token / call log of the run (zeros: no LLM in the default path). |
| `outputs/token_cost_log_llm_estimate.json` | Estimate if the optional LLM re-wording step were switched on. |
| `outputs/repro_check.txt`, `outputs/eval_report.txt` | Reproducibility diff and eval report. |

> **Assumption:** like the sample file, `flagged_routes.csv` lists only route-weeks the assistant flagged
> (`Yes` or `No (justified)`). Normal weeks are not listed there; they are all in `weekly_metrics_all.csv`.

## How it works

```
shipments.csv -> weekly cost/tonne-km -> two baselines -> flag rule -> retrieve notes -> strict verification -> reason + CSV
                     (metrics.py)         (metrics.py)    (detect.py)   (notes.py)       (grounding.py)       (explain.py)
```

1. **Weekly metric.** `week_of` = Monday of the Mon-Sun week. `cost_per_tonne_km = sum(cost) / sum(tonnes x km)` per route-week
   (ratio of sums, not mean of per-shipment ratios).
2. **Baselines, exactly as specified.**
   *vs own history* = mean of the route's weekly values over the (up to) 8 **strictly prior** weeks - no look-ahead.
   For the first weeks of a route (fewer than 8 prior weeks exist: 56 of 728 route-weeks) it uses **all prior weeks available**, no padding;
   a route's very first week has no history and cannot be flagged. *vs similar routes* = mean of the **other** routes with the same
   `route_type` in the same week (self excluded). These reproduce the three example rows in the brief to the decimal
   (4.17 / +35.5% / +21.0%, 3.29 / +29.5% / +22.5%, 3.98 / +9.2% / +23.6%) - verified by `eval/run_eval.py`.
3. **Flag rule** (my design - the brief leaves it open). A route-week is flagged when cost is **rising** vs its own history **and** one of:
   - own rise >= **8%** (weekly noise has a robust sigma of about 3.5%, so 8% is about 2.3 sigma), or
   - **peer-gap excess** >= **15 pp**, or
   - own rise >= 6% **and** peer-gap excess >= 10 pp (moderate evidence on both signals).

   *Peer-gap excess* = this week's `vs similar routes` minus the route's own usual gap to its peers (median of its prior weeks).
   Needed because some routes are structurally dearer than their peers: Mumbai-Pune normally sits about +11% above Delhi-Jaipur,
   so a raw "+12% vs peers" is normal, not an alarm. This is also how the slow Mumbai-Pune creep is caught - it only shows as 5-8% against a trailing
   average that is itself rising, but as +25-39 pp against peers. All thresholds are in `freight_watch/config.py`.
4. **Retrieval (the RAG part).** All notes are indexed as TF-IDF vectors (uni+bi-grams, cosine similarity). For each flagged row the notes are ranked by
   `text similarity + scope match (this route or "All Routes") + closeness in time`. TF-IDF rather than a neural embedding: no model download,
   no randomness, bit-for-bit reproducible, and with 10 notes retrieval quality is not the bottleneck - verification is. The index class is the only thing to swap for
   sentence-transformers + a vector DB at scale.
5. **Verification - the hallucination guard.** Retrieval only *ranks*. A note may justify a flag only if **all four** hold (`grounding.verify`):
   1. **scope** - it is for this route or "All Routes";
   2. **effect** - it actually states a cost *increase* (negations like "costs were not significantly affected", "no major disruptions",
      "returned to normal", "without a rate change", "not part of this dataset" always lose; notes that say nothing about cost = `unknown` = cannot justify);
   3. **window** - its effective window overlaps >= 3 of the week's 7 days. Windows are parsed from the text: explicit range ("from Feb 24 to Mar 8"),
      "starting this week" (treated as explaining rises for 8 weeks - my assumption, stated in the reason text), otherwise the note's own week;
   4. **coherence** - an "All Routes" note cannot explain a route that is pulling away from its peers by >= 10 pp (a network-wide shock moves peers too).

   No note passes -> **unexplained**, `matched_note_id` blank, and the reason names the nearest in-scope notes and why they were rejected.
6. **Reasons.** Built from templates around the **verbatim first sentence of the matched note**, its date, its window and the week - nothing is generated
   freehand, so nothing can be invented. If the observed rise is well above a magnitude the note quotes (N003 says "roughly 5-7%"), the reason says so.
7. **Optional LLM step (`--llm live`).** Re-words an already-decided reason with an open-source model through any OpenAI-compatible endpoint (default: local Ollama),
   `temperature=0`, `seed=0`, `max_tokens=120`. The verdict and note id are fixed before the model sees anything, and its output is discarded in favour of the
   template if it adds a note id or a number, drops the cited id, or flips the verdict (`llm.is_faithful`). **Honest status:** this path is unit-tested
   against a stub model (good rewrite used, bad rewrite rejected, network failure falls back); I did **not** run it against a live model in my build
   environment. The graded outputs come from the deterministic path.
8. **Stretch: Q&A.** `--ask "..."` parses route and month/year from the question and answers by quoting rows of the result files
   (including the cited note and reason). No generation, so it cannot contradict the findings. It answers "why" only as far as the findings do.

## Findings

| Route / weeks | Verdict | Why |
|---|---|---|
| Delhi-Jaipur 2024-11-11, 11-18 (+35%, +28%) | unexplained | no note covers Delhi-Jaipur in Nov 2024 (N007 is May 2024 and mentions no cost rise) |
| Ahmedabad-Mumbai 2025-01-20 | justified **N002** | festival-week surcharge on that corridor, dated that week |
| Chennai-Bangalore 2025-02-24, 03-03 | justified **N001** | flooding window Feb 24 - Mar 8 |
| Chennai-Bangalore 2025-03-10, 03-17 | **unexplained** | N001's window ended (roads passable Mar 9) yet cost is still +22% / +14%; N009 says the route is back to normal |
| Mumbai-Delhi 2025-05-12 | justified **N003** | nationwide diesel rise from 2025-05-05 |
| Mumbai-Pune, 10 weeks Jun-Dec 2025 | unexplained | steady creep, gap to peers grows from about +11% to +39%; nearest notes (N006 stable demand, N010 no rate change) say nothing about a rise |
| Delhi-Chennai 2024-02-05, Mumbai-Delhi 2025-06-02 | unexplained, **low confidence** | likely weekly noise (3-5 shipments per route-week); flagged conservatively - see limits |

## How I convinced myself it is right

- **Independent recomputation:** a second, pandas-free implementation (plain `csv` + `datetime`) recomputes all 728 route-weeks; worst difference 5e-5.
- **The brief's example rows** are reproduced exactly (numbers, flag, note id).
- **Hand-labelled set** (`eval/labeled_set.csv`, 19 route-weeks: 11 should be flagged with a given verdict/note, 8 ordinary or decoy weeks): 11/11 recalled, 8/8 not flagged,
  11/11 verdicts, 11/11 note ids. The labels are **mine**, read from the notes and data - there was no external ground truth - so this proves consistency with my reading, not with FreightTiger's.
  The harness also lists the 9 flagged rows outside the labelled set so they can be eyeballed.
- **Adversarial grounding traps** (`eval/adversarial_cases.py`): wrong route, window ended a day earlier, 2-day overlap, negated note, silent note, future note,
  stale note, all-routes note on an outlier route - none can become "justified"; two control cases must be, so the guard is not just refusing everything.
- **Fabrication audit:** each justified row is re-checked against note windows I read by hand (separate from the regex parser): route, window overlap, effect, and that the reason names the note and its date.
- **Mutation tests:** I broke the guard on purpose (overlap 3->1 day, removed negation handling, removed the coherence check, ignored scope) and confirmed the eval failed each time (not part of the repo - throwaway copy).

## Reproducibility

`scripts/check_reproducibility.py` runs the pipeline 3 times in **separate processes with different `PYTHONHASHSEED`s** and compares SHA-256 of every output:
all identical (`outputs/repro_check.txt`). Nothing is random: TF-IDF is deterministic, ties in ranking break on note id, floats are rounded only at write time.
If the LLM step is on, only the free-text `reason` may differ; the script also hashes the CSV without that column, which must match.

## Cost of one full run

| | calls | input tokens | output tokens | cost |
|---|---|---|---|---|
| **Default run (graded output)** | **0** | **0** | **0** | **$0** |
| Optional LLM re-wording, estimated (`--llm dry-run`, 20 flagged rows, ~4 chars/token, output estimated at draft length) | 20 | ~2,900 | ~1,400 | $0 on a self-hosted open-weights model (Ollama); use `--price-in/--price-out` for a hosted one |

The call count is bounded by the number of *flagged rows* (20), not shipments (2,940): everything else is arithmetic.

## Limits and trade-offs (please read)

- **Thresholds were set by looking at this dataset's noise level** - sensible, but tuned to it. Part of why I keep a separate labelled set and traps.
- **Short and Long route types have exactly one peer route each**, so `vs similar routes` there is noisy (one weekly average of 3-5 shipments against another). This is why two flags (Delhi-Chennai 2024-02-05, Mumbai-Delhi 2025-06-02) are probably noise rather than anomalies. I chose to surface them as "unexplained, review" rather than hide them.
- Delhi-Chennai 2024-02-05 has only 5 prior weeks, so no peer baseline; Mumbai-Pune 2025-06-23 sits right on the 10 pp coherence boundary, which is where N003 stops being accepted.
- **Note parsing is rule-based** (explicit range / "starting this week" / own week; keyword lists for effect). Unfamiliar phrasing degrades to `unknown` -> never justifies, so it fails safe, but a new note type (e.g. "costs eased") would need a rule. An LLM extractor at index time (10 notes = 10 calls, cached) is the natural upgrade.
- "Starting this week" => 8-week effect horizon and the 3-day overlap rule are **my assumptions** (conservative; both in `config.py` and disclosed in the reason text).
- A justified verdict says the note *applies*, not that it fully accounts for the size of the rise; only N003 quotes a magnitude and the check is advisory (caveat text).

## Layout

```
run.py                      CLI (run / --llm / --ask)
freight_watch/              config, metrics, detect, notes (+TF-IDF index), grounding (retrieve+verify), explain, llm, costlog, qa, report, pipeline
eval/                       run_eval.py, labeled_set.csv, adversarial_cases.py
tests/test_core.py          unit tests (week boundaries, no look-ahead, peer exclusion, parsing, LLM guardrails, determinism)
scripts/check_reproducibility.py
data/                       the three provided CSVs      outputs/   all generated files      docs/WALKTHROUGH.md   10-minute talk track
```
