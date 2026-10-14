# 10-minute walkthrough - talk track

**0:00 - Problem in one line (30 s).** Spot routes whose cost per tonne-km is rising unusually, and only call a rise "justified" if a note truly explains it.

**0:30 - Numbers first (2 min).** Weekly ratio-of-sums per route; two baselines exactly as specified (8 strictly-prior weeks / other routes of the same type).
Proof they are right: my numbers reproduce the brief's three example rows to the decimal, and a second pandas-free implementation matches all 728 route-weeks.

**2:30 - The flag rule and why (2 min).** Rising AND (>=8% vs history OR >=15 pp peer-gap excess OR 6%/10 pp on both). Show the Mumbai-Pune point: it is *always* ~+11% above its only peer, so raw "vs peers" would
alarm forever - I flag movement away from its usual gap. That is also what catches the slow creep. Say plainly: thresholds are my choice, tuned to this noise level (sigma ~3.5%).

**4:30 - Retrieval + verification (3 min).** Open `outputs/evidence_trace.csv`. TF-IDF ranks; `verify()` decides (scope, effect, window, coherence). Walk the Chennai-Bangalore flood:
Feb 24 and Mar 3 justified by N001; Mar 10 and Mar 17 **unexplained** because the window ended and N009 says "back to normal". Then Mumbai-Pune vs N006/N010: nearest notes say nothing was wrong -> unexplained.
Principle: a wrong confident reason costs more than "couldn't find one".

**7:30 - Trust and cost (1.5 min).** `python eval/run_eval.py` (49 checks), adversarial traps, mutation tests, 3-process reproducibility diff. Default run = 0 LLM calls; optional open-source rewriter
is guarded so it can't change verdict/note/numbers (stub-tested, not live-tested - say so).

**9:00 - Limits (1 min).** Single-peer noise on Short/Long routes (2 probable false alarms), regex note parser (fails safe), my assumed 8-week horizon / 3-day overlap, labels are mine.

## Questions to expect
- *Why not an LLM for everything?* Verdict needs to be reproducible and auditable; arithmetic + rules are exact, LLM only for wording.
- *Why TF-IDF, not embeddings?* Deterministic, offline, 10 notes; swap point is `NoteIndex`.
- *What if a new note says "costs eased"?* Effect = unknown -> can't justify -> unexplained (fails safe).
- *How would you scale?* Vector DB + metadata filter (route, date) before ranking; LLM extractor to turn notes into structured (scope, window, effect) once at index time.
- *Biggest weakness?* Threshold tuning on the same data; I'd backtest on a second period / seed synthetic anomalies to measure recall.
