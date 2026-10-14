#!/usr/bin/env python3
"""Run the pipeline 3 times as separate processes (different PYTHONHASHSEED each) and diff the results.

    python scripts/check_reproducibility.py                # default deterministic mode
    python scripts/check_reproducibility.py --llm live --model qwen2.5:3b   # LLM mode: reason wording may differ,
                                                                          # everything else must not
Writes outputs/repro_check.txt
"""
import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["flagged_routes.csv", "weekly_metrics_all.csv", "evidence_trace.csv"]


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def without_reason_sha(path) -> str:
    with open(path, newline="") as f:
        rows = [[c for k, c in r.items() if k != "reason"] for r in csv.DictReader(f)]
    return hashlib.sha256(repr(rows).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="off")
    ap.add_argument("--model", default="")
    args = ap.parse_args()

    scratch = ROOT / "outputs" / "_repro"
    results = []
    for i, seed in enumerate(["0", "1", "12345"], start=1):  # different hash seeds: no dict/set-order luck
        out = scratch / f"run{i}"
        cmd = [sys.executable, str(ROOT / "run.py"), "--data-dir", str(ROOT / "data"), "--out-dir", str(out), "--llm", args.llm]
        if args.model:
            cmd += ["--model", args.model]
        subprocess.run(cmd, check=True, capture_output=True, env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin:/usr/local/bin"}, cwd=ROOT)
        results.append(out)

    report = [f"Reproducibility check - 3 independent processes, llm mode = {args.llm}", ""]
    ok = True
    for name in FILES:
        hashes = [sha(r / name) for r in results]
        same = len(set(hashes)) == 1
        ok &= same or (name == "flagged_routes.csv" and args.llm != "off")  # LLM mode: judged by the next check
        report.append(f"{'IDENTICAL' if same else 'DIFFERENT'}  {name}  sha256={hashes[0][:16]}... x{len(set(hashes))} distinct")
    core = [without_reason_sha(r / "flagged_routes.csv") for r in results]
    same_core = len(set(core)) == 1
    ok &= same_core
    report.append(f"{'IDENTICAL' if same_core else 'DIFFERENT'}  flagged_routes.csv with the free-text `reason` column removed "
                  f"(flags, numbers, verdicts, cited note ids)")
    report += ["", "RESULT: " + ("PASS - flags, numbers, verdicts and cited notes match on all 3 runs" if ok else "FAIL")]
    text = "\n".join(report)
    print(text)
    (ROOT / "outputs" / "repro_check.txt").write_text(text + "\n", encoding="utf-8")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
