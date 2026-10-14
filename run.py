#!/usr/bin/env python3
"""Command line entry point.

    python run.py                                   run the full pipeline (deterministic, 0 LLM calls)
    python run.py --llm dry-run                     also estimate tokens/cost if an LLM re-worded the reasons
    python run.py --llm live --model qwen2.5:3b     re-word reasons with a local open-source model (Ollama)
    python run.py --ask "why did Chennai-Bangalore get pricier in March 2025?"
"""
import argparse
import json
import sys
from pathlib import Path

from freight_watch.pipeline import run_pipeline
from freight_watch.qa import answer


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--llm", choices=["off", "dry-run", "live"], default="off")
    p.add_argument("--model", default="qwen2.5:3b")
    p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--price-in", type=float, default=0.0, help="USD per 1M input tokens (0 for local models)")
    p.add_argument("--price-out", type=float, default=0.0, help="USD per 1M output tokens")
    p.add_argument("--ask", help="ask a plain-English question about the findings")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    if args.ask:
        if not (out_dir / "flagged_routes.csv").exists():
            run_pipeline(args.data_dir, args.out_dir)
        print(answer(args.ask, args.out_dir))
        return 0

    summary = run_pipeline(args.data_dir, args.out_dir, llm_mode=args.llm, model=args.model,
                           base_url=args.base_url, price_in=args.price_in, price_out=args.price_out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
