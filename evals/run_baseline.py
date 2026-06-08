"""Run a full eval suite (HumanEval + MBPP) on a model and save a combined report.

Usage:
  python -m evals.run_baseline --model Qwen/Qwen2.5-Coder-0.5B-Instruct --device mps --limit 20
  python -m evals.run_baseline --model ./checkpoints/coding-model-001/step-10000 --device cuda
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evals import humaneval, mbpp


SUITE = ["humaneval", "mbpp"]


def run_suite(model: str, device: str, limit: int | None, instruct: bool, results_dir: Path) -> dict:
    results = {"model": model, "device": device, "limit": limit, "instruct": instruct, "evals": {}}
    tag = model.replace("/", "__")
    for name in SUITE:
        out = results_dir / f"{name}__{tag}.jsonl"
        print(f"\n=== {name} ===")
        if name == "humaneval":
            r = humaneval.run(model, device=device, limit=limit, instruct=instruct, out_path=out)
        else:
            r = mbpp.run(model, device=device, limit=limit, instruct=instruct, out_path=out)
        results["evals"][name] = r
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap problems per eval (useful for quick smoke tests)")
    ap.add_argument("--no-instruct", action="store_true")
    ap.add_argument("--results-dir", type=Path,
                    default=Path(__file__).parent / "results")
    args = ap.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    res = run_suite(args.model, args.device, args.limit, not args.no_instruct, args.results_dir)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    tag = args.model.replace("/", "__")
    out_file = args.results_dir / f"report__{tag}__{stamp}.json"
    out_file.write_text(json.dumps(res, indent=2))
    print(f"\nSaved report → {out_file}")
    print(json.dumps({k: v for k, v in res["evals"].items()}, indent=2))


if __name__ == "__main__":
    main()
