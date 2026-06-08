"""HumanEval pass@1 runner.

Loads a HuggingFace causal-LM, generates one completion per problem using the
standard HumanEval prompt format, then scores with the `human_eval` package's
own `evaluate_functional_correctness` (which runs the test cases in a
subprocess sandbox).

Usage:
  python -m evals.humaneval --model Qwen/Qwen2.5-Coder-0.5B-Instruct --limit 10
  python -m evals.humaneval --model Qwen/Qwen2.5-Coder-0.5B-Instruct --device mps
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import torch
from human_eval.data import HUMAN_EVAL, read_problems, write_jsonl
from human_eval.execution import check_correctness
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_HEADER = (
    "Complete the following Python function. "
    "Return only the function body (no markdown, no commentary).\n\n"
)


def build_prompt(problem: dict, instruct: bool) -> str:
    """HumanEval stores the function signature + docstring; we feed it back."""
    if instruct:
        return (
            "You are an expert Python programmer. "
            "Read the function signature and docstring, then write the body.\n\n"
            f"```python\n{problem['prompt']}```\n"
        )
    return problem["prompt"]


def generate_completions(
    model,
    tokenizer,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int = 384,
    batch_size: int = 4,
) -> list[str]:
    """Greedy decode one completion per prompt. Returns raw continuation text."""
    completions: list[str] = []
    model.eval()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                num_beams=1,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )

        gen = out[:, inputs["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        completions.extend(texts)
    return completions


def cut_to_code_block(text: str) -> str:
    """Strip ```python fences and prose. Returns the most plausible Python body."""
    text = text.strip()
    if "```" in text:
        chunks = text.split("```")
        # pick the first python-looking chunk
        for j in range(1, len(chunks), 2):
            chunk = chunks[j]
            chunk = chunk.removeprefix("python").lstrip("\n")
            return chunk
    return text


def run(
    model_name: str,
    device: str = "auto",
    limit: int | None = None,
    instruct: bool = True,
    out_path: Path | None = None,
) -> dict:
    print(f"[humaneval] loading {model_name} on {device} …")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    problems = read_problems()
    if limit is not None:
        problems = dict(list(problems.items())[:limit])

    prompts = [build_prompt(p, instruct) for p in problems.values()]
    task_ids = list(problems.keys())
    print(f"[humaneval] {len(prompts)} problems; generating …")
    completions = generate_completions(model, tok, prompts, torch.device(device))

    samples = []
    for tid, raw in zip(task_ids, completions):
        body = cut_to_code_block(raw)
        # HumanEval expects the full function: prompt + body
        full = problems[tid]["prompt"] + body
        samples.append({"task_id": tid, "completion": body, "full": full})

    if out_path is None:
        out_path = Path(__file__).parent / "results" / f"humaneval__{model_name.replace('/', '__')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(out_path), [{"task_id": s["task_id"], "completion": s["completion"]} for s in samples])
    print(f"[humaneval] samples → {out_path}")

    # Score in a sandboxed subprocess (human_eval runs untrusted code)
    score_path = out_path.with_suffix(".score.json")
    print("[humaneval] scoring in subprocess sandbox …")
    subprocess.run(
        [sys.executable, "-m", "human_eval.evaluate_functional_correctness", str(out_path)],
        check=True,
        env={**os.environ, "HF_ALLOW_LOCAL": "1"},
    )
    # human_eval writes <input>.results.json beside the input
    results_file = out_path.with_name(out_path.stem + ".results.json")
    if results_file.exists():
        results = json.loads(results_file.read_text())
        pass1 = sum(1 for r in results if r.get("passed")) / max(len(results), 1)
        summary = {
            "model": model_name,
            "device": device,
            "n_problems": len(samples),
            "pass@1": round(pass1, 4),
            "raw_results_file": str(results_file),
        }
        score_path.write_text(json.dumps(summary, indent=2))
        print(f"[humaneval] pass@1 = {pass1:.4f}  → {score_path}")
        return summary
    return {"model": model_name, "error": "no results file produced"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-instruct", action="store_true",
                    help="Treat the model as a base LM (no chat template).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    res = run(args.model, args.device, args.limit, instruct=not args.no_instruct, out_path=args.out)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
