"""MBPP pass@1 runner.

MBPP = "Mostly Basic Python Problems". 974 short Python tasks with a
canonical solution and 3 test cases each. We follow the standard
sanitized split (task_ids 11-510) used in the literature.

Usage:
  python -m evals.mbpp --model Qwen/Qwen2.5-Coder-0.5B-Instruct --limit 20
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Standard MBPP prompt (matches the format used by most code-LLM papers)
PROMPT_TEMPLATE = """\
You are an expert Python programmer.

Problem:
{description}

Test cases:
{tests}

Write a Python function that solves the problem. Return ONLY the function definition, no extra commentary.

```python
"""


def _normalize_test(s: str) -> str:
    """MBPP test strings look like 'assert function_name(...) == ...'."""
    return s.strip().rstrip(",")


def build_prompt(example: dict, instruct: bool) -> str:
    tests = "\n".join(_normalize_test(t) for t in example.get("test_list", []))
    if instruct:
        return (
            f"Problem: {example['text']}\n"
            f"Function signature: {example.get('code', '').splitlines()[0] if example.get('code') else 'def solution():'}\n"
            f"Tests:\n{tests}\n\n"
            f"Return ONLY the Python function:\n```python\n"
        )
    return PROMPT_TEMPLATE.format(description=example["text"], tests=tests)


def cut_to_code_block(text: str) -> str:
    text = text.strip()
    if "```" in text:
        chunks = text.split("```")
        for j in range(1, len(chunks), 2):
            chunk = chunks[j]
            chunk = chunk.removeprefix("python").lstrip("\n")
            return chunk
    return text


def generate(model, tok, prompts, device, max_new_tokens=256, batch_size=4):
    outs = []
    pad = tok.pad_token_id or tok.eos_token_id
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inp = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            gen = model.generate(
                **inp, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=pad, eos_token_id=tok.eos_token_id,
            )
        text = gen[:, inp["input_ids"].shape[1]:]
        outs.extend(tok.batch_decode(text, skip_special_tokens=True))
    return outs


def run_one_test(code: str, test: str, entry_point: str, timeout=5) -> bool:
    """Run a single test against generated code in a subprocess."""
    full = f"{code}\n\n{test}\n"
    try:
        r = subprocess.run(
            [sys.executable, "-c", full],
            capture_output=True, timeout=timeout, text=True,
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def run(
    model_name: str,
    device: str = "auto",
    limit: int | None = None,
    instruct: bool = True,
    out_path: Path | None = None,
) -> dict:
    print(f"[mbpp] loading {model_name} on {device} …")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True)
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    print("[mbpp] loading dataset (sanitized split) …")
    ds = load_dataset("mbpp", "sanitized", split="test")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    prompts = [build_prompt(ex, instruct) for ex in ds]
    print(f"[mbpp] {len(prompts)} problems; generating …")
    raw = generate(model, tok, prompts, device)
    completions = [cut_to_code_block(r) for r in raw]

    if out_path is None:
        out_path = Path(__file__).parent / "results" / f"mbpp__{model_name.replace('/', '__')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for ex, comp, r in zip(ds, completions, raw):
            f.write(json.dumps({
                "task_id": ex["task_id"],
                "completion": comp,
                "raw": r,
                "passed": None,  # filled below
            }) + "\n")

    print("[mbpp] running test cases in subprocess …")
    passed = 0
    rows = []
    for ex, comp in zip(ds, completions):
        ok = all(
            run_one_test(comp, t, ex.get("test_setup_code", ""), timeout=5)
            for t in ex["test_list"]
        )
        passed += int(ok)
        rows.append({"task_id": ex["task_id"], "passed": ok})

    n = len(ds)
    pass1 = passed / max(n, 1)
    summary = {
        "model": model_name,
        "device": device,
        "n_problems": n,
        "pass@1": round(pass1, 4),
        "raw_results_file": str(out_path),
    }
    (out_path.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2))
    print(f"[mbpp] pass@1 = {pass1:.4f}  → {out_path.with_suffix('.summary.json')}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-instruct", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    res = run(args.model, args.device, args.limit, instruct=not args.no_instruct, out_path=args.out)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
