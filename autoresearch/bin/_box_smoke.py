#!/usr/bin/env python
"""CPU build-smoke for an _arq treatment stub — run by queue-daemon.sh on the box.

Loads the _arq file as a module (so its `if __name__ == "__main__"` training
block does NOT execute), reads its top-level `C` config class (the RUN-CONTRACT
requires this name), and constructs MinimalLLM(C()) on CPU. This catches the
classic failure where a flag is added to the dataclass but never threaded through
the model — in seconds, before any GPU time is spent.

Also statically guards the launch entrypoint: the stub must drive `train_llm`
(the repo's only trainer) and must NOT reference a non-existent legacy entrypoint
(`main.py`, `scripts/train.py`, bare `train.py`). That class of bug — a freeform
run command copied from idea.md — used to fail only on the GPU with
`can't open file '/root/universe-lm/main.py'`, wasting a claim. We catch it here
on CPU, before any GPU time, so the daemon bounces it to needs-recode instead.

Prints `SMOKE_OK` on success; a traceback + non-zero exit on failure.

Usage:  python _box_smoke.py _arq_157-conv-ffn.py
"""
import importlib.util
import re
import sys

# Entrypoints that do not exist in universe-lm. The only trainer is
# train_llm.py; the `\b` boundaries keep `train.py` from matching `train_llm.py`.
_FORBIDDEN_ENTRYPOINT = re.compile(r"\bmain\.py\b|scripts/train\.py|\btrain\.py\b")


def check_entrypoint(arq_path: str):
    """Return an error string if the stub's launch entrypoint is wrong, else None."""
    with open(arq_path, encoding="utf-8") as fh:
        src = fh.read()
    bad = _FORBIDDEN_ENTRYPOINT.search(src)
    if bad:
        return (
            f"references non-existent entrypoint '{bad.group(0)}' — the only "
            "trainer is train_llm.py; launch via `import train_llm; train_llm.main()`"
        )
    if "train_llm" not in src:
        return "does not reference train_llm — the __main__ block must call train_llm.main()"
    return None


def main(arq_path: str) -> int:
    ep_err = check_entrypoint(arq_path)
    if ep_err is not None:
        print(f"SMOKE_FAIL: {arq_path} {ep_err}")
        return 4

    spec = importlib.util.spec_from_file_location("arqmod", arq_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # top-level only; __main__ guard does not fire

    cfg_cls = getattr(mod, "C", None)
    if cfg_cls is None:
        print(f"SMOKE_FAIL: {arq_path} has no top-level `C` config class")
        return 3

    cfg = cfg_cls()
    from models.llm import MinimalLLM

    MinimalLLM(cfg)  # CPU construct; raises if a flag isn't threaded through
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python _box_smoke.py <_arq_file.py>")
        sys.exit(2)
    try:
        sys.exit(main(sys.argv[1]))
    except Exception as e:  # noqa: BLE001 — surface any construction error to the daemon
        import traceback

        traceback.print_exc()
        print(f"SMOKE_FAIL: {type(e).__name__}: {e}")
        sys.exit(1)
