# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""
Convert a KernelBench-CUDA DR.Kernel parquet dataset to NPU.

Rewrites:
  - `prompt[*].content`     — model-facing instructions/examples
  - `reward_model.ground_truth` — reference torch/Python code

Substitutions cover both code (`.cuda(...)`, `is_cuda`, `torch.cuda.*`,
`device='cuda'`, `'cuda'` device strings) and prose (`CUDA` → `NPU`).
For ground_truth, also injects `import torch_npu` after the first
`import torch` line so `.npu()` resolves.

Usage:
    python -m recipe.drkernel.scripts.data.convert_dataset_to_npu \
        --input  /data/nfs/ahmad/dataset/thinking/training_data_thinking.parquet \
        --output /data/nfs/ahmad/dataset/thinking/training_data_thinking_npu.parquet \
        [--rewrite-gpu]   # also flip 'GPU' -> 'NPU' in prose
        [--preview]       # print before/after of row 0 and skip writing
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _build_subs(rewrite_gpu: bool):
    """Each entry: (compiled_regex, replacement_or_callable). Order matters:
    most specific first so e.g. `torch.cuda.X` is rewritten before bare
    `CUDA` matches anything inside it."""
    subs = []

    # --- code-level (case-sensitive) ---

    # tensor.cuda() and tensor.cuda(device=..., non_blocking=True)
    subs.append((re.compile(r"\.cuda\(\s*\)"), ".npu()"))
    subs.append((re.compile(r"\.cuda\("), ".npu("))

    # `.is_cuda` attribute check; torch_npu adds `.is_npu`.
    subs.append((re.compile(r"\bis_cuda\b"), "is_npu"))

    # `torch.cuda.X` -> `torch.npu.X` (requires `import torch_npu` so that
    # the npu submodule is registered).
    subs.append((re.compile(r"\btorch\.cuda\b"), "torch.npu"))

    # `device='cuda'` / `device="cuda"` and bare `'cuda'` / `"cuda"` strings.
    # Use a function so we keep the same quote style.
    subs.append((
        re.compile(r"(['\"])cuda\1"),
        lambda m: f"{m.group(1)}npu{m.group(1)}",
    ))

    # --- prose-level ---

    # Standalone `CUDA` in instructions, asserts, comments.
    subs.append((re.compile(r"\bCUDA\b"), "NPU"))

    # Capital-T `Triton` in prose ("custom Triton kernels", "Triton operators")
    # -> "Triton-Ascend" so the model knows the backend target. Negative
    # lookahead `(?!-Ascend)` keeps the substitution idempotent across reruns.
    # Lowercase `triton` in code (`import triton`, `triton.jit`, `tl.X`) is
    # untouched — Triton-Ascend re-uses the upstream `triton` Python package
    # name; renaming it would break every example.
    subs.append((re.compile(r"\bTriton\b(?!-Ascend)"), "Triton-Ascend"))

    # Optional: flip 'GPU' -> 'NPU' (DR.Kernel prompts use 'GPU' generically;
    # some users prefer to keep that as a generic accelerator term).
    if rewrite_gpu:
        subs.append((re.compile(r"\bGPU\b"), "NPU"))

    return subs


def _apply_subs(text: str, subs) -> str:
    out = text
    for pat, repl in subs:
        out = pat.sub(repl, out)
    return out


def _ensure_torch_npu_import(text: str) -> str:
    """Inject `import torch_npu` after **every** standalone `import torch`
    line. Used for both single code snippets (ground_truth, one `import
    torch`) and multi-code-block prompts (3+ `import torch` blocks, each
    needs its own torch_npu so the model imitates the pattern in its
    output). Idempotent: skips lines already followed by `import torch_npu`."""
    # Match `import torch` and `import torch as X`, NOT `import torch.nn`,
    # `import torch_npu`, `from torch import ...`.
    pattern = re.compile(r"^(?P<indent>[ \t]*)import torch(?P<as>\s+as\s+\w+)?$", re.MULTILINE)
    out_chunks = []
    last = 0
    for m in pattern.finditer(text):
        out_chunks.append(text[last:m.end()])
        # peek at the rest of the string for an immediate torch_npu
        tail = text[m.end():m.end() + 64]
        if not re.match(r"\s*\nimport torch_npu\b", tail):
            out_chunks.append(f"\n{m.group('indent')}import torch_npu")
        last = m.end()
    out_chunks.append(text[last:])
    return "".join(out_chunks)


def _transform_prompt(prompt, subs):
    """`prompt` is a list/ndarray of message dicts. Return a plain Python
    list of dicts with content rewritten and `import torch_npu` injected
    after each `import torch` in the embedded code blocks (so the model
    sees the NPU import pattern and reproduces it in its generated code)."""
    out = []
    for msg in prompt:
        if isinstance(msg, np.ndarray):
            msg = msg.item() if msg.shape == () else dict(msg)
        new_msg = dict(msg)
        content = new_msg.get("content")
        if isinstance(content, str):
            content = _apply_subs(content, subs)
            content = _ensure_torch_npu_import(content)
            new_msg["content"] = content
        out.append(new_msg)
    return out


def _transform_reward_model(reward_model, subs):
    if reward_model is None:
        return reward_model
    rm = dict(reward_model)
    gt = rm.get("ground_truth")
    if isinstance(gt, str):
        gt = _apply_subs(gt, subs)
        gt = _ensure_torch_npu_import(gt)
        rm["ground_truth"] = gt
    return rm


def _transform_extra_info(extra_info, subs):
    """Best-effort: rewrite any string fields. extra_info often carries
    `entry_point` (kept verbatim — typically 'ModelNew') and free-form
    metadata. Only rewrites strings; leaves other types alone."""
    if extra_info is None:
        return extra_info
    if isinstance(extra_info, dict):
        out = {}
        for k, v in extra_info.items():
            if isinstance(v, str):
                out[k] = _apply_subs(v, subs)
            else:
                out[k] = v
        return out
    return extra_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--rewrite-gpu", action="store_true",
        help="Also rewrite 'GPU' -> 'NPU' in prose. Off by default.",
    )
    ap.add_argument(
        "--preview", action="store_true",
        help="Show before/after for row 0 and exit without writing.",
    )
    args = ap.parse_args()

    subs = _build_subs(rewrite_gpu=args.rewrite_gpu)

    df = pd.read_parquet(args.input)
    print(f"[load] {len(df)} rows from {args.input}")
    print(f"[load] columns: {list(df.columns)}")

    if args.preview:
        row = df.iloc[0]
        if "prompt" in df.columns:
            for i, (msg_before, msg_after) in enumerate(
                zip(row["prompt"], _transform_prompt(row["prompt"], subs))
            ):
                role = msg_before.get("role", "?") if isinstance(msg_before, dict) else "?"
                before = msg_before.get("content", "") if isinstance(msg_before, dict) else ""
                after = msg_after.get("content", "") if isinstance(msg_after, dict) else ""
                print(f"\n=== prompt[{i}] role={role} BEFORE ===")
                print(before)
                print(f"\n=== prompt[{i}] role={role} AFTER ===")
                print(after)
        if "reward_model" in df.columns:
            rm = row["reward_model"]
            gt_before = rm.get("ground_truth", "") if isinstance(rm, dict) else ""
            rm_after = _transform_reward_model(rm, subs) or {}
            gt_after = rm_after.get("ground_truth", "") if isinstance(rm_after, dict) else ""
            print("\n=== ground_truth BEFORE ===")
            print(gt_before)
            print("\n=== ground_truth AFTER ===")
            print(gt_after)
        if "extra_info" in df.columns:
            ei = row["extra_info"]
            ei_after = _transform_extra_info(ei, subs)
            print("\n=== extra_info BEFORE ===")
            print(ei)
            print("\n=== extra_info AFTER ===")
            print(ei_after)
        return

    if "prompt" in df.columns:
        df["prompt"] = df["prompt"].apply(lambda p: _transform_prompt(p, subs))
    if "reward_model" in df.columns:
        df["reward_model"] = df["reward_model"].apply(
            lambda rm: _transform_reward_model(rm, subs)
        )
    if "extra_info" in df.columns:
        df["extra_info"] = df["extra_info"].apply(
            lambda ei: _transform_extra_info(ei, subs)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"[save] wrote {len(df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
