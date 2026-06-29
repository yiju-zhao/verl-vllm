"""
Append the DR.Kernel "output requirements" addendum (PART 2) to the last
message of every prompt in an NPU-flavored parquet, producing a `_v2`
dataset.

Join logic (mirrors eval_l2.py's append_addendum_to_prompt):
    msgs = list(prompt_messages_from_parquet)
    msgs[-1]["content"] = msgs[-1]["content"] + "\\n\\n" + addendum

Only the prompt is modified. `reward_model`, `data_source`, `ability`,
`extra_info`, and any other columns pass through unchanged.

The addendum text is loaded from sibling file `prompt_addendum.txt` so that
backticks and trailing whitespace are preserved without escaping.

Usage:
    python append_prompt_addendum.py \
        --input  /data/.../validation_data_thinking_level1_npu.parquet \
        --output /data/.../validation_data_thinking_level1_npu_v2.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ADDENDUM = (_HERE / "prompt_addendum.txt").read_text(encoding="utf-8")


def append_addendum_to_prompt(prompt, addendum: str):
    """Return a new list of message dicts with `addendum` appended to the
    LAST message's content, separated by '\\n\\n'. Empty/whitespace-only
    addenda are a no-op."""
    msgs = []
    for m in prompt:
        if isinstance(m, np.ndarray):
            m = m.item() if m.shape == () else dict(m)
        msgs.append(dict(m))
    if not msgs:
        return msgs
    if addendum.strip():
        msgs[-1]["content"] = msgs[-1]["content"] + "\n\n" + addendum
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--preview", action="store_true",
                    help="Show row 0 last-message tail (before/after) and exit.")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    print(f"[load] {len(df)} rows from {args.input}")
    print(f"[load] columns: {list(df.columns)}")
    print(f"[addendum] {len(_ADDENDUM)} chars")

    if args.preview:
        r = df.iloc[0]
        before = r["prompt"][-1]["content"]
        after_msgs = append_addendum_to_prompt(r["prompt"], _ADDENDUM)
        after = after_msgs[-1]["content"]
        print(f"\n=== row 0 last message: BEFORE (tail 200 chars) ===\n{before[-200:]}")
        print(f"\n=== row 0 last message: AFTER (tail 400 chars) ===\n{after[-400:]}")
        print(f"\nbefore_len={len(before)}  after_len={len(after)}  delta={len(after)-len(before)}")
        return

    df["prompt"] = df["prompt"].apply(lambda p: append_addendum_to_prompt(p, _ADDENDUM))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"[save] wrote {len(df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
