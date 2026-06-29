"""
Build a DR.Kernel-format validation parquet from a KernelBench level split.

This mirrors the format of `hkust-nlp/drkernel-validation-data` (which is the
KernelBench *level 2* tasks wrapped in DR.Kernel's evaluation prompt template).
By default this script builds the *level 1* equivalent, but `--level` lets you
target any level.

The output schema matches the existing validation parquet exactly:

    data_source : string                 = "kernelbench_level<N>_validation"
    prompt      : list<struct{role,content}>
                  = [{role: "user", content: PREFIX + arch_code + SUFFIX}]
    reward_model: struct{style, ground_truth}
                  = {"style": "rule", "ground_truth": arch_code}
    ability     : string                 = "kernel_optimization"
    extra_info  : struct{difficulty, name, problem_id}
                  = {difficulty: None, name: <kernelbench name>,
                     problem_id: <kernelbench problem_id>}

Rows are sorted by `name` (lexicographic), which matches the row order in
hkust-nlp/drkernel-validation-data.

PREFIX and SUFFIX are loaded verbatim (with all trailing whitespace) from
sibling files `validation_prompt_prefix.txt` / `validation_prompt_suffix.txt`,
which were captured from row 0 of `drkernel-validation-data`. Loading from
disk (vs. embedding in source) keeps the trailing-space-after-period and
the trailing-spaces-on-blank-lines that appear in the original template
byte-perfect.

Pipeline:
    1. python build_validation_from_kernelbench.py \
         --output /path/validation_level1.parquet --level 1
    2. python -m recipe.drkernel.scripts.data.convert_dataset_to_npu \
         --input  /path/validation_level1.parquet \
         --output /path/validation_level1_npu.parquet

Step 1 produces the CUDA/Triton-flavored validation set; step 2 rewrites it
to Triton-Ascend for NPU evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_PROMPT_PREFIX = (_HERE / "validation_prompt_prefix.txt").read_text(encoding="utf-8")
_PROMPT_SUFFIX = (_HERE / "validation_prompt_suffix.txt").read_text(encoding="utf-8")


def _build_user_content(arch_code: str) -> str:
    return f"{_PROMPT_PREFIX}{arch_code}{_PROMPT_SUFFIX}"


def _build_row(arch_code: str, name: str, problem_id: int, level: int) -> dict:
    return {
        "data_source": f"kernelbench_level{level}_validation",
        "prompt": [{"role": "user", "content": _build_user_content(arch_code)}],
        "reward_model": {"style": "rule", "ground_truth": arch_code},
        "ability": "kernel_optimization",
        "extra_info": {
            "difficulty": None,
            "name": name,
            "problem_id": int(problem_id),
        },
    }


def _load_kernelbench(level: int, source: str) -> pd.DataFrame:
    """Load the requested KernelBench level as a DataFrame with columns
    [code, level, name, problem_id]. `source` is either a HuggingFace dataset
    id (default) or a path to a local parquet file/directory."""
    p = Path(source)
    if p.exists():
        if p.is_dir():
            cand = list(p.glob(f"level_{level}*.parquet"))
            if not cand:
                cand = list(p.glob("*.parquet"))
            if not cand:
                raise FileNotFoundError(f"no parquet under {p}")
            df = pd.read_parquet(cand[0])
        else:
            df = pd.read_parquet(p)
        if "level" in df.columns:
            df = df[df["level"] == level].copy()
        return df

    from datasets import load_dataset
    ds = load_dataset(source, split=f"level_{level}")
    return ds.to_pandas()


def build(level: int, source: str) -> pd.DataFrame:
    kb = _load_kernelbench(level, source)
    missing = {"code", "name", "problem_id"} - set(kb.columns)
    if missing:
        raise KeyError(
            f"KernelBench rows missing columns {missing}; got {list(kb.columns)}"
        )

    # Match hkust-nlp/drkernel-validation-data row order: sort by name string.
    kb = kb.sort_values("name", kind="stable").reset_index(drop=True)

    rows = [
        _build_row(r["code"], r["name"], r["problem_id"], level)
        for _, r in kb.iterrows()
    ]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path,
                    help="Destination parquet (DR.Kernel-format validation).")
    ap.add_argument("--level", type=int, default=1,
                    help="KernelBench level to wrap (default: 1).")
    ap.add_argument(
        "--source",
        default="ScalingIntelligence/KernelBench",
        help="HF dataset id or local parquet/dir (default: HF "
             "ScalingIntelligence/KernelBench).",
    )
    ap.add_argument("--preview", action="store_true",
                    help="Print row 0 prompt+ground_truth and exit without writing.")
    args = ap.parse_args()

    df = build(args.level, args.source)
    print(f"[build] {len(df)} rows for level {args.level}")

    if args.preview:
        r = df.iloc[0]
        print(f"\n=== data_source ===\n{r['data_source']}")
        print(f"\n=== extra_info ===\n{r['extra_info']}")
        print(f"\n=== prompt[0].content (first 800 chars) ===\n{r['prompt'][0]['content'][:800]}")
        print(f"\n=== prompt[0].content (last 800 chars) ===\n{r['prompt'][0]['content'][-800:]}")
        print(f"\n=== ground_truth ===\n{r['reward_model']['ground_truth']}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"[save] wrote {len(df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
