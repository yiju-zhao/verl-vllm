"""Eval driver for KernelBench L2 (Triton-Ascend) on 100 tasks.

Three phases (selectable via --phases):
  - gen:       call vLLM, save raw_response.txt + {name}_triton_ascend_impl.py
  - verify:    parallel verify subprocess pool (8 workers, chips 1,4,5,8-12)
  - benchmark: SERIAL benchmark on chip 13 (CPU-isolation for clean speedup)
  - summary:   aggregate all result.json into summary.csv + summary.md

Usage:
  python3 eval_l2.py --model Qwen3-8B --out runs/qwen3_8b_full \\
      --addendum eval/prompt_addendum_v1.txt --phases gen,verify,benchmark,summary

For warm-up subset:
  python3 eval_l2.py --model drkernel-14b --out runs/_warmup_v1 \\
      --addendum eval/prompt_addendum_v1.txt --filter 1,25,50,75,99 --phases gen,verify

For harness smoke test (no LLM):
  python3 eval_l2.py --filter 1 --skip-gen --hand-written-impl smoke/1_impl.py \\
      --out runs/_smoke --phases verify,benchmark
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# --- paths ---
PARQUET_PATH = Path("/data/kevin/code/KernelGen/thinking-kernel/validation_data_thinking_npu.parquet")

# Default path to the AscendOpGenAgent kernel-verifier scripts. Resolution order:
#   1. --verifier-dir CLI flag (set in main())
#   2. TRITON_VERIFIER_DIR env var (read here at import time)
#   3. <repo>/external/AscendOpGenAgent/skills/triton/kernel-verifier/scripts
#      (populated by `git submodule update --init`)
# VERIFY_PY / BENCHMARK_PY are reassigned in main() after parsing args.
_DEFAULT_VERIFIER_DIR = (
    Path(os.environ["TRITON_VERIFIER_DIR"])
    if os.environ.get("TRITON_VERIFIER_DIR")
    else Path(__file__).resolve().parent.parent /
        "external" / "AscendOpGenAgent" /
        "skills" / "triton" / "kernel-verifier" / "scripts"
)
VERIFIER_DIR = _DEFAULT_VERIFIER_DIR
VERIFY_PY = VERIFIER_DIR / "verify.py"
BENCHMARK_PY = VERIFIER_DIR / "benchmark.py"

# --- chip allocation (per plan, after pre-flight) ---
VERIFY_CHIPS = ["1", "4", "5"]   # 3 workers (chips 0/2/3 hold vLLMs)
BENCHMARK_CHIP = "3"                                          # serial benchmark (chip 5 busy with drkernel-baseline vLLM)

# --- timeouts ---
VERIFY_TIMEOUT_S = 240
BENCHMARK_TIMEOUT_S = 600
GEN_TIMEOUT_S = 600

CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
CODE_BLOCK_BARE_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def extract_python_block(text: str, first: bool = False) -> str | None:
    if not text:
        return None
    matches = CODE_BLOCK_RE.findall(text)
    if not matches:
        matches = CODE_BLOCK_BARE_RE.findall(text)
    if not matches:
        return None
    return (matches[0] if first else matches[-1]).strip()


def append_addendum_to_prompt(parquet_prompt, addendum: str):
    """Build messages list = parquet's prompt + addendum suffix on last user message."""
    msgs = [dict(m) for m in parquet_prompt]
    if addendum.strip():
        msgs[-1]["content"] = msgs[-1]["content"] + "\n\n" + addendum
    return msgs


# ============================================================================
# Phase A: gen
# ============================================================================

def phase_gen(args, df, out_dir: Path, addendum: str):
    """Call vLLM for each task, save raw response and extract code."""
    from openai import OpenAI

    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

    client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY")

    print(f"[gen] {len(df)} tasks, model={args.model}, max_tokens={args.max_tokens}")
    t_total_start = time.time()
    for ix, (_, row) in enumerate(df.iterrows(), start=1):
        pid = int(row["extra_info"]["problem_id"])
        name = row["extra_info"]["name"]
        task_dir = out_dir / str(pid)
        task_dir.mkdir(parents=True, exist_ok=True)

        msgs = append_addendum_to_prompt(row["prompt"], addendum)
        (task_dir / "prompt.txt").write_text(json.dumps(msgs, indent=2))
        (task_dir / f"{name}_torch.py").write_text(row["reward_model"]["ground_truth"])

        result_path = task_dir / "result.json"
        result = {}
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
            except Exception:
                pass
        result.update(dict(problem_id=pid, name=name))
        if args.skip_completed and result.get("finish_reason") is not None:
            print(f"[gen {ix:>3}/{len(df)}] pid={pid} {name} SKIP (already done)")
            continue

        t0 = time.time()
        try:
            create_kwargs = dict(
                model=args.model,
                messages=msgs,
                max_tokens=args.max_tokens,
                n=1,
                timeout=GEN_TIMEOUT_S,
            )
            # Only pass sampling overrides when explicitly set; otherwise vLLM
            # falls back to the model's generation_config.json defaults.
            if args.temperature is not None:
                create_kwargs["temperature"] = args.temperature
            if args.top_p is not None:
                create_kwargs["top_p"] = args.top_p
            rsp = client.chat.completions.create(**create_kwargs)
            raw = rsp.choices[0].message.content or ""
            finish = rsp.choices[0].finish_reason
            usage = rsp.usage.model_dump() if rsp.usage else {}
            result.update(
                gen_seconds=round(time.time() - t0, 2),
                response_chars=len(raw),
                finish_reason=finish,
                usage=usage,
            )
            (task_dir / "raw_response.txt").write_text(raw)

            code = extract_python_block(raw, first=args.first_block)
            result["parsed"] = code is not None
            if code is not None:
                (task_dir / f"{name}_triton_ascend_impl.py").write_text(code)
        except Exception as e:
            result["parsed"] = False
            result["gen_error"] = f"{type(e).__name__}: {e}"
            result["gen_seconds"] = round(time.time() - t0, 2)
            (task_dir / "raw_response.txt").write_text(f"<gen error>\n{traceback.format_exc()}")

        result_path.write_text(json.dumps(result, indent=2, default=str))
        elapsed = time.time() - t_total_start
        print(
            f"[gen {ix:>3}/{len(df)}] pid={pid} {name} "
            f"parsed={result.get('parsed')} {result.get('gen_seconds')}s "
            f"finish={result.get('finish_reason', '?')} "
            f"({elapsed/60:.1f}min total)"
        )


# ============================================================================
# Phase B: parallel verify
# ============================================================================

def _verify_worker(task_dir_str: str, name: str, chip: str) -> dict:
    """Subprocess verify.py pinned to the given chip. Returns dict with results."""
    task_dir = Path(task_dir_str)
    impl_path = task_dir / f"{name}_triton_ascend_impl.py"
    if not impl_path.exists():
        return dict(compiled=False, precision_pass=False, detail="no_impl_file")

    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = chip
    # Per-task triton cache to avoid cross-task pollution
    env["TRITON_CACHE_DIR"] = str(task_dir / "_tcache")

    try:
        proc = subprocess.run(
            [
                "python3",
                str(VERIFY_PY),
                "--op_name", name,
                "--verify_dir", str(task_dir),
                "--triton_impl_name", "triton_ascend_impl",
                "--timeout", str(VERIFY_TIMEOUT_S),
            ],
            capture_output=True, text=True, env=env,
            timeout=VERIFY_TIMEOUT_S + 60,
        )
        out = proc.stdout or ""
        err = proc.stderr or ""
    except subprocess.TimeoutExpired:
        return dict(compiled=False, precision_pass=False, detail="verify_timeout", chip=chip)
    except Exception as e:
        return dict(compiled=False, precision_pass=False, detail=f"verify_exc:{type(e).__name__}:{e}", chip=chip)

    combined = out + err
    precision_pass = (proc.returncode == 0 and "验证成功" in out)
    precision_failed = "验证失败" in combined
    compiled = precision_pass or precision_failed

    return dict(
        compiled=compiled,
        precision_pass=precision_pass,
        verify_returncode=proc.returncode,
        chip=chip,
        stdout_tail=out[-1000:] if out else "",
        stderr_tail=err[-1500:] if err else "",
    )


def phase_verify(args, out_dir: Path):
    """Run verify on all task dirs in parallel."""
    task_dirs = sorted([d for d in out_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                       key=lambda p: int(p.name))
    print(f"[verify] {len(task_dirs)} tasks across {len(VERIFY_CHIPS)} chips: {VERIFY_CHIPS}")
    if args.filter:
        ids = set(int(x) for x in args.filter.split(","))
        task_dirs = [d for d in task_dirs if int(d.name) in ids]
        print(f"[verify] filter applied: {len(task_dirs)} tasks")

    submissions = []
    for i, td in enumerate(task_dirs):
        # Pull the name from the existing result.json (set during gen) or the impl filename
        result = json.loads((td / "result.json").read_text())
        name = result["name"]
        chip = VERIFY_CHIPS[i % len(VERIFY_CHIPS)]
        submissions.append((td, name, chip))

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(VERIFY_CHIPS)) as pool:
        futs = {
            pool.submit(_verify_worker, str(td), name, chip): (td, name, chip)
            for td, name, chip in submissions
        }
        done = 0
        for f in as_completed(futs):
            td, name, chip = futs[f]
            try:
                res = f.result()
            except Exception as e:
                res = dict(compiled=False, precision_pass=False, detail=f"future_exc:{e}", chip=chip)
            done += 1
            # Update task's result.json
            rp = td / "result.json"
            existing = json.loads(rp.read_text())
            existing["verify"] = res
            existing["compiled"] = res.get("compiled", False)
            existing["precision_pass"] = res.get("precision_pass", False)
            rp.write_text(json.dumps(existing, indent=2, default=str))
            stage = "PRECISION" if res["precision_pass"] else ("COMPILE" if res["compiled"] else "fail")
            print(f"[verify {done:>3}/{len(submissions)}] pid={td.name} chip={chip} {stage}")

    print(f"[verify] done in {(time.time() - t0)/60:.1f}min")


# ============================================================================
# Phase C: serial benchmark
# ============================================================================

def phase_benchmark(args, out_dir: Path):
    """Run benchmark.py serially on BENCHMARK_CHIP for precision_pass tasks only."""
    task_dirs = sorted([d for d in out_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                       key=lambda p: int(p.name))
    if args.filter:
        ids = set(int(x) for x in args.filter.split(","))
        task_dirs = [d for d in task_dirs if int(d.name) in ids]

    eligible = []
    for td in task_dirs:
        rp = td / "result.json"
        if not rp.exists():
            continue
        r = json.loads(rp.read_text())
        if r.get("precision_pass"):
            eligible.append((td, r["name"]))

    print(f"[benchmark] {len(eligible)} precision_pass tasks; chip={BENCHMARK_CHIP} (serial)")
    t0 = time.time()
    for ix, (td, name) in enumerate(eligible, start=1):
        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = BENCHMARK_CHIP
        env["TRITON_CACHE_DIR"] = str(td / "_tcache_bench")
    
        bench_json = td / "benchmark_result.json"
        try:
            proc = subprocess.run(
                [
                    "python3", str(BENCHMARK_PY),
                    "--op_name", name,
                    "--verify_dir", str(td),
                    "--triton_impl_name", "triton_ascend_impl",
                    "--warmup", str(args.bench_warmup),
                    "--repeats", str(args.bench_repeats),
                    "--output", str(bench_json),
                ],
                capture_output=True, text=True, env=env,
                timeout=BENCHMARK_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            print(f"[benchmark {ix:>3}/{len(eligible)}] pid={td.name} TIMEOUT")
            time.sleep(args.bench_settle_s)
            continue
        except Exception as e:
            print(f"[benchmark {ix:>3}/{len(eligible)}] pid={td.name} EXC {e}")
            continue

        speedup = None
        if bench_json.exists():
            try:
                bd = json.loads(bench_json.read_text())
                speedup = bd.get("speedup_vs_torch")
            except Exception:
                pass

        # Update result.json
        rp = td / "result.json"
        r = json.loads(rp.read_text())
        r["speedup"] = speedup
        r["benchmark_returncode"] = proc.returncode
        rp.write_text(json.dumps(r, indent=2, default=str))
        print(f"[benchmark {ix:>3}/{len(eligible)}] pid={td.name} speedup={speedup}")

        time.sleep(args.bench_settle_s)
    print(f"[benchmark] done in {(time.time() - t0)/60:.1f}min")


# ============================================================================
# Phase D: summary
# ============================================================================

def phase_summary(args, out_dir: Path):
    rows = []
    for td in sorted([d for d in out_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                     key=lambda p: int(p.name)):
        rp = td / "result.json"
        if not rp.exists():
            continue
        r = json.loads(rp.read_text())
        rows.append(
            dict(
                problem_id=r.get("problem_id"),
                name=r.get("name"),
                parsed=r.get("parsed", False),
                compiled=r.get("compiled", False),
                precision_pass=r.get("precision_pass", False),
                speedup=r.get("speedup"),
                gen_seconds=r.get("gen_seconds"),
                response_chars=r.get("response_chars"),
                finish_reason=r.get("finish_reason"),
            )
        )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)

    n = len(df)
    n_parsed = int(df["parsed"].sum()) if n else 0
    n_compiled = int(df["compiled"].sum()) if n else 0
    n_precision = int(df["precision_pass"].sum()) if n else 0
    speedups = df.dropna(subset=["speedup"])["speedup"].astype(float)
    n_fast = int((speedups > 1.0).sum()) if len(speedups) else 0
    mean_speedup = float(speedups.mean()) if len(speedups) else 0.0
    max_speedup = float(speedups.max()) if len(speedups) else 0.0

    md_path = out_dir / "summary.md"
    with md_path.open("w") as f:
        f.write(f"# Eval summary — {out_dir.name}\n\n")
        f.write(f"Total tasks: {n}\n\n")
        f.write(f"| stage | count | rate |\n|---|---|---|\n")
        f.write(f"| Parsed       | {n_parsed} | {100*n_parsed/n:.1f}% |\n")
        f.write(f"| Compiled     | {n_compiled} | {100*n_compiled/n:.1f}% |\n")
        f.write(f"| Precision    | {n_precision} | {100*n_precision/n:.1f}% |\n")
        f.write(f"| Speedup > 1x | {n_fast} | mean {mean_speedup:.2f}x, max {max_speedup:.2f}x |\n\n")
        if len(speedups):
            top = df.dropna(subset=["speedup"]).nlargest(10, "speedup")[["problem_id", "name", "speedup"]]
            f.write("## Top speedups\n\n")
            for _, r in top.iterrows():
                f.write(f"- pid={r['problem_id']} {r['name']} → {float(r['speedup']):.2f}x\n")
    print(f"[summary] wrote {csv_path} and {md_path}")
    print(f"  parsed={n_parsed}/{n}  compiled={n_compiled}/{n}  precision={n_precision}/{n}  speedup>1x={n_fast}/{n}")


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="drkernel-14b", help="vLLM served-model-name")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--addendum", default=None, help="path to prompt addendum text file")
    ap.add_argument(
        "--phases", default="gen,verify,benchmark,summary",
        help="comma-separated subset of: gen,verify,benchmark,summary"
    )
    ap.add_argument("--filter", default=None, help="comma-separated problem_ids to limit")
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--temperature", type=float, default=None,
                    help="if unset, vLLM uses the model's generation_config.json default")
    ap.add_argument("--top_p", type=float, default=None,
                    help="if unset, vLLM uses the model's generation_config.json default")
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--first-block", action="store_true", help="prefer first complete code block (vs default last)")
    ap.add_argument("--skip-completed", action="store_true", help="skip tasks with existing complete result.json (resume mode)")
    ap.add_argument("--bench_warmup", type=int, default=5)
    ap.add_argument("--bench_repeats", type=int, default=50)
    ap.add_argument("--bench_settle_s", type=int, default=5, help="sleep seconds between benchmark runs")
    ap.add_argument("--skip-gen", action="store_true", help="don't run gen phase even if listed")
    ap.add_argument("--hand-written-impl", default=None,
                    help="path to a manually-written ModelNew impl (for harness smoke test). "
                         "Used with --skip-gen --filter <pid>. Will be copied to "
                         "<out>/<pid>/{name}_triton_ascend_impl.py before verify.")
    ap.add_argument("--verifier-dir", type=Path, default=_DEFAULT_VERIFIER_DIR,
                    help="Path to the AscendOpGenAgent kernel-verifier scripts directory. "
                         "Defaults to $TRITON_VERIFIER_DIR if set, otherwise to "
                         "<repo>/external/AscendOpGenAgent/skills/triton/kernel-verifier/scripts "
                         "(populated by `git submodule update --init`).")
    args = ap.parse_args()

    # Resolve verifier scripts based on the CLI flag (overrides import-time default).
    global VERIFIER_DIR, VERIFY_PY, BENCHMARK_PY
    VERIFIER_DIR = args.verifier_dir
    VERIFY_PY = VERIFIER_DIR / "verify.py"
    BENCHMARK_PY = VERIFIER_DIR / "benchmark.py"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    if args.filter:
        ids = set(int(x) for x in args.filter.split(","))
        df = df[df["extra_info"].apply(lambda e: int(e["problem_id"]) in ids)]

    addendum = ""
    if args.addendum:
        addendum = Path(args.addendum).read_text()

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]

    # If hand-written impl is given, stage it before verify
    if args.hand_written_impl:
        if df.empty or len(df) != 1:
            sys.exit("--hand-written-impl requires --filter with a single problem_id")
        row = df.iloc[0]
        pid = int(row["extra_info"]["problem_id"])
        name = row["extra_info"]["name"]
        td = out_dir / str(pid)
        td.mkdir(parents=True, exist_ok=True)
        # Stage torch reference
        (td / f"{name}_torch.py").write_text(row["reward_model"]["ground_truth"])
        # Copy hand-written impl
        impl_src = Path(args.hand_written_impl).read_text()
        (td / f"{name}_triton_ascend_impl.py").write_text(impl_src)
        # Init result.json so verify can find name
        rp = td / "result.json"
        rp.write_text(json.dumps(dict(problem_id=pid, name=name, parsed=True), indent=2))
        print(f"[hand-written] staged impl for pid={pid} name={name}")

    if "gen" in phases and not args.skip_gen:
        phase_gen(args, df, out_dir, addendum)
    if "verify" in phases:
        phase_verify(args, out_dir)
    if "benchmark" in phases:
        phase_benchmark(args, out_dir)
    if "summary" in phases:
        phase_summary(args, out_dir)


if __name__ == "__main__":
    main()