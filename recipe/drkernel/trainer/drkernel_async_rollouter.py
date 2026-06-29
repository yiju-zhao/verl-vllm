# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""DR.Kernel rollouter override.

Subclasses `verl.experimental.fully_async_policy.fully_async_rollouter.FullyAsyncRollouter`
to do three things on the validation side:

  1. Rewrite `val-core/<data_source>/reward/mean@N` from per-trajectory
     sum-of-turn-rewards to per-(traj, turn) mean, so it lines up with
     training's `critic/rewards/mean`. See `_val_metrics_update` below.

  2. Emit naive Pass@N metrics for binary indicators when
     `actor_rollout_ref.rollout.val_kwargs.n > 1`:

         val-core/<src>/Pass@<N>/correctness
         val-core/<src>/Pass@<N>/Fast@<X>/last
         val-core/<src>/Pass@<N>/Fast@<X>/best
         val-core/<src>/Pass@<N>/compilation
         val-core/<src>/Pass@<N>/is_speedup_positive

     For each (data_source, uid), Pass@N(key) = 1 iff any of the N rollouts
     hit that indicator. Averaged across uids per data_source. This is the
     "kernel-RL Pass@N": at least one of N stochastic samples produces a
     correct (or Fast@X-meeting) kernel.

     Note: `val-aux/<src>/<key>/best@<N>/mean` is already emitted by the
     parent via `process_validation_metrics` for the same indicators —
     that is a *bootstrap* estimate of best-of-N. Pass@<N>/<key> is the
     direct per-prompt any-of-N indicator, mean-aggregated. The two
     converge as the number of prompts grows.

  3. Capture a per-prompt rollout table (prompt text, ground truth, the N
     rollout summaries and the Pass@N indicators) so `main_validate.py`
     can include it in the standalone `val_metrics_step*.json` dump.
     Exposed via `get_per_prompt_table()`.

The per-prompt table is populated by intercepting `_dump_generations` —
the parent's `_validate` calls it once after the full validation loop and
before `_val_metrics_update`, so the captured input/output/gt/score lists
line up row-for-row with the data passed to `_val_metrics_update`. The
capture is a no-op when `trainer.validation_data_dir` is unset (parent
skips `_dump_generations` in that case) — `rollouts[*].output` falls
back to `None` and the rest of the per-prompt table is still produced
from `reward_extra_infos_dict`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import ray

from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncRollouter as _RemoteFullyAsyncRollouter,
)


# `FullyAsyncRollouter` is `@ray.remote(num_cpus=10, max_concurrency=100)`-decorated.
# Unwrap to subclass; re-decorate at the bottom with the same spec.
_BaseFullyAsyncRollouter = _RemoteFullyAsyncRollouter.__ray_actor_class__


# Scalar binary indicators (0/1) for which we emit a Pass@N value. These
# are populated per-trajectory by
# `recipe/drkernel/workers/reward_manager/kernel_async_adapter.py::
# _build_response_from_turn_rewards`. Float-valued keys (e.g. `speedup`,
# `reward`, `*/turn_avg`) are intentionally excluded — Pass@N is a
# binary-indicator concept; for continuous quantities the existing
# `val-aux/.../best@N/mean` bootstrap is the right summary.
_SCALAR_BINARY_PASS_KEYS = (
    "correctness",
    "compilation",
    "is_speedup_positive",
)


def _is_binary_pass_at_n_key(key: str) -> bool:
    """True if `key` is a per-trajectory 0/1 indicator we report Pass@N for."""
    if key in _SCALAR_BINARY_PASS_KEYS:
        return True
    # Fast@<X>/last (last-turn) and Fast@<X>/best (any-turn-achieved) are
    # the two binary forms emitted by the kernel_async adapter; the
    # corresponding `/turn_avg` is a float in [0, 1] and skipped here.
    if key.startswith("Fast@") and (key.endswith("/last") or key.endswith("/best")):
        return True
    return False


class DrKernelFullyAsyncRollouterImpl(_BaseFullyAsyncRollouter):
    """FullyAsyncRollouter with per-turn reward aggregation, naive Pass@N
    metrics for binary indicators, and a per-prompt rollout table dumped
    via `main_validate.py`."""

    def _dump_generations(
        self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path
    ):
        """Tee the row-wise sample lists onto `self` so `_val_metrics_update`
        can build the per-prompt rollout table. Forwards to the parent
        unchanged so the existing `validation_data_dir/<step>.jsonl` dump
        is preserved byte-for-byte."""
        self._captured_dump_inputs = list(inputs)
        self._captured_dump_outputs = list(outputs)
        self._captured_dump_gts = list(gts)
        self._captured_dump_scores = list(scores)
        return super()._dump_generations(
            inputs=inputs,
            outputs=outputs,
            gts=gts,
            scores=scores,
            reward_extra_infos_dict=reward_extra_infos_dict,
            dump_path=dump_path,
        )

    def _val_metrics_update(
        self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns,
    ):
        """Rewrite ``reward_extra_infos_dict["reward"]`` from per-trajectory
        sum to per-(traj, turn) mean, delegate to the parent, then layer
        naive Pass@N metrics + per-prompt table on top.

        Reward denominator sources, in priority order:
          1. ``reward_extra_infos_dict["num_turns"]`` — emitted by
             `recipe/drkernel/workers/reward_manager/kernel_async_adapter.py::
             _build_response_from_turn_rewards` as ``len(turn_rewards)``.
             This is the assistant-turn count that produced the reward
             sum, so the division is consistent.
          2. ``sample_turns`` (concatenated) — falls back to the
             ``__num_turns__`` non-tensor key, which counts
             ``user_turns + assistant_turns + 1`` (see `KernelAgentLoop.run`).
             Not strictly equivalent to ``len(turn_rewards)``, but a
             reasonable approximation for callers that didn't go through
             the fast path (e.g. legacy single-call reward manager).

        Falls back to the parent untouched if neither denominator lines up
        with the reward length (defensive — keeps validation working even
        if something upstream changes the shape).
        """
        rewards = reward_extra_infos_dict.get("reward")
        if rewards:
            num_rewards = len(rewards)
            num_turns = reward_extra_infos_dict.get("num_turns")
            if num_turns is None or len(num_turns) != num_rewards:
                num_turns = None
                if sample_turns:
                    try:
                        candidate = np.concatenate(sample_turns)
                    except Exception:
                        candidate = None
                    if candidate is not None and len(candidate) == num_rewards:
                        num_turns = candidate
            if num_turns is not None and len(num_turns) == num_rewards:
                rewards_arr = np.asarray(rewards, dtype=np.float64)
                turns_arr = np.asarray(num_turns, dtype=np.float64)
                # Defensive: avoid div-by-zero on degenerate (no-turn)
                # trajectories — value is 0/1 = 0.0 either way.
                safe_turns = np.where(turns_arr > 0, turns_arr, 1.0)
                reward_extra_infos_dict["reward"] = (rewards_arr / safe_turns).tolist()

        metric_dict = super()._val_metrics_update(
            data_sources, sample_uids, reward_extra_infos_dict, sample_turns,
        )

        # ---- Pass@N + per-prompt rollout table ----
        try:
            n_rollouts = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)
        except Exception:
            n_rollouts = 1

        if n_rollouts > 1:
            pass_n_metrics, per_prompt = self._compute_pass_at_n_and_per_prompt(
                data_sources=data_sources,
                sample_uids=sample_uids,
                reward_extra_infos_dict=reward_extra_infos_dict,
                n_rollouts=n_rollouts,
            )
            metric_dict.update(pass_n_metrics)
            self._captured_per_prompt = per_prompt
        else:
            # N=1 — Pass@N degenerates to the per-prompt indicator mean
            # already in val-aux/<src>/<key>/mean@1. Skip emission and
            # clear any stale per-prompt table from a prior do_validate().
            self._captured_per_prompt = []

        return metric_dict

    def _compute_pass_at_n_and_per_prompt(
        self,
        *,
        data_sources,
        sample_uids,
        reward_extra_infos_dict: dict[str, list],
        n_rollouts: int,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Group rows by (data_source, uid), compute naive Pass@N for binary
        indicators and build the per-prompt rollout table.

        Row index `i` is consistent across `data_sources[i]`,
        `sample_uids[i]`, `reward_extra_infos_dict[<key>][i]`, and the
        captured `_dump_*[i]` lists — they are all extended in lockstep
        by the parent's `_validate` loop.

        Returns:
            pass_n_metrics: dict of TB-tag → mean indicator value, e.g.
                ``{"val-core/<src>/Pass@8/correctness": 0.62, ...}``.
            per_prompt: list[dict], one entry per (data_source, uid), each
                containing the prompt text, ground truth, per-rollout dicts
                and the Pass@N indicators for that prompt.
        """
        ds_list = (
            data_sources.tolist()
            if isinstance(data_sources, np.ndarray)
            else list(data_sources)
        )
        sample_uids = [str(u) for u in sample_uids]
        n_rows = len(sample_uids)

        # Captured by `_dump_generations` (None when validation_data_dir
        # is unset — per-rollout output/prompt fall back to None).
        cap_in = getattr(self, "_captured_dump_inputs", None)
        cap_out = getattr(self, "_captured_dump_outputs", None)
        cap_gt = getattr(self, "_captured_dump_gts", None)
        cap_score = getattr(self, "_captured_dump_scores", None)

        def _safe_get(lst, idx):
            if lst is None or idx >= len(lst):
                return None
            return lst[idx]

        # Group row indices by (data_source, uid), preserving the order
        # in which each unique key first appears so the dumped table is
        # deterministic and matches the val_dataloader iteration order.
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        first_seen_order: list[tuple[str, str]] = []
        for i in range(n_rows):
            key = (ds_list[i], sample_uids[i])
            if key not in groups:
                first_seen_order.append(key)
            groups[key].append(i)

        binary_keys = [
            k for k in reward_extra_infos_dict.keys() if _is_binary_pass_at_n_key(k)
        ]

        per_source_pass: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        per_prompt: list[dict[str, Any]] = []

        for group_key in first_seen_order:
            ds, uid = group_key
            row_idxs = groups[group_key]
            first = row_idxs[0]

            rollouts: list[dict[str, Any]] = []
            for ri in row_idxs:
                rollout: dict[str, Any] = {
                    "row_index": ri,
                    "output": _safe_get(cap_out, ri),
                    "score": _safe_get(cap_score, ri),
                }
                for k, v in reward_extra_infos_dict.items():
                    if isinstance(v, list) and ri < len(v):
                        rollout[k] = v[ri]
                rollouts.append(rollout)

            pass_indicators: dict[str, float] = {}
            for bkey in binary_keys:
                vals: list[float] = []
                bvals = reward_extra_infos_dict.get(bkey)
                if not isinstance(bvals, list):
                    continue
                for ri in row_idxs:
                    if ri >= len(bvals):
                        continue
                    try:
                        vals.append(float(bvals[ri]))
                    except (TypeError, ValueError):
                        continue
                if not vals:
                    continue
                # Naive Pass@N: 1 iff at least one rollout hit the
                # indicator. Threshold at 0.5 to tolerate stray 0.0/1.0
                # encodings as bools, ints, or floats.
                ind = 1.0 if max(vals) >= 0.5 else 0.0
                pass_indicators[bkey] = ind
                per_source_pass[ds][bkey].append(ind)

            per_prompt.append(
                {
                    "data_source": ds,
                    "uid": uid,
                    "n_rollouts": len(row_idxs),
                    "prompt": _safe_get(cap_in, first),
                    "ground_truth": _safe_get(cap_gt, first),
                    "pass_at_n": pass_indicators,
                    "rollouts": rollouts,
                }
            )

        pass_n_metrics: dict[str, float] = {}
        for ds, key2vals in per_source_pass.items():
            for bkey, vals in key2vals.items():
                if not vals:
                    continue
                pass_n_metrics[
                    f"val-core/{ds}/Pass@{n_rollouts}/{bkey}"
                ] = float(np.mean(vals))

        return pass_n_metrics, per_prompt

    def get_per_prompt_table(self) -> list[dict[str, Any]]:
        """Return the per-prompt rollout table built during the latest
        `do_validate()` call. Empty list when `val_kwargs.n <= 1` or
        when `do_validate()` has not yet been invoked."""
        return getattr(self, "_captured_per_prompt", []) or []


# Re-decorate with the same resource spec as the parent.
DrKernelFullyAsyncRollouter = ray.remote(num_cpus=10, max_concurrency=100)(
    DrKernelFullyAsyncRollouterImpl
)
