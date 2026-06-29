"""KernelGym caller for the rollout-side multi-turn agent loop.

Why not reuse `recipe.drkernel.rewards.reward_client.KernelRewardClient`?
- That client lives in the `RewardLoopWorker` Ray actors and spawns its
  own `_HybridHttpWorker` actor pool to fan out batched evals across
  many trajectories. Calling it from the rollout-side agent loop would
  mean either re-spawning the same actor pool inside every
  `AgentLoopWorker` (wasteful, creates ownership puzzles) or piggy-
  backing on the reward-side pool (cross-actor ref-passing complexity
  we already paid for once).
- This caller does the per-turn job in httpx: POST /evaluate, poll
  /status, fetch /results. One trajectory at a time, in the agent
  loop's own event loop. No Ray actors involved.

The reward path remains the canonical training-signal computation; this
is just the "feedback message between turns" call. Both ultimately hit
the same KernelGym server with the same payload schema.

Faithful-to-original payload: matches the reward-path defaults
(`reward_client.py:621-636`) so the per-turn feedback the model sees
contains the same compile/correctness/speedup/profiling fields the
training reward is computed against. The original DR.Kernel pipeline
(`vllm_async_engine_multi_iter.py:1965-1989`) JSON-dumps the full
`env_state` (= reward manager `results` dict) into the user-feedback
template; `format_feedback(..., mode="full_json")` mirrors that.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)


_DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=5.0)


def _submit_backoff(attempt: int, base: int = 2, cap: int = 30) -> float:
    """Exponential backoff with a cap. Matches `reward_client.py:51-52`."""
    return min(base ** attempt, cap)


async def evaluate_kernel(
    *,
    server_url: str,
    reference_code: str,
    kernel_code: str,
    entry_point: str = "Model",
    task_timeout: int = 600,
    client_timeout: int = 1200,
    poll_interval: float = 1.0,
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    enable_profiling: bool = True,
    verbose_errors: bool = True,
    detect_decoy_kernel: bool = True,
    is_valid: bool = False,
    summarizer: Optional[Any] = None,
    extra_payload: Optional[Dict[str, Any]] = None,
    submit_max_retries: int = 3,
) -> Dict[str, Any]:
    """Evaluate a kernel candidate against the reference and return the
    KernelGym result dict.

    Defaults match `recipe.drkernel.rewards.reward_client.KernelRewardClient`
    so the rollout-side feedback uses the same payload as the training
    reward path. In particular `num_perf_trials >= 1` (server-side
    `EvaluationRequest` validator: `Field(default=100, ge=1, le=1000)`)
    — sending 0 used to trigger an HTTP 400 from the server's
    `validation_exception_handler`, masking every per-turn evaluation.

    Result schema (matches what `_HybridHttpWorker.submit_and_poll` returns):
        {"status": "completed"|"failed"|"timeout"|"cancelled", ...}

    Failure modes (non-`completed` status, network errors, JSON errors)
    return a dict with `status="failed"` and an `error_message` so the
    caller can format a feedback string instead of raising. On HTTP
    error responses, the server-provided error body is included so
    schema/validation problems surface in the feedback rather than as
    an opaque "HTTP NNN" string.
    """
    task_id = f"agent_loop_{uuid4().hex[:12]}"
    payload: Dict[str, Any] = {
        "task_id": task_id,
        "reference_code": reference_code,
        "kernel_code": kernel_code,
        "backend": "triton",
        "entry_point": entry_point,
        "timeout": int(task_timeout),
        "priority": "normal",
        "verbose_errors": bool(verbose_errors),
        "enable_profiling": bool(enable_profiling),
        "detect_decoy_kernel": bool(detect_decoy_kernel),
        "num_correct_trials": int(num_correct_trials),
        "num_perf_trials": int(num_perf_trials),
        "is_valid": bool(is_valid),
    }
    if extra_payload:
        payload.update(extra_payload)

    start_ts = time.time()
    try:
        async with httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8, keepalive_expiry=30.0),
            headers={"Content-Type": "application/json"},
        ) as client:
            # POST /evaluate with bounded retry on transient failures
            # (429/503 backpressure + httpx.TimeoutException/ConnectError).
            # Mirrors `reward_client.py::_HybridHttpWorker.submit_and_poll`
            # lines 73-100. Without this a transient KernelGym hiccup
            # zeros out the turn's reward.
            resp = None
            last_error = None
            for attempt in range(submit_max_retries):
                try:
                    resp = await client.post(f"{server_url}/evaluate", json=payload)
                    if resp.status_code == 200:
                        break
                    if resp.status_code in (429, 503):
                        # Server-side backpressure — back off and retry.
                        backoff = _submit_backoff(
                            attempt, base=2 if resp.status_code == 429 else 5
                        )
                        await asyncio.sleep(backoff)
                        last_error = f"HTTP {resp.status_code} (will retry)"
                        continue
                    # Other non-200: don't retry, return error.
                    body_excerpt = _extract_error_body(resp)
                    return _maybe_summarize(
                        {
                            "status": "failed",
                            "error_message": f"HTTP {resp.status_code} on /evaluate: {body_excerpt}",
                        },
                        summarizer,
                    )
                except (httpx.TimeoutException, httpx.ConnectError) as exc:
                    last_error = str(exc)
                    if attempt < submit_max_retries - 1:
                        await asyncio.sleep(_submit_backoff(attempt))
                        continue
                    return _maybe_summarize(
                        {
                            "status": "failed",
                            "error_message": f"submit_and_poll transient error: {exc}",
                        },
                        summarizer,
                    )
            if resp is None or resp.status_code != 200:
                return _maybe_summarize(
                    {
                        "status": "failed",
                        "error_message": f"HTTP {resp.status_code if resp else 'no_response'} "
                                         f"on /evaluate after {submit_max_retries} attempts: {last_error}",
                    },
                    summarizer,
                )

            while time.time() - start_ts < client_timeout:
                try:
                    s = await client.get(f"{server_url}/status/{task_id}")
                    if s.status_code == 200:
                        data = s.json()
                        status = data.get("status", "unknown")
                        if status in ("completed", "failed", "timeout", "cancelled"):
                            if status == "completed":
                                r = await client.get(f"{server_url}/results/{task_id}")
                                if r.status_code == 200:
                                    result = r.json()
                                    result["status"] = status
                                    return _maybe_summarize(result, summarizer)
                                return _maybe_summarize(
                                    {
                                        "status": "failed",
                                        "error_message": f"HTTP {r.status_code} on /results: {_extract_error_body(r)}",
                                    },
                                    summarizer,
                                )
                            return _maybe_summarize(
                                {
                                    "status": status,
                                    "error_message": data.get("error_message", f"Task {status}"),
                                },
                                summarizer,
                            )
                except Exception as exc:
                    logger.debug("kernel_eval poll error: %s", exc)
                await asyncio.sleep(poll_interval)

            return _maybe_summarize(
                {"status": "timeout", "error_message": f"Client timeout after {client_timeout}s"},
                summarizer,
            )
    except Exception as exc:
        return _maybe_summarize(
            {"status": "failed", "error_message": f"kernel_eval exception: {exc}"},
            summarizer,
        )


def _maybe_summarize(result: Dict[str, Any], summarizer: Optional[Any]) -> Dict[str, Any]:
    """Apply the reward summarizer (if provided) so the dict that goes
    into `{feedback}` matches the original DR.Kernel `env_state`
    (raw KernelGym fields + `_merge_reward_result(reward_func(raw))`).

    Errors are also routed through the summarizer because the original
    reward funcs (`calculate_reward_speedup`, `calculate_reward_weighted`)
    handle non-`completed` status and return a dict with `reward`,
    `score`, etc. set to the configured penalty — preserving that
    means the model sees the same penalty-shaped feedback the original
    pipeline shows.
    """
    if summarizer is None:
        return result
    try:
        return summarizer.summarize(result)
    except Exception as exc:
        logger.warning("[kernel_eval] summarizer.summarize failed (%s); returning raw result", exc)
        return result


def _extract_error_body(resp: "httpx.Response") -> str:
    try:
        body = resp.json()
        msg = body.get("message") or body.get("detail") or body
        return json.dumps(msg, ensure_ascii=False)[:512]
    except Exception:
        try:
            return resp.text[:512]
        except Exception:
            return ""


# Keys stripped from the model-facing feedback dict in `drop_keys` mode.
# The same dict is still used elsewhere (metrics, reward path) — only the
# string the model sees is trimmed.
_FEEDBACK_DROP_KEYS = frozenset({
    "metadata",
    "num_custom_kernel",
    "num_total_kernels",
    "custom_kernel_cuda_time_in_profiling_us",
    "total_kernel_run_time_in_profiling_us",
    "task_id",
    "submitted_at",
    "completed_at",
    "processing_time",
    "profiling",
})


def format_feedback(
    result: Dict[str, Any],
    *,
    mode: str = "full_json",
    drop_keys: Optional[Any] = None,
) -> str:
    """Render a KernelGym result dict into the feedback string that
    fills `{feedback}` in the per-turn user prompt.

    Modes:
      - "full_json" (default, faithful to original): JSON-dump the full
        result dict the same way the original
        `vllm_async_engine_multi_iter.py:1968` does
        (`json.dumps(env_state, ensure_ascii=False, indent=2)`).
      - "drop_keys": same as `full_json`, but strip a configured key
        list (verbose IDs/timestamps/coverage internals that bloat the
        prompt without informing the next turn). The list is taken from
        the `drop_keys` argument if non-empty, else from
        `_FEEDBACK_DROP_KEYS`.
      - "summary": legacy compact key:value form (status, compiled,
        correctness, [speedup], [error]) — keeps the option around for
        ablations but is NOT what the original DR.Kernel pipeline does.
    """
    if mode == "summary":
        return _format_summary(result)
    if mode == "drop_keys" and isinstance(result, dict):
        keys_to_drop = (
            frozenset(drop_keys) if drop_keys else _FEEDBACK_DROP_KEYS
        )
        feedback_dict = {k: v for k, v in result.items() if k not in keys_to_drop}
    else:
        feedback_dict = result
    try:
        return json.dumps(feedback_dict, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        logger.warning("[kernel_eval] json.dumps failed (%s); falling back to summary", exc)
        return _format_summary(result)


def _format_summary(result: Dict[str, Any]) -> str:
    status = result.get("status", "unknown")
    if status == "completed":
        compiled = result.get("compiled", False)
        correctness = result.get("correctness", False)
        speedup = result.get("speedup", None)
        err = result.get("error", "") or result.get("error_message", "") or ""
        parts = [
            "status: completed",
            f"compiled: {compiled}",
            f"correctness: {correctness}",
        ]
        if speedup is not None:
            parts.append(f"speedup: {speedup}")
        if err:
            parts.append(f"error: {err}")
        return "\n".join(parts)
    err = result.get("error_message", "") or result.get("error", "") or ""
    return f"status: {status}\nerror: {err}"
