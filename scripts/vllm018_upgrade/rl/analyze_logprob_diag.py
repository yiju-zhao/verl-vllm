"""Token-level localization of the rollout-vs-actor logprob mismatch.

Loads the batch dumped by calculate_debug_metrics (VERL_LOGPROB_DIAG_DUMP) and answers:
where do the disagreeing tokens live?
  - clustered in a few sequences (-> alignment / assembly bug for those seqs)
  - step-function within a sequence: fine until pos k, then all bad (-> position shift)
  - concentrated at boundaries: first tokens / last tokens / around EOS (-> boundary artifact)
  - scattered uniformly (-> kernel/numeric noise amplified by the model)
Also decodes the top offending tokens.

Usage: python analyze_logprob_diag.py /tmp/logprob_diag/logprob_diag_*.pt
"""

import glob
import sys

import torch

FILES = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "/tmp/logprob_diag/*.pt"))
assert FILES, "no dump files found"
d = torch.load(FILES[0], map_location="cpu", weights_only=False)
rl, al = d["rollout_log_probs"].float(), d["actor_log_probs"].float()
mask, resp = d["response_mask"].bool(), d["responses"]
B, T = rl.shape
print(f"file={FILES[0]}  batch={B} seqs x {T} tokens, valid={int(mask.sum())}")

pr, pa = rl.exp(), al.exp()
diff = (pr - pa).abs() * mask  # prob-space diff, the metric's basis
ratio = torch.where(mask, (al - rl).exp(), torch.ones_like(rl))  # actor/rollout

BAD = 0.5  # "extreme disagreement" threshold in prob space
bad = (diff > BAD) & mask
low = (ratio < 0.5) & mask  # the fraction_low population

print(f"\n== overall ==")
print(f"extreme (diff>{BAD}): {int(bad.sum())} tokens ({bad.sum() / mask.sum() * 100:.2f}% of valid)")
print(f"low-ratio (<0.5):     {int(low.sum())} tokens ({low.sum() / mask.sum() * 100:.2f}% of valid)")

# --- 1. per-sequence clustering ---
per_seq_bad = bad.sum(1)
per_seq_len = mask.sum(1)
n_seq_with_bad = int((per_seq_bad > 0).sum())
print(f"\n== clustering ==")
print(f"sequences with >=1 extreme token: {n_seq_with_bad}/{B}")
top = per_seq_bad.argsort(descending=True)[:5]
for i in top.tolist():
    if per_seq_bad[i] == 0:
        break
    print(f"  seq {i}: {int(per_seq_bad[i])}/{int(per_seq_len[i])} extreme ({per_seq_bad[i] / per_seq_len[i] * 100:.1f}%)")

# --- 2. within-sequence pattern for the worst sequences ---
print(f"\n== within-seq pattern (worst 3 seqs; positions of extreme tokens) ==")
for i in top[:3].tolist():
    pos = bad[i].nonzero().flatten().tolist()
    if not pos:
        continue
    L = int(per_seq_len[i])
    runs = []
    s = pos[0]
    prev = pos[0]
    for p in pos[1:] + [None]:
        if p is None or p != prev + 1:
            runs.append((s, prev))
            s = p
        prev = p if p is not None else prev
    run_str = ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs[:12])
    print(f"  seq {i} (len {L}): {run_str}{' ...' if len(runs) > 12 else ''}")

# --- 3. positional histogram (relative position within response) ---
print(f"\n== relative-position histogram of extreme tokens (deciles of each seq) ==")
relpos_counts = torch.zeros(10)
allpos_counts = torch.zeros(10)
for i in range(B):
    L = int(per_seq_len[i])
    if L == 0:
        continue
    idx = torch.arange(T)[: L]
    decile = (idx.float() / L * 10).clamp(max=9).long()
    allpos_counts += torch.bincount(decile, minlength=10).float()
    bpos = bad[i, :L]
    if bpos.any():
        relpos_counts += torch.bincount(decile[bpos], minlength=10).float()
rate = torch.where(allpos_counts > 0, relpos_counts / allpos_counts, torch.zeros(10))
for k in range(10):
    print(f"  decile {k}: {rate[k] * 100:6.2f}% extreme  ({int(relpos_counts[k])} tokens)")

# --- 4. first/last token specialness ---
first_bad = sum(bool(bad[i, 0]) for i in range(B) if per_seq_len[i] > 0)
last_bad = sum(bool(bad[i, int(per_seq_len[i]) - 1]) for i in range(B) if per_seq_len[i] > 0)
print(f"\nfirst-token extreme: {first_bad}/{B} seqs | last-valid-token extreme: {last_bad}/{B} seqs")

# --- 5. top offending token ids (decoded) ---
print(f"\n== top offending token ids ==")
bad_ids = resp[bad]
if bad_ids.numel():
    vals, counts = bad_ids.unique(return_counts=True)
    order = counts.argsort(descending=True)[:10]
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        for j in order.tolist():
            tid = int(vals[j])
            print(f"  id={tid:7d} x{int(counts[j]):4d}  {tok.decode([tid])!r}")
    except Exception as e:  # noqa: BLE001
        for j in order.tolist():
            print(f"  id={int(vals[j]):7d} x{int(counts[j]):4d}")
        print(f"  (decode unavailable: {e})")

# --- 6. does rollout or actor assign the higher prob on bad tokens? ---
n_roll_hi = int(((pr > pa) & bad).sum())
print(f"\non extreme tokens: rollout-prob higher on {n_roll_hi}/{int(bad.sum())} "
      f"(rollout confident + actor not = {n_roll_hi}; actor confident + rollout not = {int(bad.sum()) - n_roll_hi})")
