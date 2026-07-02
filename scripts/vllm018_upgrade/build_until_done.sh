#!/usr/bin/env bash
# Self-restarting vLLM 0.18 build. The long compile can be interrupted (the
# background-task runtime cap killed earlier attempts). ccache makes each retry
# resume from where the previous one stopped, so this loop converges.
#
# Exits 0 as soon as `import vllm` reports a 0.18 version; non-zero if it still
# hasn't after MAX attempts.
PY=/home/yubaifeng/e84381970/envs/drkernel310/bin/python3.10
BUILD=/home/yubaifeng/e84381970/experiment/verl-vllm/drkernel-verl-port-drkernel/scripts/vllm018_upgrade/build_vllm_cuda.sh
MAX="${MAX_ATTEMPTS:-20}"

is_done() {
  # top-level `import vllm` works even without the CUDA ext loaded; just check version.
  $PY -c "import vllm,sys; sys.exit(0 if vllm.__version__.startswith('0.18') else 1)" 2>/dev/null
}

for i in $(seq 1 "$MAX"); do
  if is_done; then
    echo "=== vllm 0.18 importable; build complete (before attempt $i) ==="
    exit 0
  fi
  echo "############ BUILD ATTEMPT $i/$MAX ############"
  bash "$BUILD" || echo ">>> build attempt $i exited non-zero (will retry; ccache resumes)"
done

if is_done; then
  echo "=== vllm 0.18 importable after $MAX attempts ==="
  exit 0
fi
echo "!!! vllm 0.18 still not importable after $MAX attempts !!!"
exit 1
