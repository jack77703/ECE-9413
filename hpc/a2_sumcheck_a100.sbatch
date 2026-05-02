#!/usr/bin/env bash
#SBATCH --job-name=ece9413-a2-sumcheck-a100
#SBATCH --output=/scratch/cc9171/results/%x_%j.out
#SBATCH --error=/scratch/cc9171/results/%x_%j.err
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --requeue

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/cc9171/ECE-9413}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/cc9171/ece9413_results}"
OVERLAY="${OVERLAY:-/scratch/cc9171/my_env/my_overlay.ext3:ro}"
SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0-devel-ubuntu22.04.1.sif}"
CONDA_ENV="${CONDA_ENV:-ece9413}"
RUN_UV_SYNC="${RUN_UV_SYNC:-1}"
RUNS="${RUNS:-8}"
WARMUP="${WARMUP:-3}"
RUN_CHALLENGE32="${RUN_CHALLENGE32:-1}"
CHALLENGE32_NUM_VARS="${CHALLENGE32_NUM_VARS:-16 20}"
RUN_CORE64_CORRECTNESS="${RUN_CORE64_CORRECTNESS:-1}"
RUN_CORE64_BENCH="${RUN_CORE64_BENCH:-0}"
CORE64_NUM_VARS="${CORE64_NUM_VARS:-4 16}"

mkdir -p "${OUTPUT_DIR}"

singularity exec --nv \
  --overlay "${OVERLAY}" \
  "${SIF}" \
  bash << INNER
set -eo pipefail

source /ext3/miniconda3/bin/activate
conda activate "${CONDA_ENV}" || true

export PATH="\$HOME/.local/bin:\$PATH"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

cd "${PROJECT_DIR}"
mkdir -p hpc/logs hpc/results "${OUTPUT_DIR}"

RESULT_FILE="${OUTPUT_DIR}/a2_sumcheck_a100_${SLURM_JOB_ID:-manual}.txt"
exec > >(tee "\${RESULT_FILE}") 2>&1

echo "=== Assignment 2 SumCheck A100 benchmark ==="
echo "started_at=\$(date -Is)"
echo "repo=${PROJECT_DIR}"
echo "host=\$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo

echo "=== GPU ==="
nvidia-smi || true
echo

echo "=== Environment ==="
which python || true
python --version || true
which uv || true
uv --version || true
echo

cd assignment2

if [[ "${RUN_UV_SYNC}" == "1" ]]; then
  echo "=== uv sync cuda12 ==="
  uv sync --extra cuda12
  echo
fi

echo "=== JAX device check ==="
uv run python - <<'PY'
import jax
print("jax_version=", jax.__version__)
print("default_backend=", jax.default_backend())
print("devices=", jax.devices())
if jax.default_backend() not in ("gpu", "cuda"):
    raise SystemExit("JAX did not select a GPU backend; check CUDA image, driver, and JAX install.")
PY
echo

echo "=== Required correctness ==="
uv run pytest --bits 32 --num-vars 4
uv run pytest --bits 32 --num-vars 16
uv run pytest --bits 32 --num-vars 20
echo

echo "=== Required 32-bit benchmarks ==="
for num_vars in 4 16 20; do
  echo
  echo "--- A2 required bench bits=32 num_vars=\${num_vars} runs=${RUNS} warmup=${WARMUP} ---"
  uv run python -m tests.benchmark --bench --bits 32 --num-vars "\${num_vars}" --runs "${RUNS}" --warmup "${WARMUP}"
done

if [[ "${RUN_CHALLENGE32}" == "1" ]]; then
  echo
  echo "=== Optional 32-bit advanced polynomial benchmark ==="
  for num_vars in ${CHALLENGE32_NUM_VARS}; do
    echo
    echo "--- A2 challenge32 bench num_vars=\${num_vars} ---"
    uv run python -m tests.benchmark --bench --bits 32 --num-vars "\${num_vars}" --enable-challenge32 --runs "${RUNS}" --warmup "${WARMUP}"
  done
fi

if [[ "${RUN_CORE64_CORRECTNESS}" == "1" ]]; then
  echo
  echo "=== Optional 64-bit core correctness ==="
  uv run pytest --all-64 --enable-core64
fi

if [[ "${RUN_CORE64_BENCH}" == "1" ]]; then
  echo
  echo "=== Optional 64-bit core benchmarks ==="
  for num_vars in ${CORE64_NUM_VARS}; do
    echo
    echo "--- A2 core64 bench num_vars=\${num_vars} ---"
    uv run python -m tests.benchmark --bench --bits 64 --num-vars "\${num_vars}" --enable-core64 --runs "${RUNS}" --warmup "${WARMUP}"
  done
fi

echo
echo "finished_at=\$(date -Is)"
echo "result_file=\${RESULT_FILE}"
INNER
