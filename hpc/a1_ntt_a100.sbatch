#!/usr/bin/env bash
#SBATCH --job-name=ece9413-a1-ntt-a100
#SBATCH --output=/scratch/cc9171/results/%x_%j.out
#SBATCH --error=/scratch/cc9171/results/%x_%j.err
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --requeue

set -euo pipefail

# Override at submit time, e.g.:
#   sbatch --export=ALL,A1_LOGNS="10 12 14 15",A1_BATCHES="1 4 16" hpc/a1_ntt_a100.sbatch
PROJECT_DIR="${PROJECT_DIR:-/scratch/cc9171/ECE-9413}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/cc9171/ece9413_results}"
OVERLAY="${OVERLAY:-/scratch/cc9171/my_env/my_overlay.ext3:ro}"
SIF="${SIF:-/share/apps/images/cuda12.3.2-cudnn9.0-devel-ubuntu22.04.1.sif}"
CONDA_ENV="${CONDA_ENV:-ece9413}"
A1_LOGNS="${A1_LOGNS:-10 12 14}"
A1_BATCHES="${A1_BATCHES:-1 4 16}"
RUN_UV_SYNC="${RUN_UV_SYNC:-1}"

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

RESULT_FILE="${OUTPUT_DIR}/a1_ntt_a100_${SLURM_JOB_ID:-manual}.txt"
exec > >(tee "\${RESULT_FILE}") 2>&1

echo "=== Assignment 1 NTT A100 benchmark ==="
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

cd assignment1

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

echo "=== Correctness ==="
uv run pytest
uv run pytest --logn 10 --batch 4
echo

echo "=== Benchmarks ==="

for logn in ${A1_LOGNS}; do
  for batch in ${A1_BATCHES}; do
    echo
    echo "--- A1 bench logn=\${logn} batch=\${batch} ---"
    uv run python -m tests.benchmark --bench --logn "\${logn}" --batch "\${batch}"
  done
done

echo
echo "finished_at=\$(date -Is)"
echo "result_file=\${RESULT_FILE}"
INNER
