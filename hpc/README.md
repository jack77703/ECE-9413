# NYU HPC A100 Benchmark Jobs

Run these from the repository root on the NYU HPC login node.

These scripts are set up for the NYU ECE 9413 Spring 2026 A100 partition:

```bash
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
```

They use the same Singularity pattern as your working example:

```bash
singularity exec --nv \
  --overlay /scratch/cc9171/my_env/my_overlay.ext3:ro \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
```

## 1. Copy Or Pull The Repo

Make sure the HPC copy has the latest code:

```bash
cd /scratch/cc9171/ECE-9413
git status
```

By default, the scripts expect:

```bash
PROJECT_DIR=/scratch/cc9171/ECE-9413
OUTPUT_DIR=/scratch/cc9171/ece9413_results
OVERLAY=/scratch/cc9171/my_env/my_overlay.ext3:ro
CONDA_ENV=ece9413
```

Override any of these with `sbatch --export=ALL,VAR=value`.

## 2. Environment Setup

The scripts try:

```bash
source /ext3/miniconda3/bin/activate
conda activate ece9413 || true
uv sync --extra cuda12
```

If your overlay uses a different conda env name, submit with:

```bash
sbatch --export=ALL,CONDA_ENV=YOUR_ENV hpc/a1_ntt_a100.sbatch
```

If `uv` is not installed in the overlay, install it inside your writable environment once, or install before submission:

```bash
python -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

## 3. Submit Jobs

From the repo root:

```bash
mkdir -p /scratch/cc9171/ece9413_results
sbatch hpc/a1_ntt_a100.sbatch
sbatch hpc/a2_sumcheck_a100.sbatch
```

Check status:

```bash
squeue -u "$USER"
```

## 4. Optional Overrides

Assignment 1 benchmark sweep:

```bash
sbatch --export=ALL,A1_LOGNS="10 12 14 15",A1_BATCHES="1 4 16" hpc/a1_ntt_a100.sbatch
```

Assignment 2 benchmark settings:

```bash
sbatch --export=ALL,RUNS=8,WARMUP=3 hpc/a2_sumcheck_a100.sbatch
```

Skip optional advanced and 64-bit checks if queue time is tight:

```bash
sbatch --export=ALL,RUN_CHALLENGE32=0,RUN_CORE64_CORRECTNESS=0 hpc/a2_sumcheck_a100.sbatch
```

Run optional 64-bit benchmarks too:

```bash
sbatch --export=ALL,RUN_CORE64_BENCH=1,CORE64_NUM_VARS="4 16" hpc/a2_sumcheck_a100.sbatch
```

## 5. What To Send Back

After jobs finish, send or copy these files back:

```bash
/scratch/cc9171/ece9413_results/ece9413-a1-ntt-a100_*.out
/scratch/cc9171/ece9413_results/ece9413-a1-ntt-a100_*.err
/scratch/cc9171/ece9413_results/ece9413-a2-sumcheck-a100_*.out
/scratch/cc9171/ece9413_results/ece9413-a2-sumcheck-a100_*.err
/scratch/cc9171/ece9413_results/a1_ntt_a100_*.txt
/scratch/cc9171/ece9413_results/a2_sumcheck_a100_*.txt
```

The important checks inside each result file are:

- `default_backend= gpu` or `default_backend= cuda`
- `nvidia-smi` shows an A100
- correctness tests pass
- benchmark tables list median latency, p90, and throughput

Those outputs will be used directly in `report/report.tex`.
