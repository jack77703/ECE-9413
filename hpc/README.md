# HPC: A100 benchmarks (NYU Slurm)

Optional helpers for running **Assignment 1 (NTT)** and **Assignment 2 (SumCheck)** benchmarks on NYU’s Slurm cluster with an **NVIDIA A100** inside a **CUDA 12** Singularity image.

Run commands from the **repository root** on an HPC login node (`ssh`, OnDemand shell, etc.).

---

## Defaults (override as needed)

Scripts read environment variables so different users/paths do not require editing files:

| Variable | Purpose |
|----------|---------|
| `PROJECT_DIR` | Git checkout of this repo on scratch |
| `OUTPUT_DIR` | Where `.out` / `.err` / result text files are written |
| `OVERLAY` | Singularity overlay with your Python/conda + `uv` (e.g. `.../my_overlay.ext3:ro`) |
| `SIF` | CUDA-capable container image (e.g. under `/share/apps/images/`) |
| `CONDA_ENV` | Conda env name inside the overlay |

**Example (replace with your netid and paths):**

```bash
export PROJECT_DIR=/scratch/$USER/ECE-9413
export OUTPUT_DIR=/scratch/$USER/ece9413_results
export OVERLAY=/scratch/$USER/my_mamba/my_overlay.ext3:ro
export CONDA_ENV=ece9413
```

Typical Slurm headers used for this course (confirm each semester with staff):

```bash
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
```

---

## One-time: sync the repo

```bash
cd "$PROJECT_DIR"
git fetch && git status
mkdir -p "$OUTPUT_DIR"
```

If the default **CUDA image path** in the `.sbatch` file does not exist on the cluster, discover one:

```bash
ls /share/apps/images/*cuda12*
```

Submit with an explicit image:

```bash
sbatch --export=ALL,SIF=/share/apps/images/YOUR_IMAGE.sif hpc/a1_ntt_a100.sbatch
```

---

## Environment inside the job

Jobs `singularity exec` the image, activate conda from the overlay (if present), then run `uv sync` / benchmarks.

If `uv` is missing in the overlay, install once in a writable layer or on the login node’s user site-packages:

```bash
python -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
```

To use a **different conda env name**:

```bash
sbatch --export=ALL,CONDA_ENV=your_env_name hpc/a1_ntt_a100.sbatch
```

---

## Submit jobs

From repo root:

```bash
sbatch hpc/a1_ntt_a100.sbatch
sbatch hpc/a2_sumcheck_a100.sbatch
```

Check queue:

```bash
squeue -u "$USER"
```

### Optional tuning

Assignment 1 sweep (example):

```bash
sbatch --export=ALL,A1_LOGNS="10 12 14 15",A1_BATCHES="1 4 16" hpc/a1_ntt_a100.sbatch
```

Assignment 2 benchmark iterations:

```bash
sbatch --export=ALL,RUNS=8,WARMUP=3 hpc/a2_sumcheck_a100.sbatch
```

Faster queue / shorter jobs (skip optional tracks):

```bash
sbatch --export=ALL,RUN_CHALLENGE32=0,RUN_CORE64_CORRECTNESS=0 hpc/a2_sumcheck_a100.sbatch
```

Optional 64-bit benchmarks:

```bash
sbatch --export=ALL,RUN_CORE64_BENCH=1,CORE64_NUM_VARS="4 16" hpc/a2_sumcheck_a100.sbatch
```

---

## Results to keep

After completion, copy or archive files under **`OUTPUT_DIR`**, typically:

- `ece9413-a1-ntt-a100_*.out` / `*.err`
- `ece9413-a2-sumcheck-a100_*.out` / `*.err`
- Any summarized `.txt` artifacts written next to those logs

**Sanity checks in the logs:**

- JAX reports `gpu` / `cuda` backend (not CPU-only for GPU jobs)
- `nvidia-smi` shows an **A100**
- Correctness sections report **passed**
- Benchmark tables include **median**, **p90**, and **throughput** where applicable

Use those numbers in `report/report.tex` and note non-default hardware explicitly.

---

## Related documentation

- Root overview: [`../README.md`](../README.md)
- Assignment CLI details: `assignment1/README.md`, `assignment2/README.md`
