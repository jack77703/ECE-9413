# ECE-9413: High-performance kernels in JAX

NYU **ECE-9413** course project: two self-contained assignments implementing exact modular-arithmetic workloads on CPU/GPU using [JAX](https://github.com/google/jax).

| Assignment | Focus | Editable surface |
|------------|--------|------------------|
| **A1** | Negacyclic Number Theoretic Transform (NTT) | `assignment1/student.py` |
| **A2** | SumCheck prover (Boolean hypercube, MLE updates) | `assignment2/student.py` |

Public APIs, test harnesses, and benchmarks are fixed; grading expects those contracts to stay unchanged.

---

## Repository layout

```
assignment1/          # NTT: pyproject, tests, scripts, student.py
assignment2/          # SumCheck: pyproject, tests, scripts, student.py
hpc/                  # Slurm + Singularity helpers for NYU HPC (A100)
report/               # Final IEEE-style report (LaTeX source + PDF)
```

Each assignment is an independent **uv** project with its own `pyproject.toml` and virtual environment.

---

## Prerequisites

- **Python** ≥ 3.11 (see each `pyproject.toml`)
- **[uv](https://docs.astral.sh/uv/)** for environments and dependency sync
- **GPU (optional)** — NVIDIA driver + CUDA build of JAX for GPU benchmarks; CPU wheels suffice for correctness

Install uv (macOS / Linux):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Quick start

From the **assignment directory** (recommended — matches course READMEs):

**Assignment 1**

```bash
cd assignment1
bash scripts/setup.sh          # CPU JAX by default; CUDA if a driver is visible
uv run pytest
uv run pytest --logn 10 --batch 4
uv run python -m tests.benchmark --tests --logn 10 --batch 4
```

**Assignment 2**

```bash
cd assignment2
bash scripts/setup.sh
uv run pytest --bits 32 --num-vars 4
uv run pytest --bits 32 --num-vars 16
uv run pytest --bits 32 --num-vars 20
uv run python -m tests.benchmark --bench --bits 32 --num-vars 20 --runs 8 --warmup 3
```

**From repo root** (equivalent — sets the project explicitly):

```bash
uv --directory assignment1 run pytest
uv --directory assignment2 run pytest --bits 32 --num-vars 20
```

---

## Development notes

### JAX backends

```bash
export JAX_PLATFORMS=cpu    # CPU-only
export JAX_PLATFORMS=cuda   # NVIDIA GPU (synonyms: gpu/cuda depend on JAX version)
```

### Assignment 1 — optional Pallas kernel

The default path is **reshape + vectorized butterflies** (stable across JAX releases). An experimental **Pallas** NTT path is **opt-in** (newer JAX can reject the prototype without this guard):

```bash
export ECE9413_USE_PALLAS=1
```

Leave unset for the configuration used in the course report’s primary A100 numbers.

### Assignment 2 — protocol ordering

Per assignment spec and `sumcheck_intro.md`: **round polynomial first**, then **MLE / challenge fold** for the next round. Do not pre-fold tables using later-round challenges ahead of time.

### Submission bundles

Each assignment ships `scripts/make_submission.sh` (runs required checks and emits `code.zip`). Run from the corresponding assignment root:

```bash
cd assignment1 && bash scripts/make_submission.sh
cd assignment2 && bash scripts/make_submission.sh
```

---

## Benchmarking & HPC

- Detailed **pytest** and **`tests.benchmark`** options: see `assignment1/README.md` and `assignment2/README.md`.
- **NYU HPC (Slurm, A100, Singularity)**: see [`hpc/README.md`](hpc/README.md) for `sbatch` submission, `PROJECT_DIR` / `OUTPUT_DIR` overrides, and artifact paths.

---

## Report

The final write-up for both assignments lives under **`report/`**:

- Source: `report/report.tex`
- Built PDF: `report/report.pdf` (e.g. `pdflatex` or `tectonic -X compile report.tex`)

---

## Documentation map

| Doc | Purpose |
|-----|---------|
| `assignment1/README.md` | NTT algorithm, setup, tests, benchmarks |
| `assignment2/README.md` | SumCheck protocol, expressions, CLI flags, extras |
| `assignment2/sumcheck_intro.md` | Step-by-step protocol / debugging |
| `hpc/README.md` | A100 job scripts and environment overrides |

---

## Contributing / course context

This tree is structured for **ECE-9413** grading and reproducibility: keep changes scoped to each `student.py` unless the course staff asks otherwise. For group work, coordinate a single shared report PDF and consistent benchmark hardware notes (see report abstract).

---

## License / use

Course instructional materials and assignment scaffolding; reuse outside NYU is governed by your instructor’s policy.
