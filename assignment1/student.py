"""
Negacyclic Number Theoretic Transform (NTT) implementation — optimised for A100 GPU.

y[k] = Σ_{n=0}^{N-1} x[n] · ψ^{(2k+1)·n}  (mod q)

Optimisation notes
------------------
* mod_add / mod_sub stay in uint32 — q < 2^31 means a+b < 2^32 (no overflow).
* _mont_mul uses a uint32 wrap multiplication for 'm' (low-32 bits of T*q_inv)
  instead of a 64-bit multiply + 64-bit AND.
* On GPU the entire NTT (all stages) is fused into a single Triton kernel via
  jax.experimental.pallas so that intermediate data never hits global memory.
  On CPU the same algorithm runs as plain JAX (pallas interpret-mode is skipped).
"""

import functools

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.experimental.pallas as pl
import numpy as np

# ---------------------------------------------------------------------------
# Global state set by prepare_tables()
# ---------------------------------------------------------------------------
_MONT = None          # dict of precomputed constants + tables
_NTT_PALLAS = None   # compiled pallas callable (GPU only)

_IS_GPU = jax.default_backend() == "gpu"


# ---------------------------------------------------------------------------
# Montgomery helpers (numpy, used only in prepare_tables)
# ---------------------------------------------------------------------------

def _compute_montgomery_constants(q: int) -> dict:
    R = 1 << 32
    q_inv       = pow(q, -1, R)
    q_neg_inv   = (R - q_inv) % R      # -q^{-1} mod 2^32
    r_mod_q     = R % q
    r2_mod_q    = (R * R) % q
    return {
        "q":        np.uint64(q),
        "q_inv":    np.uint64(q_neg_inv),
        "r_mod_q":  np.uint64(r_mod_q),
        "r2_mod_q": np.uint64(r2_mod_q),
    }


def _np_to_mont(a: np.ndarray, mc: dict) -> np.ndarray:
    """a * R mod q  (standard Montgomery form)."""
    return ((a.astype(np.uint64) * int(mc["r_mod_q"])) % int(mc["q"])).astype(np.uint32)


def _np_to_pretwist(a: np.ndarray, mc: dict) -> np.ndarray:
    """a * R^2 mod q  — so REDC(x_normal * pretwist[n]) = x*a*R (fused convert+twist)."""
    return ((a.astype(np.uint64) * int(mc["r2_mod_q"])) % int(mc["q"])).astype(np.uint32)


# ---------------------------------------------------------------------------
# Public modular arithmetic (hot path)
# ---------------------------------------------------------------------------

def mod_add(a, b, q):
    """(a + b) mod q  — stays in uint32 (q < 2^31 ⟹ a+b < 2^32)."""
    s   = a + b                           # uint32, no overflow
    q32 = jnp.uint32(q)
    return jnp.where(s >= q32, s - q32, s)


def mod_sub(a, b, q):
    """(a - b) mod q  — stays in uint32."""
    q32 = jnp.uint32(q)
    return jnp.where(a >= b, a - b, a + q32 - b)


def mod_mul(a, b, q):
    """(a * b) mod q  — needs uint64 for the product."""
    return (a.astype(jnp.uint64) * b.astype(jnp.uint64) % jnp.uint64(q)).astype(jnp.uint32)


# ---------------------------------------------------------------------------
# Montgomery multiply
# ---------------------------------------------------------------------------

def _mont_mul(a, b):
    """REDC(a*b) = a*b*R^{-1} mod q.

    Optimisation: m uses uint32 wrapping multiply (low 32 bits of T*q_inv)
    instead of a 64-bit multiply followed by a 64-bit AND.
    """
    mc       = _MONT
    q64      = mc["q_jax"]
    q_inv_u32 = mc["q_inv_u32"]

    T = a.astype(jnp.uint64) * b.astype(jnp.uint64)   # 1 × u64 mul
    m = T.astype(jnp.uint32) * q_inv_u32               # u32 wrap = low 32 bits of T*q_inv
    t = (T + m.astype(jnp.uint64) * q64) >> jnp.uint64(32)
    return jnp.where(t >= q64, t - q64, t).astype(jnp.uint32)


def _mont_to_normal(a):
    return _mont_mul(a, jnp.uint32(1))


# ---------------------------------------------------------------------------
# Bit-reversal permutation
# ---------------------------------------------------------------------------

def _make_bit_reversal(N: int) -> jnp.ndarray:
    bits = int(N).bit_length() - 1
    idx  = np.arange(N, dtype=np.int32)
    rev  = np.zeros(N, dtype=np.int32)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return jnp.asarray(rev)


# ---------------------------------------------------------------------------
# Pallas (GPU-fused) NTT kernel
# ---------------------------------------------------------------------------

def _build_pallas_ntt(N: int, num_stages: int, q_int: int, q_inv_int: int,
                      psi_pretwist: jnp.ndarray, tw_mont: jnp.ndarray,
                      bit_rev: jnp.ndarray):
    """Return a (batch, N) -> (batch, N) callable backed by a single Triton kernel."""

    # Capture all constants in the closure so XLA folds them.
    q64       = jnp.uint64(q_int)
    q_inv_u32 = jnp.uint32(q_inv_int)
    q32       = jnp.uint32(q_int)

    # ---- inner ops (no global state) -----------------------------------
    def mm(a, b):
        T = a.astype(jnp.uint64) * b.astype(jnp.uint64)
        m = T.astype(jnp.uint32) * q_inv_u32
        t = (T + m.astype(jnp.uint64) * q64) >> jnp.uint64(32)
        return jnp.where(t >= q64, t - q64, t).astype(jnp.uint32)

    def ma(a, b):
        s = a + b
        return jnp.where(s >= q32, s - q32, s)

    def ms(a, b):
        return jnp.where(a >= b, a - b, a + q32 - b)

    # ---- kernel (processes one row = one batch element) ----------------
    def _kernel(x_ref, psi_ref, tw_ref, birev_ref, out_ref):
        x = x_ref[:]          # (N,) uint32
        psi = psi_ref[:]      # (N,) uint32
        birev = birev_ref[:]  # (N,) int32

        # Fused convert-to-Montgomery + negacyclic twist
        x = mm(x, psi)
        # Bit-reversal permutation
        x = x[birev]

        # Cooley-Tukey DIT butterfly stages — all in registers
        for s in range(num_stages):
            span = 1 << s
            ng   = N >> (s + 1)           # num_groups
            w    = tw_ref[span : 2 * span]  # (span,) — static slice
            x3   = x.reshape(ng, 2 * span)
            lo, hi = x3[:, :span], x3[:, span:]
            t    = mm(hi, w[jnp.newaxis, :])
            x    = jnp.concatenate([ma(lo, t), ms(lo, t)], axis=-1).reshape(N)

        # Convert from Montgomery: REDC(x · 1) = x · R^{-1} mod q
        T_f = x.astype(jnp.uint64)
        m_f = T_f.astype(jnp.uint32) * q_inv_u32
        t_f = (T_f + m_f.astype(jnp.uint64) * q64) >> jnp.uint64(32)
        out_ref[:] = jnp.where(t_f >= q64, t_f - q64, t_f).astype(jnp.uint32)

    # ---- single-row pallas call ----------------------------------------
    ntt_row = pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((N,), jnp.uint32),
    )

    # ---- vmap over batch dimension -------------------------------------
    # psi, twiddles, bit_rev are shared (in_axes=None); x varies (in_axes=0)
    ntt_batched = jax.vmap(
        lambda xi: ntt_row(xi, psi_pretwist, tw_mont, bit_rev),
        in_axes=0,
    )

    return ntt_batched


# ---------------------------------------------------------------------------
# Pure-JAX NTT (CPU and fallback)
# ---------------------------------------------------------------------------

def _ntt_jax(x, q):
    mc         = _MONT
    batch      = x.shape[0]
    N          = x.shape[1]
    num_stages = N.bit_length() - 1

    # Fused convert + twist
    a = _mont_mul(x, mc["psi_pretwist"][jnp.newaxis, :])

    # Bit-reversal
    a = a[:, mc["bit_rev"]]

    # Butterfly stages
    stage_twiddles = mc["stage_twiddles"]
    q32 = mc["q32"]
    for s in range(num_stages):
        span = 1 << s
        ng   = N >> (s + 1)
        a    = a.reshape(batch, ng, 2 * span)
        lo, hi = a[:, :, :span], a[:, :, span:]
        w    = stage_twiddles[s][jnp.newaxis, jnp.newaxis, :]
        t    = _mont_mul(hi, w)
        # uint32 mod_add / mod_sub (no cast to uint64)
        s_lo = lo + t
        new_lo = jnp.where(s_lo >= q32, s_lo - q32, s_lo)
        new_hi = jnp.where(lo >= t, lo - t, lo + q32 - t)
        a = jnp.concatenate([new_lo, new_hi], axis=-1).reshape(batch, N)

    return _mont_to_normal(a)


# ---------------------------------------------------------------------------
# Public API: ntt
# ---------------------------------------------------------------------------

def ntt(x, *, q, psi_powers, twiddles):
    """
    Forward negacyclic NTT.

    Args:
        x:          (batch, N) uint32 input, values in [0, q)
        q:          prime modulus
        psi_powers: precomputed ψ-power table (shape N, already converted)
        twiddles:   precomputed twiddle table (shape N, already converted)

    Returns:
        (batch, N) uint32 output
    """
    if _NTT_PALLAS is not None:
        return _NTT_PALLAS(x)
    if _MONT is not None:
        return _ntt_jax(x, q)

    # --- no prepare_tables called: plain modular arithmetic fallback ---
    batch      = x.shape[0]
    N          = x.shape[1]
    num_stages = N.bit_length() - 1
    a = mod_mul(x, psi_powers[jnp.newaxis, :], q)
    a = a[:, _make_bit_reversal(N)]
    for s in range(num_stages):
        span = 1 << s
        ng   = N >> (s + 1)
        a    = a.reshape(batch, ng, 2 * span)
        lo, hi = a[:, :, :span], a[:, :, span:]
        w  = twiddles[span : 2 * span][jnp.newaxis, jnp.newaxis, :]
        t  = mod_mul(hi, w, q)
        a  = jnp.concatenate([mod_add(lo, t, q), mod_sub(lo, t, q)], axis=-1)
        a  = a.reshape(batch, N)
    return a


# ---------------------------------------------------------------------------
# Public API: prepare_tables
# ---------------------------------------------------------------------------

def prepare_tables(*, q, psi_powers, twiddles):
    """
    Precompute and cache all constants needed by ntt().

    * On GPU: compiles a Pallas kernel that fuses every butterfly stage.
    * On CPU: precomputes typed JAX constants used by the pure-JAX path.

    Cost excluded from benchmark timing.
    """
    global _MONT, _NTT_PALLAS

    q_int = int(q)
    mc    = _compute_montgomery_constants(q_int)

    psi_np = np.asarray(psi_powers, dtype=np.uint32)
    tw_np  = np.asarray(twiddles,   dtype=np.uint32)
    N      = len(psi_np)
    num_stages = N.bit_length() - 1

    # psi_pretwist[n] = ψ^n · R² mod q
    # REDC(x_normal · pretwist[n]) = x·ψ^n·R  (Montgomery form of x·ψ^n)
    psi_pretwist = _np_to_pretwist(psi_np, mc)

    # twiddles in Montgomery form: val · R mod q
    tw_mont = _np_to_mont(tw_np, mc)

    # Per-stage twiddle slices as JAX arrays (avoid runtime indexing in hot loop)
    stage_twiddles = [
        jnp.asarray(tw_mont[1 << s : 1 << (s + 1)], dtype=jnp.uint32)
        for s in range(num_stages)
    ]

    bit_rev      = _make_bit_reversal(N)
    psi_jax      = jnp.asarray(psi_pretwist, dtype=jnp.uint32)
    tw_jax       = jnp.asarray(tw_mont,      dtype=jnp.uint32)

    _MONT = {
        "q_jax":          jnp.uint64(q_int),
        "q_inv_u32":      jnp.uint32(int(mc["q_inv"])),
        "q32":            jnp.uint32(q_int),
        "bit_rev":        bit_rev,
        "psi_pretwist":   psi_jax,
        "stage_twiddles": stage_twiddles,
    }
    _MONT.update(mc)

    # Build the Pallas kernel on GPU; skip on CPU (pallas CPU = interpret only)
    _NTT_PALLAS = None
    if _IS_GPU:
        try:
            _NTT_PALLAS = _build_pallas_ntt(
                N, num_stages, q_int, int(mc["q_inv"]),
                psi_jax, tw_jax, bit_rev,
            )
        except Exception:
            _NTT_PALLAS = None

    return psi_jax, tw_jax
