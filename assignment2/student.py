"""
Assignment 2: Sumcheck prover implementation in JAX.

Protocol recap (prover side only):
  For each of n rounds, the prover sends evaluations of the univariate
  round polynomial g_i(t) at t = 0, 1, ..., degree, where
      g_i(t) = sum_{remaining Boolean inputs} f(t, x_{i+1}, ..., x_n)
  The verifier checks g_i(0)+g_i(1) = claim_{i-1} then samples a
  challenge r_i and folds the tables:
      table[j] <- mle_update(table[2j], table[2j+1], r_i)

Only the 32-bit track is compulsory. 64-bit is implemented as well.
"""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# 32-bit primitives  (compulsory)
# ---------------------------------------------------------------------------

def mod_add_32(a, b, q):
    """(a + b) mod q — 32-bit.

    q can be up to MAX_PRIME_32 ~ 2^32-5, so a+b may exceed 2^32.
    Use uint64 intermediate to avoid overflow.
    """
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    s   = a.astype(jnp.uint64) + b.astype(jnp.uint64)
    return jnp.where(s >= q64, s - q64, s).astype(jnp.uint32)


def mod_sub_32(a, b, q):
    """(a - b) mod q — 32-bit."""
    a64 = a.astype(jnp.uint64)
    b64 = b.astype(jnp.uint64)
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    return jnp.where(a64 >= b64, a64 - b64, a64 + q64 - b64).astype(jnp.uint32)


def mod_mul_32(a, b, q):
    """(a * b) mod q — 32-bit.  Product fits in uint64."""
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    return (a.astype(jnp.uint64) * b.astype(jnp.uint64) % q64).astype(jnp.uint32)


# ---------------------------------------------------------------------------
# 64-bit primitives  (optional track)
# ---------------------------------------------------------------------------

def mod_add_64(a, b, q):
    """(a + b) mod q — 64-bit.

    Addition can overflow uint64 (q ~ 2^64-59).
    We detect the wrap via a >= 2^64 - b and correct with 2^64 mod q.
    """
    MASK = jnp.uint64(0xFFFFFFFFFFFFFFFF)
    q64  = jnp.asarray(q, dtype=jnp.uint64)
    a64  = jnp.asarray(a, dtype=jnp.uint64)
    b64  = jnp.asarray(b, dtype=jnp.uint64)

    s       = a64 + b64
    wrapped = a64 > (MASK - b64)        # True iff mathematical sum > 2^64-1

    # 2^64 mod q  (= -q mod 2^64 for uint64 arithmetic = MASK - q + 1 = 2^64-q)
    neg_q        = MASK - q64 + jnp.uint64(1)
    pow2_64_modq = jnp.where(neg_q >= q64, neg_q - q64, neg_q)

    s_corr = jnp.where(wrapped, s + pow2_64_modq, s)
    return jnp.where(s_corr >= q64, s_corr - q64, s_corr)


def mod_sub_64(a, b, q):
    """(a - b) mod q — 64-bit."""
    a64 = jnp.asarray(a, dtype=jnp.uint64)
    b64 = jnp.asarray(b, dtype=jnp.uint64)
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    return jnp.where(a64 >= b64, a64 - b64, a64 + q64 - b64)


def mod_mul_64(a, b, q):
    """(a * b) mod q — 64-bit without requiring a uint128 dtype."""
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    x = jnp.asarray(a, dtype=jnp.uint64) % q64
    y = jnp.asarray(b, dtype=jnp.uint64)
    acc = jnp.zeros_like(x, dtype=jnp.uint64)

    # Russian peasant multiplication: keep every intermediate reduced mod q,
    # so uint64 overflow is handled only through mod_add_64.
    for _ in range(64):
        acc = jnp.where((y & jnp.uint64(1)) != 0, mod_add_64(acc, x, q64), acc)
        x = mod_add_64(x, x, q64)
        y = y >> jnp.uint64(1)
    return acc


# ---------------------------------------------------------------------------
# 128-bit primitives  (optional, not required)
# ---------------------------------------------------------------------------

def mod_add_128(a, b, q):
    raise NotImplementedError

def mod_sub_128(a, b, q):
    raise NotImplementedError

def mod_mul_128(a, b, q):
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Frozen dispatch API
# ---------------------------------------------------------------------------

def mod_add(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_add_32(a, b, q)
    if int(bit_width) == 64:
        return mod_add_64(a, b, q)
    if int(bit_width) == 128:
        return mod_add_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def mod_sub(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_sub_32(a, b, q)
    if int(bit_width) == 64:
        return mod_sub_64(a, b, q)
    if int(bit_width) == 128:
        return mod_sub_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def mod_mul(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_mul_32(a, b, q)
    if int(bit_width) == 64:
        return mod_mul_64(a, b, q)
    if int(bit_width) == 128:
        return mod_mul_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


# ---------------------------------------------------------------------------
# MLE update
# ---------------------------------------------------------------------------

def mle_update_32(zero_eval, one_eval, target_eval, *, q):
    """(one - zero) * target + zero  mod q  — 32-bit linear interpolation."""
    diff   = mod_sub_32(one_eval, zero_eval, q)
    scaled = mod_mul_32(diff, jnp.asarray(target_eval, dtype=jnp.uint32), q)
    return mod_add_32(scaled, zero_eval, q)


def mle_update_64(zero_eval, one_eval, target_eval, *, q):
    """64-bit MLE update."""
    diff   = mod_sub_64(one_eval, zero_eval, q)
    scaled = mod_mul_64(diff, jnp.asarray(target_eval, dtype=jnp.uint64), q)
    return mod_add_64(scaled, zero_eval, q)


def mle_update_128(zero_eval, one_eval, target_eval, *, q):
    raise NotImplementedError


def mle_update(zero_eval, one_eval, target_eval, *, q, bit_width=32):
    if int(bit_width) == 32:
        return mle_update_32(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 64:
        return mle_update_64(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 128:
        return mle_update_128(zero_eval, one_eval, target_eval, q=q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


# ---------------------------------------------------------------------------
# Core sumcheck engine
# ---------------------------------------------------------------------------

def _reduce_sum_64(arr, q):
    """Sum a uint64 array mod q safely (avoids uint64 overflow).

    Pairwise tree reduction: at each level, add adjacent pairs then reduce
    mod q, so values never exceed 2*q < 2^65 — which fits in uint64 since
    q < 2^64 - 59 means 2*q wraps but we catch it with the conditional sub.
    """
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    x   = arr % q64
    while x.shape[0] > 1:
        # Pad if odd length
        if x.shape[0] % 2 == 1:
            x = jnp.concatenate([x, jnp.zeros(1, dtype=jnp.uint64)])
        lo  = x[0::2]
        hi  = x[1::2]
        x   = mod_add_64(lo, hi, q64)
    return x[0]


def _sumcheck_core(eval_tables, *, q, expression, challenges, num_rounds,
                   dtype, add_fn, sub_fn, mul_fn, mle_fn):
    """Shared sumcheck logic for all bit-widths.

    Args:
        eval_tables : dict {var: jnp.array}
        q           : modulus scalar (typed JAX array)
        expression  : list[list[str]] — outer=additive terms, inner=factors
        challenges  : 1-D JAX array, length = num_rounds - 1
        num_rounds  : total rounds (= num_vars)
        dtype       : jnp.uint32 or jnp.uint64
        *_fn        : typed arithmetic callables

    Returns:
        (claim0, round_evals) — both JAX arrays.
        round_evals shape: (num_rounds, degree+1)
    """
    degree   = max(len(term) for term in expression)
    used     = set(f for term in expression for f in term)

    # Working tables (only used variables, correct dtype)
    tables = {
        name: jnp.asarray(eval_tables[name], dtype=dtype)
        for name in used
    }

    all_round_evals = []

    for round_idx in range(num_rounds):
        # Split by LSB: even indices → free var = 0, odd → free var = 1
        zero_vals = {name: tables[name][0::2] for name in tables}
        one_vals  = {name: tables[name][1::2] for name in tables}

        round_t_list = []
        for t in range(degree + 1):
            # Values of each variable at evaluation point t
            if t == 0:
                var_t = zero_vals
            elif t == 1:
                var_t = one_vals
            else:
                t_typed = jnp.asarray(t, dtype=dtype)
                var_t = {
                    name: mle_fn(zero_vals[name], one_vals[name], t_typed, q=q)
                    for name in tables
                }

            # Evaluate the composed expression for every pair simultaneously
            total = None
            for term in expression:
                term_val = None
                for factor in term:
                    fv = var_t[factor]
                    term_val = fv if term_val is None else mul_fn(term_val, fv, q)
                total = term_val if total is None else add_fn(total, term_val, q)

            # Reduce sum of all pair contributions mod q
            if dtype == jnp.uint32:
                # sum of M < 2^20 values each < q < 2^32 → total < 2^52 ⊂ uint64
                q64  = jnp.asarray(q, dtype=jnp.uint64)
                g_t  = (jnp.sum(total.astype(jnp.uint64)) % q64).astype(jnp.uint32)
            else:
                # 64-bit: pairwise tree-reduce to avoid overflow
                g_t = _reduce_sum_64(total, q)

            round_t_list.append(g_t)

        all_round_evals.append(jnp.stack(round_t_list))

        # Fold tables with this round's challenge.
        # challenges has length num_rounds-1, so skip the last round.
        if round_idx < challenges.shape[0]:
            r = challenges[round_idx]
            tables = {
                name: mle_fn(zero_vals[name], one_vals[name], r, q=q)
                for name in tables
            }

    # (num_rounds, degree+1)
    round_evals = jnp.stack(all_round_evals)

    # claim0 = g_0(0) + g_0(1) = total sum of f over the Boolean cube
    claim0 = add_fn(all_round_evals[0][0], all_round_evals[0][1], q)

    return claim0, round_evals


# ---------------------------------------------------------------------------
# Public sumcheck entry points
# ---------------------------------------------------------------------------

def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    """Compulsory 32-bit sumcheck path."""
    return _sumcheck_core(
        eval_tables,
        q=q,
        expression=expression,
        challenges=jnp.asarray(challenges, dtype=jnp.uint32),
        num_rounds=num_rounds,
        dtype=jnp.uint32,
        add_fn=mod_add_32,
        sub_fn=mod_sub_32,
        mul_fn=mod_mul_32,
        mle_fn=mle_update_32,
    )


def sumcheck_64(eval_tables, *, q, expression, challenges, num_rounds):
    """Optional 64-bit sumcheck path."""
    return _sumcheck_core(
        eval_tables,
        q=q,
        expression=expression,
        challenges=jnp.asarray(challenges, dtype=jnp.uint64),
        num_rounds=num_rounds,
        dtype=jnp.uint64,
        add_fn=mod_add_64,
        sub_fn=mod_sub_64,
        mul_fn=mod_mul_64,
        mle_fn=mle_update_64,
    )


def sumcheck_128(eval_tables, *, q, expression, challenges, num_rounds):
    """Optional 128-bit sumcheck path."""
    raise NotImplementedError


def sumcheck(eval_tables, *, q, expression, challenges, num_rounds, bit_width=32):
    """Frozen dispatcher entrypoint used by the harness."""
    if int(bit_width) == 32:
        return sumcheck_32(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    if int(bit_width) == 64:
        return sumcheck_64(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    if int(bit_width) == 128:
        return sumcheck_128(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    raise ValueError(f"Unsupported bit_width={bit_width}")
