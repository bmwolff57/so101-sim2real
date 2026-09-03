"""Autoreset helpers kept outside the environment step function."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _select_leaf(done: jnp.ndarray, reset_leaf, stepped_leaf):
    if not hasattr(stepped_leaf, "ndim") or stepped_leaf.ndim == 0:
        return stepped_leaf
    mask = done.reshape(done.shape + (1,) * (stepped_leaf.ndim - done.ndim))
    return jnp.where(mask, reset_leaf, stepped_leaf)


def select_reset(done: jnp.ndarray, reset_value, stepped_value):
    """Choose reset state only for done lanes, preserving terminal output.

    The caller emits the terminal `StepOutput` before calling this helper. It
    therefore cannot hide terminal rewards or flags inside `env.step`.
    """
    return jax.tree.map(
        lambda reset_leaf, stepped_leaf: _select_leaf(done, reset_leaf, stepped_leaf),
        reset_value,
        stepped_value,
    )
