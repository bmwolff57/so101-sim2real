"""Robot-only proprioception for the SO-101 policy contract."""

from __future__ import annotations

import jax.numpy as jnp


def assemble_proprio(
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    previous_action: jnp.ndarray,
    command_target: jnp.ndarray,
) -> jnp.ndarray:
    """Concatenate only robot/control signals into float32 policy input.

    Cube, tub, contact, reward, and terminal truth are intentionally absent.
    Inputs have a leading batch dimension and six robot/control values each.
    """
    return jnp.concatenate(
        (qpos, qvel, previous_action, command_target), axis=-1
    ).astype(jnp.float32)
