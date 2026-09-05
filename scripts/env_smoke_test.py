"""Run the SO-101 environment under JIT and internal VMAP."""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from sim.env import So101Env, raise_if_nonfinite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=10)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    print(f"[jax ] devices: {jax.devices()}")
    print(f"[jax ] backend: {jax.default_backend()}")
    env = So101Env.from_scene(batch_size=args.batch_size)
    keys = jax.random.split(jax.random.key(0), args.batch_size)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    started = time.perf_counter()
    carry, observation = reset(keys)
    jax.block_until_ready(observation["wrist"])
    reset_s = time.perf_counter() - started
    print(f"[reset] compile + run: {reset_s:.3f}s")
    print(
        "[obs  ] "
        f"wrist={observation['wrist'].shape}/{observation['wrist'].dtype}, "
        f"overhead={observation['overhead'].shape}/{observation['overhead'].dtype}, "
        f"proprio={observation['proprio'].shape}/{observation['proprio'].dtype}"
    )

    action = jnp.zeros((args.batch_size, 6), dtype=jnp.float32)
    started = time.perf_counter()
    for _ in range(args.steps):
        carry, observation, output = step(carry, action)
    jax.block_until_ready(observation["wrist"])
    elapsed = time.perf_counter() - started
    raise_if_nonfinite(output)
    print(
        f"[step ] {args.steps} control steps, batch={args.batch_size}: "
        f"{args.steps * args.batch_size / elapsed:.1f} environment-steps/s"
    )
    print(
        f"[state] step={np.asarray(carry.step)[:3].tolist()}, "
        f"reward={np.asarray(output.reward)[:3].tolist()}, "
        f"terminated={np.asarray(output.terminated)[:3].tolist()}, "
        f"truncated={np.asarray(output.truncated)[:3].tolist()}"
    )
    print("[pass] environment smoke test passed")


if __name__ == "__main__":
    main(parse_args())
