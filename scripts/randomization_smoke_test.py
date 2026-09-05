"""Batch-64 Warp/MJX smoke test for staged domain randomization.

This is deliberately not a rollout logger: it measures reset and one compiled
control step, prints fixed-per-lane parameter bounds, and saves only two
contact sheets for human inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from configs.domain_randomization import NO_RANDOMIZATION, STAGED_RANDOMIZATION
from sim.env import So101Env, raise_if_nonfinite
from sim.randomization import parameter_summary


BATCH_SIZE = 64
SAMPLE_DIR = Path("scripts/randomization_samples")


def save_contact_sheet(frames: jax.Array, path: Path) -> None:
    """Write 64 128px RGB frames as an 8 by 8 1024px contact sheet."""
    array = np.asarray(frames)
    height, width = array.shape[1:3]
    sheet = np.zeros((height * 8, width * 8, 3), dtype=np.uint8)
    for index, frame in enumerate(array):
        row, column = divmod(index, 8)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = frame
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet, "RGB").save(path)


def run_profile(name: str, config) -> None:
    env = So101Env.from_scene(BATCH_SIZE, randomization=config)
    keys = jax.random.split(jax.random.key(20260905), BATCH_SIZE)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    started = time.perf_counter()
    carry, observation = reset(keys)
    jax.block_until_ready(observation["wrist"])
    reset_seconds = time.perf_counter() - started

    action = jnp.zeros((BATCH_SIZE, 6), dtype=jnp.float32)
    # Compile (or recover the cached executable) separately from steady-state.
    started = time.perf_counter()
    carry, observation, output = step(carry, action)
    jax.block_until_ready(observation["wrist"])
    first_step_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(10):
        carry, observation, output = step(carry, action)
    jax.block_until_ready(observation["wrist"])
    steady_seconds = time.perf_counter() - started
    raise_if_nonfinite(output)

    print(f"\n[{name}]")
    print(f"reset (compile included): {reset_seconds:.3f} s")
    print(f"first step (compile included): {first_step_seconds:.3f} s")
    print(f"steady throughput: {BATCH_SIZE * 10 / steady_seconds:.1f} env-steps/s")
    print("final time:", np.asarray(carry.data.time)[0])
    print("finite lanes:", int(np.asarray(output.diagnostics.physics_finite).sum()), "/", BATCH_SIZE)
    for parameter, bounds in parameter_summary(carry.params).items():
        print(f"{parameter}: {np.asarray(bounds)}")
    if name == "staged":
        save_contact_sheet(observation["wrist"], SAMPLE_DIR / "staged_wrist_contact_sheet.png")
        save_contact_sheet(observation["overhead"], SAMPLE_DIR / "staged_overhead_contact_sheet.png")
        print("samples:", SAMPLE_DIR / "staged_wrist_contact_sheet.png")
        print("samples:", SAMPLE_DIR / "staged_overhead_contact_sheet.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-randomization-only",
        action="store_true",
        help="run only the nominal comparison profile",
    )
    args = parser.parse_args()
    if not args.no_randomization_only:
        run_profile("staged", STAGED_RANDOMIZATION)
    run_profile("no_randomization", NO_RANDOMIZATION)


if __name__ == "__main__":
    main()
