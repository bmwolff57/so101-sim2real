"""
Render throughput measurement for sim/scene_cube_tub.xml.

Purpose (Step 3): produce THE number that sizes the whole project.
Measures wallclock per step for physics-only vs physics+rendering,
across a sweep of batch sizes.  Rendering uses mujoco.Renderer on
one host per batch (NOT batched on GPU the way MJX physics is —
that's the whole point of measuring).

Run from project root (with the mjx venv active):
    python scripts/render_check.py
    python scripts/render_check.py --batch-sizes 1 64 256 1024
    python scripts/render_check.py --render-resolution 96
    python scripts/render_check.py --arm-mesh-collisions

What to look for:
  - Physics-only steps/sec by batch size — should scale close to
    linearly on GPU (that's the MJX payoff).  Ratio of
    wallclock-per-step at batch=N vs batch=1 should be 2-3x or
    less; if it's 10x, batching is broken.
  - Rendering cost per frame per camera at your chosen resolution.
  - Combined (physics + render) throughput.  This is the number
    you write in docs/measurements.md.
  - If OOM on the 4070, DROP RESOLUTION before dropping batch.
    A 128->96 shrink is ~45% less memory + faster inference.

Requires:
  - sim/scene_cube_tub.xml with two cameras named "wrist" and
    "overhead" (added in Step 3 setup).
  - jax, mujoco, mujoco-mjx, numpy, pillow.
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from PIL import Image

SCENE_PATH = "sim/scene_cube_tub.xml"
DEFAULT_BATCH_SIZES = (1, 64, 256, 1024)
DEFAULT_RENDER_RES = 128
DEFAULT_STEPS_PER_MEASUREMENT = 200
DEFAULT_RENDER_EVERY = 4  # render one obs per N physics steps (typical for RL)
CAMERA_NAMES = ("wrist", "overhead")
SAMPLES_DIR = Path("scripts/render_samples")
FULL_MESH_MIN_GPU_MEMORY_GIB = 16.0


# ---------------------------------------------------------------------------
# CLI + config
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help="Batch sizes to sweep. Default: 1 64 256 1024.",
    )
    parser.add_argument(
        "--render-resolution",
        type=int,
        default=DEFAULT_RENDER_RES,
        help="Square resolution (HxW) for both cameras. Default: 128.",
    )
    parser.add_argument(
        "--steps-per-measurement",
        type=int,
        default=DEFAULT_STEPS_PER_MEASUREMENT,
        help="Physics steps to time per config. Default: 200.",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=DEFAULT_RENDER_EVERY,
        help="Render obs once per N physics steps. Default: 4.",
    )
    parser.add_argument(
        "--arm-mesh-collisions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override auto arm mesh collision selection. Default: enable "
            f"only on GPUs with >= {FULL_MESH_MIN_GPU_MEMORY_GIB:g} GiB VRAM."
        ),
    )
    parser.add_argument(
        "--save-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"Save one wrist+overhead PNG to {SAMPLES_DIR}. Default: on.",
    )
    return parser.parse_args()


def gpu_total_memory_gib(device_index: int = 0) -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        memory_mib = [float(line) for line in result.stdout.splitlines() if line.strip()]
        return memory_mib[device_index] / 1024
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError, ValueError):
        return None


def select_arm_mesh_collisions(requested: bool | None) -> bool:
    if requested is not None:
        state = "enabled" if requested else "disabled"
        print(f"[config] arm mesh collisions: {state} (explicit CLI override)")
        return requested

    total_gib = gpu_total_memory_gib()
    if total_gib is None:
        print("[config] GPU VRAM unavailable; arm mesh collisions: disabled")
        return False

    enabled = total_gib >= FULL_MESH_MIN_GPU_MEMORY_GIB
    state = "enabled" if enabled else "disabled"
    print(
        f"[config] GPU VRAM: {total_gib:.1f} GiB; arm mesh collisions: {state} "
        f"(threshold: {FULL_MESH_MIN_GPU_MEMORY_GIB:g} GiB)"
    )
    return enabled


def configure_arm_mesh_collisions(mj_model: mujoco.MjModel, enabled: bool) -> None:
    collision_geoms = mj_model.geom_group == 3
    count = int(np.count_nonzero(collision_geoms))
    if count == 0:
        raise RuntimeError("No arm collision geoms found in group 3.")
    mask_value = 1 if enabled else 0
    mj_model.geom_contype[collision_geoms] = mask_value
    mj_model.geom_conaffinity[collision_geoms] = mask_value
    state = "enabled" if enabled else "disabled"
    print(f"[config] arm mesh collision state: {state} ({count} geoms in group 3)")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def verify_cameras_exist(mj_model: mujoco.MjModel) -> None:
    """Confirm both expected cameras are declared in the scene."""
    for cam_name in CAMERA_NAMES:
        cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id < 0:
            raise RuntimeError(
                f"Camera '{cam_name}' not found in {SCENE_PATH}. "
                "Check Step 3 XML setup — expected named cameras: "
                f"{', '.join(CAMERA_NAMES)}."
            )
    print(f"[cams] verified {len(CAMERA_NAMES)} cameras: {', '.join(CAMERA_NAMES)}")


def save_sample_frames(
    mj_model: mujoco.MjModel,
    resolution: int,
) -> None:
    """Render one frame per camera from initial state; save as PNGs.

    This is not a timing measurement — it's a sanity check that the
    cameras render sensible views before you commit to the throughput
    numbers. Look at the images.
    """
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    renderer = mujoco.Renderer(mj_model, height=resolution, width=resolution)
    for cam_name in CAMERA_NAMES:
        renderer.update_scene(mj_data, camera=cam_name)
        pixels = renderer.render()
        out_path = SAMPLES_DIR / f"{cam_name}_{resolution}x{resolution}.png"
        Image.fromarray(pixels).save(out_path)
        print(f"[samp] saved {out_path}")
    renderer.close()


# ---------------------------------------------------------------------------
# Physics timing (batched via vmap)
# ---------------------------------------------------------------------------
def time_physics_only(
    mjx_model,
    batch_size: int,
    n_steps: int,
) -> tuple[float, float]:
    """Return (wallclock_seconds, steps_per_sec) for batched physics-only stepping."""
    # Build a batch of initial states via vmap.
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

    def make_data(_key):
        return mjx.make_data(mjx_model)

    batched_data = jax.vmap(make_data)(keys)
    batched_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

    # Warm-up (trace + compile), not counted.
    batched_data = batched_step(mjx_model, batched_data)
    batched_data.qpos.block_until_ready()

    # Timed run.
    t0 = time.perf_counter()
    for _ in range(n_steps):
        batched_data = batched_step(mjx_model, batched_data)
    batched_data.qpos.block_until_ready()
    wall = time.perf_counter() - t0

    total_steps = n_steps * batch_size
    return wall, total_steps / wall


# ---------------------------------------------------------------------------
# Render timing (per-env, on host — the whole point of the measurement)
# ---------------------------------------------------------------------------
def time_render_only(
    mj_model: mujoco.MjModel,
    batch_size: int,
    resolution: int,
    n_renders: int,
) -> tuple[float, float]:
    """Return (wallclock_seconds, frames_per_sec) for CPU-side per-env rendering.

    Renders BOTH cameras (wrist + overhead) once per env per iteration.
    Uses a single mujoco.Renderer instance re-used across envs (the
    typical MJX+render pattern — physics state is copied from MJX back
    into an MjData, then rendered).
    """
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=resolution, width=resolution)

    # Give MJX-side qpos to render.  For a timing measurement, we don't
    # need real batched physics state — we just need SOMETHING valid to
    # feed into update_scene.  Zero qvel + default qpos is fine.
    mujoco.mj_forward(mj_model, mj_data)

    t0 = time.perf_counter()
    for _ in range(n_renders):
        for _env_idx in range(batch_size):
            # In a real training loop you'd copy per-env qpos/qvel here.
            # For timing, we skip that (measures the render call cost).
            for cam_name in CAMERA_NAMES:
                renderer.update_scene(mj_data, camera=cam_name)
                _ = renderer.render()
    wall = time.perf_counter() - t0
    renderer.close()

    # Frames rendered = n_renders * batch_size * len(CAMERA_NAMES)
    total_frames = n_renders * batch_size * len(CAMERA_NAMES)
    return wall, total_frames / wall


# ---------------------------------------------------------------------------
# Combined (physics + render at RENDER_EVERY cadence)
# ---------------------------------------------------------------------------
def time_combined(
    mj_model: mujoco.MjModel,
    mjx_model,
    batch_size: int,
    resolution: int,
    n_steps: int,
    render_every: int,
) -> tuple[float, float, int]:
    """Physics steps + render(both cameras) every `render_every` steps.

    Returns (wallclock, effective_step_throughput, n_renders_performed).
    Effective throughput is (n_steps * batch_size) / wallclock — the same
    denominator as physics-only, so the numbers are directly comparable.
    """
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

    def make_data(_key):
        return mjx.make_data(mjx_model)

    batched_data = jax.vmap(make_data)(keys)
    batched_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

    # Warm-up.
    batched_data = batched_step(mjx_model, batched_data)
    batched_data.qpos.block_until_ready()

    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=resolution, width=resolution)

    n_renders = 0
    t0 = time.perf_counter()
    for step_idx in range(n_steps):
        batched_data = batched_step(mjx_model, batched_data)

        if (step_idx + 1) % render_every == 0:
            batched_data.qpos.block_until_ready()
            # Pull batch qpos to host, render each env's frame.
            qpos_batch = np.asarray(batched_data.qpos)
            qvel_batch = np.asarray(batched_data.qvel)
            for env_idx in range(batch_size):
                mj_data.qpos[:] = qpos_batch[env_idx]
                mj_data.qvel[:] = qvel_batch[env_idx]
                mujoco.mj_forward(mj_model, mj_data)
                for cam_name in CAMERA_NAMES:
                    renderer.update_scene(mj_data, camera=cam_name)
                    _ = renderer.render()
            n_renders += 1

    batched_data.qpos.block_until_ready()
    wall = time.perf_counter() - t0
    renderer.close()

    total_steps = n_steps * batch_size
    return wall, total_steps / wall, n_renders


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    print(f"[jax ] devices: {jax.devices()}")
    print(f"[jax ] default backend: {jax.default_backend()}\n")

    # Load and configure.
    mj_model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    print(f"[load] MJCF parsed. nbody={mj_model.nbody}, ngeom={mj_model.ngeom}")
    arm_mesh_on = select_arm_mesh_collisions(args.arm_mesh_collisions)
    configure_arm_mesh_collisions(mj_model, arm_mesh_on)
    verify_cameras_exist(mj_model)

    if args.save_samples:
        print()
        save_sample_frames(mj_model, args.render_resolution)

    mjx_model = mjx.put_model(mj_model)
    print(f"\n[mjx ] model uploaded. render res: "
          f"{args.render_resolution}x{args.render_resolution}, "
          f"render_every: {args.render_every} steps")

    # Table headers.
    print("\n" + "=" * 88)
    print(f"{'batch':>6}  {'phys_wall':>10}  {'phys_sps':>12}  "
          f"{'rend_wall':>10}  {'rend_fps':>12}  "
          f"{'comb_wall':>10}  {'comb_sps':>12}")
    print("=" * 88)

    results: list[dict] = []
    for batch_size in args.batch_sizes:
        # Physics only.
        try:
            phys_wall, phys_sps = time_physics_only(
                mjx_model, batch_size, args.steps_per_measurement
            )
        except Exception as exc:
            print(f"{batch_size:>6}  [physics OOM/error: {exc.__class__.__name__}]")
            break

        # Render only (per-env host render cost).
        rend_n = max(1, args.steps_per_measurement // args.render_every)
        try:
            rend_wall, rend_fps = time_render_only(
                mj_model, batch_size, args.render_resolution, rend_n
            )
        except Exception as exc:
            print(f"{batch_size:>6}  {phys_wall:>10.3f}  {phys_sps:>12.0f}  "
                  f"[render error: {exc.__class__.__name__}]")
            break

        # Combined.
        try:
            comb_wall, comb_sps, n_rend = time_combined(
                mj_model, mjx_model, batch_size,
                args.render_resolution, args.steps_per_measurement,
                args.render_every,
            )
        except Exception as exc:
            print(f"{batch_size:>6}  {phys_wall:>10.3f}  {phys_sps:>12.0f}  "
                  f"{rend_wall:>10.3f}  {rend_fps:>12.0f}  "
                  f"[combined error: {exc.__class__.__name__}]")
            break

        print(
            f"{batch_size:>6}  {phys_wall:>10.3f}  {phys_sps:>12,.0f}  "
            f"{rend_wall:>10.3f}  {rend_fps:>12,.0f}  "
            f"{comb_wall:>10.3f}  {comb_sps:>12,.0f}"
        )
        results.append({
            "batch_size": batch_size,
            "physics_wall_s": phys_wall,
            "physics_sps": phys_sps,
            "render_wall_s": rend_wall,
            "render_fps": rend_fps,
            "combined_wall_s": comb_wall,
            "combined_sps": comb_sps,
        })

    # Interpretation guidance.
    print("\n" + "=" * 88)
    print("INTERPRETATION")
    print("=" * 88)
    if len(results) < 2:
        print("  Not enough data points for scaling analysis.")
    else:
        base = results[0]
        base_wall_per_step = base["physics_wall_s"] / args.steps_per_measurement
        print(f"  Physics baseline: batch={base['batch_size']}, "
              f"{base['physics_sps']:,.0f} steps/sec.")
        for r in results[1:]:
            wall_per_step = r["physics_wall_s"] / args.steps_per_measurement
            ratio = wall_per_step / base_wall_per_step
            verdict = "GOOD (vmap working)" if ratio < 3.0 else "BAD (check batching)"
            print(f"  batch={r['batch_size']:>5}: wallclock/step is "
                  f"{ratio:.1f}x baseline  -> {verdict}")

    if results:
        print()
        print("  THE NUMBER (for docs/measurements.md):")
        best = max(results, key=lambda r: r["combined_sps"])
        print(f"    Best combined throughput: {best['combined_sps']:,.0f} "
              f"steps/sec  @ batch={best['batch_size']}, "
              f"{args.render_resolution}x{args.render_resolution}, "
              f"render_every={args.render_every}, both cameras.")

    print()


if __name__ == "__main__":
    main(parse_args())