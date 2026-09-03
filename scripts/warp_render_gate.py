"""Validate the MJX Warp physics-and-rendering path for Step 5.

This is deliberately a gate, not an environment.  It proves that the current
scene can use Warp model/data to render both policy cameras under JIT and at a
batched size, while retaining the existing cube-settling numerical check.
"""

from __future__ import annotations

import argparse
import functools
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from PIL import Image


SCENE_PATH = Path("sim/scene_cube_tub.xml")
SAMPLES_DIR = Path("scripts/warp_render_samples")
CAMERA_NAMES = ("wrist", "overhead")
FULL_MESH_MIN_GPU_MEMORY_GIB = 16.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--settle-steps", type=int, default=2_000)
    return parser.parse_args()


def gpu_memory_mib() -> tuple[float | None, float | None]:
    """Return (total, used) memory for GPU 0 without requiring pynvml."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        total, used = result.stdout.splitlines()[0].split(",")
        return float(total), float(used)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        return None, None


def configure_arm_mesh_collisions(mj_model: mujoco.MjModel) -> None:
    """Match the 8 GiB collision policy used by the existing smoke scripts."""
    total_mib, _ = gpu_memory_mib()
    enabled = total_mib is not None and total_mib / 1024 >= FULL_MESH_MIN_GPU_MEMORY_GIB
    mask = mj_model.geom_group == 3
    mj_model.geom_contype[mask] = int(enabled)
    mj_model.geom_conaffinity[mask] = int(enabled)
    print(f"[config] arm mesh collisions: {'enabled' if enabled else 'disabled'}")


def camera_ids(mj_model: mujoco.MjModel) -> tuple[int, int]:
    ids = tuple(
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        for name in CAMERA_NAMES
    )
    if any(camera_id < 0 for camera_id in ids):
        raise RuntimeError(f"Missing expected cameras: {CAMERA_NAMES}")
    return ids


def cube_qpos_slice(mj_model: mujoco.MjModel) -> slice:
    joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    if joint_id < 0:
        raise RuntimeError("cube_free joint not found")
    start = int(mj_model.jnt_qposadr[joint_id])
    return slice(start, start + 7)


def make_batched_data(mj_model: mujoco.MjModel, batch_size: int):
    keys = jax.random.split(jax.random.key(0), batch_size)

    def make_data(_key):
        del _key
        return mjx.make_data(mj_model, impl="warp")

    return jax.vmap(make_data)(keys)


def main(args: argparse.Namespace) -> None:
    print(f"[jax ] devices: {jax.devices()}")
    print(f"[jax ] backend: {jax.default_backend()}")
    total_mib, used_mib = gpu_memory_mib()
    print(f"[gpu ] memory before: used={used_mib} MiB / total={total_mib} MiB")

    mj_model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    configure_arm_mesh_collisions(mj_model)
    wrist_id, overhead_id = camera_ids(mj_model)
    cube_slice = cube_qpos_slice(mj_model)
    print(f"[load] cameras: wrist={wrist_id}, overhead={overhead_id}")

    mjx_model = mjx.put_model(mj_model, impl="warp")
    render_context = mjx.create_render_context(
        mjm=mj_model,
        nworld=args.batch_size,
        cam_res=(args.resolution, args.resolution),
        render_rgb=[True] * mj_model.ncam,
        render_depth=False,
        render_seg=False,
    )
    context_pytree = render_context.pytree()
    data = make_batched_data(mj_model, args.batch_size)

    @jax.jit
    def forward_refit_render(batch_data):
        batch_data = jax.vmap(mjx.forward, in_axes=(None, 0))(mjx_model, batch_data)
        batch_data = jax.vmap(mjx.refit_bvh, in_axes=(None, 0, None))(
            mjx_model, batch_data, context_pytree
        )
        packed_rgb, _packed_depth = mjx.render(mjx_model, batch_data, context_pytree)
        wrist_rgb = jnp.clip(
            mjx.get_rgb(context_pytree, wrist_id, packed_rgb) * 255.0, 0, 255
        ).astype(jnp.uint8)
        overhead_rgb = jnp.clip(
            mjx.get_rgb(context_pytree, overhead_id, packed_rgb) * 255.0, 0, 255
        ).astype(jnp.uint8)
        return batch_data, wrist_rgb, overhead_rgb

    started = time.perf_counter()
    data, wrist_rgb, overhead_rgb = forward_refit_render(data)
    jax.block_until_ready(wrist_rgb)
    compile_and_first_run_s = time.perf_counter() - started

    started = time.perf_counter()
    data, wrist_rgb, overhead_rgb = forward_refit_render(data)
    jax.block_until_ready(wrist_rgb)
    render_run_s = time.perf_counter() - started
    print(f"[render] compile + first run: {compile_and_first_run_s:.3f}s")
    print(
        f"[render] batch={args.batch_size}, res={args.resolution}, "
        f"run={render_run_s:.3f}s, env-renders/s={args.batch_size / render_run_s:.1f}"
    )
    print(f"[render] wrist: shape={wrist_rgb.shape}, dtype={wrist_rgb.dtype}")
    print(f"[render] overhead: shape={overhead_rgb.shape}, dtype={overhead_rgb.dtype}")

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, pixels in (("wrist", wrist_rgb), ("overhead", overhead_rgb)):
        path = SAMPLES_DIR / f"{name}_{args.resolution}x{args.resolution}.png"
        Image.fromarray(np.asarray(pixels[0])).save(path)
        print(f"[sample] saved {path}")

    one_data = mjx.make_data(mj_model, impl="warp")
    step_fn = jax.jit(functools.partial(mjx.step, mjx_model))
    started = time.perf_counter()
    for _ in range(args.settle_steps):
        one_data = step_fn(one_data)
    jax.block_until_ready(one_data.qpos)
    settle_s = time.perf_counter() - started

    cube_z = float(np.asarray(one_data.qpos[cube_slice])[2])
    finite = bool(np.isfinite(np.asarray(one_data.qpos)).all()) and bool(
        np.isfinite(np.asarray(one_data.qvel)).all()
    )
    print(
        f"[settle] {args.settle_steps} steps in {settle_s:.3f}s "
        f"({args.settle_steps / settle_s:.1f} steps/s)"
    )
    print(f"[settle] cube_z={cube_z:+.4f} m, finite={finite}")
    total_mib, used_mib = gpu_memory_mib()
    print(f"[gpu ] memory after: used={used_mib} MiB / total={total_mib} MiB")

    if wrist_rgb.dtype != jnp.uint8 or overhead_rgb.dtype != jnp.uint8:
        raise AssertionError("Warp renderer did not produce uint8 RGB")
    if wrist_rgb.shape != (args.batch_size, args.resolution, args.resolution, 3):
        raise AssertionError("unexpected wrist render shape")
    if overhead_rgb.shape != (args.batch_size, args.resolution, args.resolution, 3):
        raise AssertionError("unexpected overhead render shape")
    if not finite:
        raise AssertionError("non-finite Warp physics state")
    if not np.isclose(cube_z, 0.019, atol=0.003):
        raise AssertionError(f"cube did not settle on table: z={cube_z}")
    print("[pass] Warp physics + dual-camera rendering gate passed")


if __name__ == "__main__":
    main(parse_args())
