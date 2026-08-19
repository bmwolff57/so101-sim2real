"""
MJX smoke test for sim/scene_cube_tub.xml.

Purpose: confirm the scene loads under MJX, the JIT compiles, and
physics steps behave sanely. Prints timing, energy, and state to
stdout — no rendering.

Run from project root (with the mjx venv active):
    python scripts/mjx_smoke_test.py
    python scripts/mjx_smoke_test.py --arm-mesh-collisions

What to look for:
  - Load, upload, and JIT times printed separately.
  - Steps/sec baseline for a single env (compare later against
    vmapped batch throughput in Step 3).
  - Cube z position falling, then stabilizing near the table top.
  - Cube translational KE and gravitational PE settling, not growing.
    If they grow with no actuator input, the integrator is unstable.
  - ncon (active contacts) going 0 -> nonzero when the cube lands.
  - Final sanity checks pass: no NaN, cube not through the table.

Requires:
  - sim/scene_cube_tub.xml exists (with a body named "cube" and
    freejoint named "cube_free")
"""
import argparse
import subprocess
import time

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

SCENE_PATH = "sim/scene_cube_tub.xml"
N_STEPS = 2000  # ~4 sec of sim at dt=0.002
LOG_EVERY = 200
FULL_MESH_MIN_GPU_MEMORY_GIB = 16.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm-mesh-collisions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override automatic arm mesh collision selection. By default, "
            "they are enabled only on GPUs with at least "
            f"{FULL_MESH_MIN_GPU_MEMORY_GIB:g} GiB of VRAM."
        ),
    )
    return parser.parse_args()


def gpu_total_memory_gib(device_index: int = 0) -> float | None:
    """Return total VRAM for the selected NVIDIA GPU, if nvidia-smi is available."""
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
    """Use an explicit flag when supplied; otherwise choose from total VRAM."""
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
        f"(automatic threshold: {FULL_MESH_MIN_GPU_MEMORY_GIB:g} GiB)"
    )
    return enabled


def configure_arm_mesh_collisions(mj_model: mujoco.MjModel, enabled: bool) -> None:
    """Toggle the vendor arm's collision group before uploading to MJX."""
    collision_geoms = mj_model.geom_group == 3
    count = int(np.count_nonzero(collision_geoms))
    if count == 0:
        raise RuntimeError("No arm collision geoms found in group 3.")

    # The vendor's collision class normally uses MuJoCo's default masks (1, 1).
    # The XML disables them for the local 8 GB GPU; restore those masks only
    # when the explicit large-GPU option is requested.
    mask_value = 1 if enabled else 0
    mj_model.geom_contype[collision_geoms] = mask_value
    mj_model.geom_conaffinity[collision_geoms] = mask_value
    state = "enabled" if enabled else "disabled"
    print(f"[config] applied arm mesh collision state: {state} ({count} geoms in group 3)")


def summarize_model(mj_model: mujoco.MjModel) -> None:
    """Print static model info once at startup."""
    print("=" * 60)
    print(f"MODEL: {SCENE_PATH}")
    print("=" * 60)
    print(f"  nq (position dim)     : {mj_model.nq}")
    print(f"  nv (velocity dim)     : {mj_model.nv}")
    print(f"  nu (actuator dim)     : {mj_model.nu}")
    print(f"  nbody                 : {mj_model.nbody}")
    print(f"  ngeom                 : {mj_model.ngeom}")
    print(f"  timestep              : {mj_model.opt.timestep}")
    print(f"  integrator            : {mj_model.opt.integrator}")
    print(f"  gravity               : {mj_model.opt.gravity}")

    print("\n  Bodies:")
    for i in range(mj_model.nbody):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
        mass = mj_model.body_mass[i]
        print(f"    [{i:2d}] {name:<30} mass={mass:.4f} kg")

    print("\n  Actuators:")
    for i in range(mj_model.nu):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        lo, hi = mj_model.actuator_ctrlrange[i]
        print(f"    [{i}] {name:<20} ctrlrange=[{lo:+.3f}, {hi:+.3f}]")
    print()


def find_cube_indices(mj_model: mujoco.MjModel) -> tuple[slice, int]:
    """Locate the cube's freejoint in qpos and its dof start index in qvel."""
    jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    if jid < 0:
        raise RuntimeError(
            "Joint 'cube_free' not found. Check scene XML for <freejoint name='cube_free'/>."
        )
    qpos_start = mj_model.jnt_qposadr[jid]
    # Freejoint occupies 7 slots in qpos: [x, y, z, qw, qx, qy, qz]
    qpos_slice = slice(qpos_start, qpos_start + 7)
    # And 6 slots in qvel: [vx, vy, vz, wx, wy, wz]
    qvel_start = int(mj_model.jnt_dofadr[jid])
    return qpos_slice, qvel_start


def main(args: argparse.Namespace) -> None:
    # ---- Load MJCF ----
    t0 = time.perf_counter()
    mj_model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    print(f"[load] MJCF parsed in {time.perf_counter() - t0:.3f}s")

    arm_mesh_collisions = select_arm_mesh_collisions(args.arm_mesh_collisions)
    configure_arm_mesh_collisions(mj_model, arm_mesh_collisions)

    summarize_model(mj_model)

    cube_qpos, cube_qvel_start = find_cube_indices(mj_model)
    cube_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_mass = float(mj_model.body_mass[cube_body_id])
    gravity_z = float(mj_model.opt.gravity[2])
    print(f"[info] cube qpos slice: {cube_qpos}")
    print(f"[info] cube qvel start: {cube_qvel_start}\n")

    # ---- Convert to MJX ----
    t0 = time.perf_counter()
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.make_data(mjx_model)
    print(f"[mjx ] model + data uploaded in {time.perf_counter() - t0:.3f}s")

    # ---- JIT compile step ----
    step_fn = jax.jit(mjx.step)

    t0 = time.perf_counter()
    mjx_data = step_fn(mjx_model, mjx_data)  # trace + compile
    mjx_data.qpos.block_until_ready()  # force actual completion
    compile_time = time.perf_counter() - t0
    print(f"[jit ] first step (trace + compile): {compile_time:.3f}s\n")

    # ---- Warm run (real per-step cost, no compile overhead) ----
    t0 = time.perf_counter()
    for _ in range(50):
        mjx_data = step_fn(mjx_model, mjx_data)
    mjx_data.qpos.block_until_ready()
    warm_time = time.perf_counter() - t0
    print(
        f"[warm] 50 steps in {warm_time * 1000:.1f}ms "
        f"({50 / warm_time:.0f} steps/sec, single env)\n"
    )

    # ---- Main loop with periodic logging ----
    print("=" * 60)
    print(
        f"STEPPING {N_STEPS} steps  (dt={mj_model.opt.timestep}, "
        f"sim time = {N_STEPS * mj_model.opt.timestep:.2f}s)"
    )
    print("=" * 60)
    print(
        f"{'step':>6}  {'sim_t':>7}  "
        f"{'cube_z':>8}  {'|cube_v|':>9}  "
        f"{'cube_KE':>10}  {'cube_PE':>10}  {'ncon':>5}"
    )

    t0 = time.perf_counter()
    for i in range(1, N_STEPS + 1):
        mjx_data = step_fn(mjx_model, mjx_data)

        if i % LOG_EVERY == 0:
            # Pull values off device. Cheap for a small state; do it
            # sparingly in real training loops.
            qpos = np.asarray(mjx_data.qpos)
            qvel = np.asarray(mjx_data.qvel)

            cube_z = qpos[cube_qpos][2]
            cube_v = np.linalg.norm(qvel[cube_qvel_start : cube_qvel_start + 3])

            # MJX Data does not expose MuJoCo's optional energy field.
            # Compute the cube's translational KE and gravitational PE
            # directly from its state instead.
            ke = 0.5 * cube_mass * cube_v**2
            pe = -cube_mass * gravity_z * cube_z
            ncon = int(mjx_data._impl.ncon)
            sim_t = float(mjx_data.time)

            print(
                f"{i:>6}  {sim_t:>7.3f}  "
                f"{cube_z:>8.4f}  {cube_v:>9.4f}  "
                f"{ke:>10.4e}  {pe:>10.4e}  {ncon:>5d}"
            )

    mjx_data.qpos.block_until_ready()
    total = time.perf_counter() - t0
    print(
        f"\n[done] {N_STEPS} steps in {total:.3f}s "
        f"({N_STEPS / total:.0f} steps/sec, single env)"
    )
    rtf = (N_STEPS * mj_model.opt.timestep) / total
    print(f"[done] realtime factor: {rtf:.1f}x")

    # ---- Sanity checks ----
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)
    final_qpos = np.asarray(mjx_data.qpos)
    final_qvel = np.asarray(mjx_data.qvel)

    cube_z_final = final_qpos[cube_qpos][2]
    cube_v_final = np.abs(final_qvel[cube_qvel_start : cube_qvel_start + 3]).max()
    print(f"  cube final z            : {cube_z_final:+.4f} m")
    print(f"  cube max |v| final      : {cube_v_final:.4f} m/s")

    passed = True
    if cube_z_final < -0.05:
        print("  ✗ cube fell through the table")
        passed = False
    elif cube_v_final > 0.01:
        print("  ⚠ cube still moving — may need more settle time or check contact")
    else:
        print("  ✓ cube settled on table")

    if np.any(np.isnan(final_qpos)) or np.any(np.isnan(final_qvel)):
        print("  ✗ NaN in state — physics blew up")
        passed = False
    else:
        print("  ✓ no NaN in state")

    print()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    args = parse_args()
    print(f"[jax ] devices: {jax.devices()}")
    print(f"[jax ] default backend: {jax.default_backend()}\n")
    main(args)
