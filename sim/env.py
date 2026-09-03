"""Mechanics-only, Warp-rendered SO-101 MJX environment.

This module deliberately stops before task design: reset is deterministic,
reward is always zero, and no success/failure condition exists yet.  Its job is
to prove the JAX carry/observation/action contract with the real scene.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco.mjx.warp import types as warp_types

from sim.proprio import assemble_proprio


SCENE_PATH = Path("sim/scene_cube_tub.xml")
CAMERA_NAMES = ("wrist", "overhead")
ARM_ACTION_SCALE = 0.025
GRIPPER_ACTION_SCALE = 0.05
SIM_DT = 0.002
CTRL_DT = 0.05
SUBSTEPS = 25
MAX_EPISODE_STEPS = 200
# Warp's contact buffer spans the full VMAP batch, unlike MuJoCo's ordinary
# per-world contact allocation.  The scene's default of 48 contacts per world
# is retained and scaled when constructing batched data.
MAX_CONTACTS_PER_WORLD = 48
MAX_CONSTRAINTS_PER_WORLD = 64


class EnvParams(NamedTuple):
    """Per-lane reset data reserved for later randomization."""

    world_key: jax.Array


class Carry(NamedTuple):
    data: Any
    params: EnvParams
    step: jax.Array
    previous_action: jax.Array
    command_target: jax.Array


class RewardTerms(NamedTuple):
    mechanics_only: jax.Array


class Diagnostics(NamedTuple):
    physics_finite: jax.Array


class StepOutput(NamedTuple):
    reward: jax.Array
    reward_terms: RewardTerms
    terminated: jax.Array
    truncated: jax.Array
    diagnostics: Diagnostics


Observation = dict[str, jax.Array]


def _disable_arm_mesh_collisions(mj_model: mujoco.MjModel) -> None:
    """Keep the established 8 GiB scene collision configuration."""
    mask = mj_model.geom_group == 3
    mj_model.geom_contype[mask] = 0
    mj_model.geom_conaffinity[mask] = 0


def _named_id(mj_model: mujoco.MjModel, obj_type, name: str) -> int:
    object_id = mujoco.mj_name2id(mj_model, obj_type, name)
    if object_id < 0:
        raise ValueError(f"{name!r} was not found in {SCENE_PATH}")
    return object_id


@dataclass(frozen=True)
class So101Env:
    """Fixed-batch immutable Warp environment configuration.

    The batch size is static because the Warp render context owns buffers for a
    fixed world count. All changing per-world state lives in `Carry`.
    """

    mj_model: Any
    model: Any
    render_context: Any
    batch_size: int
    resolution: int
    qpos_indices: jax.Array
    qvel_indices: jax.Array
    ctrl_low: jax.Array
    ctrl_high: jax.Array
    action_scale: jax.Array
    wrist_camera_id: int
    overhead_camera_id: int

    @classmethod
    def from_scene(
        cls,
        batch_size: int,
        resolution: int = 128,
        scene_path: Path | str = SCENE_PATH,
    ) -> "So101Env":
        """Construct immutable scene metadata and a Warp render context."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if resolution < 1:
            raise ValueError("resolution must be positive")

        host_model = mujoco.MjModel.from_xml_path(str(scene_path))
        _disable_arm_mesh_collisions(host_model)
        actuator_ids = np.arange(host_model.nu)
        if host_model.nu != 6:
            raise ValueError(f"expected six actuators, found {host_model.nu}")
        joint_ids = host_model.actuator_trnid[actuator_ids, 0]
        qpos_indices = host_model.jnt_qposadr[joint_ids]
        qvel_indices = host_model.jnt_dofadr[joint_ids]
        wrist_id = _named_id(host_model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAMES[0])
        overhead_id = _named_id(host_model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAMES[1])

        # ``WARP_STAGED`` uses stable staging buffers across JAX retraces.
        # That avoids the duplicate FFI/graph-cache path encountered when
        # reset and step are separate JIT executables.
        model = mjx.put_model(
            host_model, impl="warp", graph_mode=warp_types.GraphMode.WARP_STAGED
        )
        render_context = mjx.create_render_context(
            mjm=host_model,
            nworld=batch_size,
            cam_res=(resolution, resolution),
            render_rgb=[True] * host_model.ncam,
            render_depth=False,
            render_seg=False,
        )
        return cls(
            mj_model=host_model,
            model=model,
            render_context=render_context,
            batch_size=batch_size,
            resolution=resolution,
            qpos_indices=jnp.asarray(qpos_indices),
            qvel_indices=jnp.asarray(qvel_indices),
            ctrl_low=jnp.asarray(host_model.actuator_ctrlrange[:, 0]),
            ctrl_high=jnp.asarray(host_model.actuator_ctrlrange[:, 1]),
            action_scale=jnp.asarray(
                [ARM_ACTION_SCALE] * 5 + [GRIPPER_ACTION_SCALE], dtype=jnp.float32
            ),
            wrist_camera_id=wrist_id,
            overhead_camera_id=overhead_id,
        )

    def _make_data(self, world_keys: jax.Array):
        # JAX 0.11's typed PRNG keys have scalar shape.  Splitting one key for
        # each lane therefore produces shape ``(batch_size,)`` rather than the
        # legacy uint32 key representation ``(batch_size, 2)``.  Only the
        # leading lane dimension is part of this environment's contract.
        if world_keys.ndim < 1 or world_keys.shape[0] != self.batch_size:
            raise ValueError(
                "world_keys must have leading shape "
                f"({self.batch_size},), got {world_keys.shape}"
            )

        def make_data(_key):
            del _key
            return mjx.make_data(
                self.mj_model,
                impl="warp",
                naconmax=MAX_CONTACTS_PER_WORLD * self.batch_size,
                njmax=MAX_CONSTRAINTS_PER_WORLD,
            )

        return jax.vmap(make_data)(world_keys)

    def _render_observation(self, data, previous_action, command_target) -> Observation:
        context = self.render_context.pytree()
        data = jax.vmap(mjx.forward, in_axes=(None, 0))(self.model, data)
        data = jax.vmap(mjx.refit_bvh, in_axes=(None, 0, None))(
            self.model, data, context
        )
        packed_rgb, _ = mjx.render(self.model, data, context)

        def rgb(camera_id: int) -> jax.Array:
            return jnp.clip(
                mjx.get_rgb(context, camera_id, packed_rgb) * 255.0, 0, 255
            ).astype(jnp.uint8)

        proprio = assemble_proprio(
            data.qpos[:, self.qpos_indices],
            data.qvel[:, self.qvel_indices],
            previous_action,
            command_target,
        )
        return data, {
            "wrist": rgb(self.wrist_camera_id),
            "overhead": rgb(self.overhead_camera_id),
            "proprio": proprio,
        }

    def reset(self, world_keys: jax.Array) -> tuple[Carry, Observation]:
        """Create deterministic, batched reset state for the provided keys."""
        data = self._make_data(world_keys)
        target = jnp.clip(data.qpos[:, self.qpos_indices], self.ctrl_low, self.ctrl_high)
        data = data.replace(ctrl=target)
        previous_action = jnp.zeros((self.batch_size, 6), dtype=jnp.float32)
        data, observation = self._render_observation(data, previous_action, target)
        carry = Carry(
            data=data,
            params=EnvParams(world_key=world_keys),
            step=jnp.zeros((self.batch_size,), dtype=jnp.int32),
            previous_action=previous_action,
            command_target=target,
        )
        return carry, observation

    def step(self, carry: Carry, action: jax.Array) -> tuple[Carry, Observation, StepOutput]:
        """Apply one 20 Hz control action and return mechanics-only outputs."""
        action = jnp.clip(action.astype(jnp.float32), -1.0, 1.0)
        target = jnp.clip(
            carry.command_target + action * self.action_scale,
            self.ctrl_low,
            self.ctrl_high,
        )
        data = carry.data.replace(ctrl=target)

        def one_substep(_, current_data):
            return jax.vmap(mjx.step, in_axes=(None, 0))(self.model, current_data)

        data = jax.lax.fori_loop(0, SUBSTEPS, one_substep, data)
        next_step = carry.step + 1
        data, observation = self._render_observation(data, action, target)
        finite = jnp.logical_and(
            jnp.all(jnp.isfinite(data.qpos), axis=-1),
            jnp.all(jnp.isfinite(data.qvel), axis=-1),
        )
        output = StepOutput(
            reward=jnp.zeros((self.batch_size,), dtype=jnp.float32),
            reward_terms=RewardTerms(
                mechanics_only=jnp.zeros((self.batch_size,), dtype=jnp.float32)
            ),
            terminated=jnp.zeros((self.batch_size,), dtype=jnp.bool_),
            truncated=next_step >= MAX_EPISODE_STEPS,
            diagnostics=Diagnostics(physics_finite=finite),
        )
        return (
            Carry(
                data=data,
                params=carry.params,
                step=next_step,
                previous_action=action,
                command_target=target,
            ),
            observation,
            output,
        )


def raise_if_nonfinite(output: StepOutput) -> None:
    """Host-side fail-fast boundary; never admit corrupted lanes to a learner."""
    finite = np.asarray(output.diagnostics.physics_finite)
    if not finite.all():
        failed_lanes = np.flatnonzero(~finite).tolist()
        raise FloatingPointError(f"non-finite physics state in lanes {failed_lanes}")
