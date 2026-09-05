"""Fixed-shape, per-lane domain randomization for the SO-101 scene.

The configuration is static Python data.  The sampled :class:`EnvParams` is a
JAX pytree carried with each world, so a lane's key recreates exactly the same
world when it is reset.  Texture pixels stay static in the MuJoCo model; only
the table geometry's material ID is batched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


NOMINAL_CUBE_MASS = 0.0345
NOMINAL_CUBE_INERTIA = 9.2e-6
MAX_ACTION_DELAY_STEPS = 5
MAX_IMAGE_DELAY_STEPS = 2


@dataclass(frozen=True)
class RandomizationConfig:
    """All axes are explicit, bounded, and static for a compiled environment."""

    name: str
    cube_mass_enabled: bool = False
    cube_spawn_enabled: bool = False
    overhead_camera_enabled: bool = False
    wrist_camera_enabled: bool = False
    light_enabled: bool = False
    material_color_enabled: bool = False
    table_material_enabled: bool = False
    cube_friction_enabled: bool = False
    arm_damping_enabled: bool = False
    actuator_force_enabled: bool = False
    cube_contact_enabled: bool = False
    action_delay_enabled: bool = False
    image_delay_enabled: bool = False

    cube_mass_range: tuple[float, float] = (0.0276, 0.0414)
    cube_spawn_x_range: tuple[float, float] = (0.16, 0.28)
    cube_spawn_y_range: tuple[float, float] = (-0.10, 0.02)
    overhead_position_radius: float = 0.010
    wrist_position_radius: float = 0.005
    overhead_angle_degrees: float = 2.0
    wrist_angle_degrees: float = 2.0
    overhead_fovy_range: tuple[float, float] = (41.0, 45.0)
    wrist_fovy_range: tuple[float, float] = (78.0, 84.0)
    light_position_radius: float = 0.10
    light_rgb_scale_range: tuple[float, float] = (0.75, 1.25)
    material_brightness_range: tuple[float, float] = (0.85, 1.15)
    friction_scale_range: tuple[float, float] = (0.8, 1.2)
    damping_scale_range: tuple[float, float] = (0.8, 1.2)
    force_scale_range: tuple[float, float] = (0.8, 1.2)
    solref_time_scale_range: tuple[float, float] = (0.8, 1.2)
    solimp_delta: float = 0.03
    action_delay_range: tuple[int, int] = (0, MAX_ACTION_DELAY_STEPS)
    image_delay_range: tuple[int, int] = (0, MAX_IMAGE_DELAY_STEPS)


class EnvParams(NamedTuple):
    """All per-world sampled values.  Every field has a leading lane axis."""

    world_key: jax.Array
    cube_mass: jax.Array
    cube_inertia: jax.Array
    cube_position: jax.Array
    overhead_camera_position: jax.Array
    overhead_camera_quaternion: jax.Array
    overhead_camera_fovy: jax.Array
    wrist_camera_position: jax.Array
    wrist_camera_quaternion: jax.Array
    wrist_camera_fovy: jax.Array
    light_position: jax.Array
    light_rgb_scale: jax.Array
    material_brightness: jax.Array
    table_material_index: jax.Array
    cube_friction_scale: jax.Array
    arm_damping_scale: jax.Array
    actuator_force_scale: jax.Array
    cube_solref_time_scale: jax.Array
    cube_solimp_delta: jax.Array
    action_delay_steps: jax.Array
    image_delay_steps: jax.Array


@dataclass(frozen=True)
class RandomizationIds:
    """Resolved static scene IDs and nominal leaves used for replacement."""

    cube_body: int
    cube_geom: int
    cube_qpos_adr: int
    table_geom: int
    tub_geom_ids: tuple[int, ...]
    table_material_ids: tuple[int, ...]
    tub_material_id: int
    cube_material_id: int
    wrist_camera: int
    overhead_camera: int
    light: int
    arm_dof_ids: tuple[int, ...]
    actuator_ids: tuple[int, ...]


def _uniform(key, bounds: tuple[float, float]) -> jax.Array:
    return jax.random.uniform(key, (), minval=bounds[0], maxval=bounds[1])


def _unit_vector(key) -> jax.Array:
    raw = jax.random.normal(key, (3,), dtype=jnp.float32)
    return raw / jnp.maximum(jnp.linalg.norm(raw), 1e-6)


def _quat_multiply(left: jax.Array, right: jax.Array) -> jax.Array:
    """MuJoCo's scalar-first (w, x, y, z) quaternion product."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return jnp.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=jnp.float32,
    )


def _orientation_perturbation(key, base_quaternion, max_angle_degrees: float) -> jax.Array:
    axis_key, angle_key = jax.random.split(key)
    axis = _unit_vector(axis_key)
    max_angle = jnp.deg2rad(jnp.asarray(max_angle_degrees, dtype=jnp.float32))
    angle = jax.random.uniform(angle_key, (), minval=-max_angle, maxval=max_angle)
    delta = jnp.concatenate(
        (jnp.cos(angle / 2.0)[None], axis * jnp.sin(angle / 2.0))
    )
    return _quat_multiply(delta, base_quaternion)


def _maybe(enabled: bool, sampled: jax.Array, nominal: jax.Array) -> jax.Array:
    return sampled if enabled else nominal


def sample_params(
    world_keys: jax.Array,
    config: RandomizationConfig,
    base_model: Any,
    ids: RandomizationIds,
) -> EnvParams:
    """Sample deterministic lane parameters from fixed reset keys.

    This is intentionally vectorized over *only* `world_keys`; the model and
    resolved IDs are static environment configuration.
    """

    base_cube_pos = jnp.array([0.22, 0.0, 0.021], dtype=jnp.float32)
    base_overhead_pos = base_model.cam_pos[ids.overhead_camera]
    base_wrist_pos = base_model.cam_pos[ids.wrist_camera]
    base_overhead_quat = base_model.cam_quat[ids.overhead_camera]
    base_wrist_quat = base_model.cam_quat[ids.wrist_camera]
    base_light_pos = base_model.light_pos[ids.light]

    def one(key):
        keys = jax.random.split(key, 16)
        mass = _maybe(config.cube_mass_enabled, _uniform(keys[0], config.cube_mass_range), jnp.asarray(NOMINAL_CUBE_MASS))
        mass_scale = mass / NOMINAL_CUBE_MASS
        spawn = _maybe(
            config.cube_spawn_enabled,
            jnp.array([_uniform(keys[1], config.cube_spawn_x_range), _uniform(keys[2], config.cube_spawn_y_range), 0.021], dtype=jnp.float32),
            base_cube_pos,
        )
        overhead_pos = _maybe(config.overhead_camera_enabled, base_overhead_pos + jax.random.uniform(keys[3], (3,), minval=-config.overhead_position_radius, maxval=config.overhead_position_radius), base_overhead_pos)
        wrist_pos = _maybe(config.wrist_camera_enabled, base_wrist_pos + jax.random.uniform(keys[4], (3,), minval=-config.wrist_position_radius, maxval=config.wrist_position_radius), base_wrist_pos)
        overhead_quat = _maybe(config.overhead_camera_enabled, _orientation_perturbation(keys[5], base_overhead_quat, config.overhead_angle_degrees), base_overhead_quat)
        wrist_quat = _maybe(config.wrist_camera_enabled, _orientation_perturbation(keys[6], base_wrist_quat, config.wrist_angle_degrees), base_wrist_quat)
        overhead_fovy = _maybe(config.overhead_camera_enabled, _uniform(keys[7], config.overhead_fovy_range), base_model.cam_fovy[ids.overhead_camera])
        wrist_fovy = _maybe(config.wrist_camera_enabled, _uniform(keys[8], config.wrist_fovy_range), base_model.cam_fovy[ids.wrist_camera])
        light_position = _maybe(config.light_enabled, base_light_pos + jax.random.uniform(keys[9], (3,), minval=-config.light_position_radius, maxval=config.light_position_radius), base_light_pos)
        light_scale = _maybe(config.light_enabled, _uniform(keys[10], config.light_rgb_scale_range), jnp.asarray(1.0))
        brightness = _maybe(config.material_color_enabled, _uniform(keys[11], config.material_brightness_range), jnp.asarray(1.0))
        table_material = jnp.where(
            config.table_material_enabled,
            jax.random.randint(keys[12], (), 0, len(ids.table_material_ids), dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        friction = _maybe(config.cube_friction_enabled, _uniform(keys[13], config.friction_scale_range), jnp.asarray(1.0))
        damping = _maybe(config.arm_damping_enabled, _uniform(keys[14], config.damping_scale_range), jnp.asarray(1.0))
        force = _maybe(config.actuator_force_enabled, _uniform(keys[15], config.force_scale_range), jnp.asarray(1.0))
        # Disabled physics axes retain their nominal values.  Separate folds
        # keep their future activation independent of the fields above.
        solref = _maybe(config.cube_contact_enabled, _uniform(jax.random.fold_in(key, 16), config.solref_time_scale_range), jnp.asarray(1.0))
        solimp = _maybe(config.cube_contact_enabled, jax.random.uniform(jax.random.fold_in(key, 17), (), minval=-config.solimp_delta, maxval=config.solimp_delta), jnp.asarray(0.0))
        action_delay = jnp.where(config.action_delay_enabled, jax.random.randint(jax.random.fold_in(key, 18), (), config.action_delay_range[0], config.action_delay_range[1] + 1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
        image_delay = jnp.where(config.image_delay_enabled, jax.random.randint(jax.random.fold_in(key, 19), (), config.image_delay_range[0], config.image_delay_range[1] + 1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
        return (
            mass,
            jnp.full((3,), NOMINAL_CUBE_INERTIA, dtype=jnp.float32) * mass_scale,
            spawn,
            overhead_pos,
            overhead_quat,
            overhead_fovy,
            wrist_pos,
            wrist_quat,
            wrist_fovy,
            light_position,
            light_scale,
            brightness,
            table_material,
            friction,
            damping,
            force,
            solref,
            solimp,
            action_delay,
            image_delay,
        )

    values = jax.vmap(one)(world_keys)
    return EnvParams(world_keys, *values)


def apply_params_to_model(base_model: Any, params: EnvParams, ids: RandomizationIds) -> tuple[Any, Any]:
    """Build the selected batched MJX leaves and matching `jax.vmap` axes.

    A full model is never stored in the carry.  This inexpensive pure
    reconstruction starts from the one static model and batches only leaves
    whose per-lane values can affect physics or Warp rendering.
    """

    batch_size = params.cube_mass.shape[0]
    repeat = lambda leaf: jnp.broadcast_to(leaf, (batch_size,) + leaf.shape)
    replacements: dict[str, jax.Array] = {}

    body_mass = repeat(base_model.body_mass).at[:, ids.cube_body].set(params.cube_mass)
    body_inertia = repeat(base_model.body_inertia).at[:, ids.cube_body, :].set(params.cube_inertia)
    replacements["body_mass"] = body_mass
    replacements["body_inertia"] = body_inertia

    cam_pos = repeat(base_model.cam_pos)
    cam_pos = cam_pos.at[:, ids.overhead_camera, :].set(params.overhead_camera_position)
    cam_pos = cam_pos.at[:, ids.wrist_camera, :].set(params.wrist_camera_position)
    replacements["cam_pos"] = cam_pos
    cam_quat = repeat(base_model.cam_quat)
    cam_quat = cam_quat.at[:, ids.overhead_camera, :].set(params.overhead_camera_quaternion)
    cam_quat = cam_quat.at[:, ids.wrist_camera, :].set(params.wrist_camera_quaternion)
    replacements["cam_quat"] = cam_quat
    cam_fovy = repeat(base_model.cam_fovy)
    cam_fovy = cam_fovy.at[:, ids.overhead_camera].set(params.overhead_camera_fovy)
    cam_fovy = cam_fovy.at[:, ids.wrist_camera].set(params.wrist_camera_fovy)
    replacements["cam_fovy"] = cam_fovy

    light_pos = repeat(base_model.light_pos).at[:, ids.light, :].set(params.light_position)
    replacements["light_pos"] = light_pos
    light_diffuse = repeat(base_model.light_diffuse) * params.light_rgb_scale[:, None, None]
    light_specular = repeat(base_model.light_specular) * params.light_rgb_scale[:, None, None]
    replacements["light_diffuse"] = light_diffuse
    replacements["light_specular"] = light_specular

    mat_rgba = repeat(base_model.mat_rgba)
    material_ids = tuple(dict.fromkeys((*ids.table_material_ids, ids.tub_material_id, ids.cube_material_id)))
    mat_rgba = mat_rgba.at[:, material_ids, :3].set(
        mat_rgba[:, material_ids, :3] * params.material_brightness[:, None, None]
    )
    replacements["mat_rgba"] = mat_rgba
    geom_matid = repeat(base_model.geom_matid)
    table_choices = jnp.asarray(ids.table_material_ids, dtype=jnp.int32)
    geom_matid = geom_matid.at[:, ids.table_geom].set(table_choices[params.table_material_index])
    replacements["geom_matid"] = geom_matid

    geom_friction = repeat(base_model.geom_friction).at[:, ids.cube_geom, :].set(base_model.geom_friction[ids.cube_geom] * params.cube_friction_scale[:, None])
    replacements["geom_friction"] = geom_friction
    dof_damping = repeat(base_model.dof_damping).at[:, ids.arm_dof_ids].set(base_model.dof_damping[jnp.asarray(ids.arm_dof_ids)] * params.arm_damping_scale[:, None])
    replacements["dof_damping"] = dof_damping
    force_range = repeat(base_model.actuator_forcerange)
    force_range = force_range.at[:, ids.actuator_ids, :].set(base_model.actuator_forcerange[jnp.asarray(ids.actuator_ids)] * params.actuator_force_scale[:, None, None])
    replacements["actuator_forcerange"] = force_range
    solref = repeat(base_model.geom_solref)
    solref = solref.at[:, ids.cube_geom, 0].set(base_model.geom_solref[ids.cube_geom, 0] * params.cube_solref_time_scale)
    replacements["geom_solref"] = solref
    solimp = repeat(base_model.geom_solimp)
    adjusted_solimp = jnp.clip(base_model.geom_solimp[ids.cube_geom, :2] + params.cube_solimp_delta[:, None], 1e-6, 0.999)
    solimp = solimp.at[:, ids.cube_geom, :2].set(adjusted_solimp)
    replacements["geom_solimp"] = solimp

    model = base_model.tree_replace(replacements)
    in_axes = jax.tree.map(lambda _leaf: None, base_model).tree_replace(
        {name: 0 for name in replacements}
    )
    return model, in_axes


def set_cube_reset_pose(data: Any, params: EnvParams, ids: RandomizationIds) -> Any:
    """Write XY spawn with fixed upright quaternion into batched freejoint qpos."""
    adr = ids.cube_qpos_adr
    qpos = data.qpos
    qpos = qpos.at[:, adr : adr + 3].set(params.cube_position)
    qpos = qpos.at[:, adr + 3 : adr + 7].set(
        jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    )
    return data.replace(qpos=qpos)


def parameter_summary(params: EnvParams) -> dict[str, jax.Array]:
    """Small device-friendly min/max summary for the randomization smoke test."""
    return {
        "cube_mass_kg": jnp.array([params.cube_mass.min(), params.cube_mass.max()]),
        "cube_x_m": jnp.array([params.cube_position[:, 0].min(), params.cube_position[:, 0].max()]),
        "cube_y_m": jnp.array([params.cube_position[:, 1].min(), params.cube_position[:, 1].max()]),
        "overhead_fovy_deg": jnp.array([params.overhead_camera_fovy.min(), params.overhead_camera_fovy.max()]),
        "wrist_fovy_deg": jnp.array([params.wrist_camera_fovy.min(), params.wrist_camera_fovy.max()]),
        "action_delay_steps": jnp.array([params.action_delay_steps.min(), params.action_delay_steps.max()]),
        "image_delay_steps": jnp.array([params.image_delay_steps.min(), params.image_delay_steps.max()]),
    }
