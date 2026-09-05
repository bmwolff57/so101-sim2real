"""Pure, simulator-truth reward and success calculations for Step 7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from configs.task_reward import TaskRewardConfig


@dataclass(frozen=True)
class RewardIds:
    cube_body: int
    tub_body: int
    gripper_site: int
    fixed_jaw_touch_sensor: int
    moving_jaw_touch_sensor: int


class RewardTerms(NamedTuple):
    reach: jnp.ndarray
    transport: jnp.ndarray
    pickup: jnp.ndarray
    success: jnp.ndarray


class RewardDiagnostics(NamedTuple):
    gripper_cube_distance: jnp.ndarray
    cube_tub_xy_distance: jnp.ndarray
    cube_linear_speed: jnp.ndarray
    cube_angular_speed: jnp.ndarray
    fixed_jaw_contact: jnp.ndarray
    moving_jaw_contact: jnp.ndarray
    secure_pickup: jnp.ndarray
    released: jnp.ndarray
    contained: jnp.ndarray
    settled: jnp.ndarray
    success: jnp.ndarray


class RewardResult(NamedTuple):
    reward: jnp.ndarray
    terms: RewardTerms
    diagnostics: RewardDiagnostics
    pickup_rewarded: jnp.ndarray
    settle_steps: jnp.ndarray


def compute_reward(data, config: TaskRewardConfig, ids: RewardIds, pickup_rewarded, settle_steps) -> RewardResult:
    """Calculate fixed-shape rewards without exposing truth to the policy."""
    cube_position = data.xpos[:, ids.cube_body]
    tub_position = data.xpos[:, ids.tub_body]
    gripper_position = data.site_xpos[:, ids.gripper_site]
    gripper_cube_distance = jnp.linalg.norm(gripper_position - cube_position, axis=-1)
    cube_tub_xy_distance = jnp.linalg.norm(cube_position[:, :2] - tub_position[:, :2], axis=-1)
    # A freejoint owns six consecutive generalized velocities: linear then angular.
    cube_linear_speed = jnp.linalg.norm(data.qvel[:, -6:-3], axis=-1)
    cube_angular_speed = jnp.linalg.norm(data.qvel[:, -3:], axis=-1)
    fixed_jaw_contact = data.sensordata[:, ids.fixed_jaw_touch_sensor] > 1e-6
    moving_jaw_contact = data.sensordata[:, ids.moving_jaw_touch_sensor] > 1e-6
    secure_pickup = (
        fixed_jaw_contact
        & moving_jaw_contact
        & (cube_position[:, 2] >= config.pickup_height_m)
    )
    new_pickup = secure_pickup & ~pickup_rewarded
    next_pickup_rewarded = pickup_rewarded | secure_pickup
    contained = (
        (jnp.abs(cube_position[:, 0] - tub_position[:, 0]) <= config.containment_half_extent_m)
        & (jnp.abs(cube_position[:, 1] - tub_position[:, 1]) <= config.containment_half_extent_m)
        & (cube_position[:, 2] <= config.max_cube_center_height_m)
    )
    released = ~(fixed_jaw_contact | moving_jaw_contact)
    settled = (
        contained
        & released
        & (cube_linear_speed <= config.max_linear_speed_mps)
        & (cube_angular_speed <= config.max_angular_speed_radps)
    )
    next_settle_steps = jnp.where(settled, settle_steps + 1, 0).astype(jnp.int32)
    # Success is the one terminal transition that reaches the dwell threshold.
    # A collector/autoreset wrapper must preserve this transition, then reset
    # before another policy action; never keep emitting the +10 bonus.
    success = next_settle_steps == config.settle_control_steps
    reach = jnp.where(
        secure_pickup,
        0.0,
        config.reach_weight * (1.0 - jnp.tanh(gripper_cube_distance / config.reach_scale_m)),
    )
    transport = jnp.where(
        secure_pickup,
        config.transport_weight * (1.0 - jnp.tanh(cube_tub_xy_distance / config.transport_scale_m)),
        0.0,
    )
    pickup = jnp.where(new_pickup, config.pickup_bonus, 0.0)
    success_term = jnp.where(success, config.success_bonus, 0.0)
    terms = RewardTerms(reach, transport, pickup, success_term)
    reward = reach + transport + pickup + success_term
    diagnostics = RewardDiagnostics(
        gripper_cube_distance, cube_tub_xy_distance, cube_linear_speed,
        cube_angular_speed, fixed_jaw_contact, moving_jaw_contact,
        secure_pickup, released, contained, settled, success,
    )
    return RewardResult(reward, terms, diagnostics, next_pickup_rewarded, next_settle_steps)
