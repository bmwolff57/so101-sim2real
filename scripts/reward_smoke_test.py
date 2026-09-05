"""Fixture, collision, random, and fixed-seed IK reward validation.

The host IK here is a validation-only oracle.  It emits ordinary normalized
joint-delta actions into the JIT MJX environment and is never a learner input.
"""

from __future__ import annotations

import mujoco
import jax
import jax.numpy as jnp
import numpy as np

from configs.domain_randomization import NO_RANDOMIZATION
from sim.env import So101Env


RANDOM_BATCH_SIZE = 64
RANDOM_HORIZON = 200


def pad_midpoint_ik(model: mujoco.MjModel, target: np.ndarray) -> np.ndarray:
    """Damped least-squares IK for the midpoint between the two jaw pads."""
    data = mujoco.MjData(model)
    fixed = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_pad")
    moving = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_pad")
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1745])
    lower, upper = model.jnt_range[:5, 0], model.jnt_range[:5, 1]

    def midpoint(candidate: np.ndarray) -> np.ndarray:
        data.qpos[:6] = candidate
        mujoco.mj_forward(model, data)
        return (data.geom_xpos[fixed] + data.geom_xpos[moving]) / 2.0

    for _ in range(200):
        point = midpoint(q)
        error = target - point
        jacobian = np.empty((3, 5))
        epsilon = 1e-5
        for joint in range(5):
            perturbed = q.copy()
            perturbed[joint] += epsilon
            jacobian[:, joint] = (midpoint(perturbed) - point) / epsilon
        q[:5] = np.clip(
            q[:5] + 0.5 * jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 1e-4 * np.eye(3), error
            ),
            lower,
            upper,
        )
    return q[:5]


def follow_target(env, step, carry, target, controls: int):
    total = 0.0
    output = None
    for _ in range(controls):
        action = jnp.clip((target[None] - carry.command_target) / env.action_scale, -1.0, 1.0)
        carry, _observation, output = step(carry, action)
        total += float(output.reward[0])
        if bool(output.terminated[0]):
            break
    return carry, output, total


def random_action_comparison() -> None:
    """Reject accidental task completion under one fixed batch of random actions."""
    env = So101Env.from_scene(RANDOM_BATCH_SIZE, randomization=NO_RANDOMIZATION)
    world_keys = jax.random.split(jax.random.key(20260906), RANDOM_BATCH_SIZE)
    actions = jax.random.uniform(
        jax.random.key(20260907),
        (RANDOM_HORIZON, RANDOM_BATCH_SIZE, 6),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )

    def rollout(keys, action_sequence):
        carry, _observation = env.reset(keys)
        initial = (
            jnp.zeros((RANDOM_BATCH_SIZE,), dtype=jnp.float32),
            jnp.zeros((RANDOM_BATCH_SIZE,), dtype=jnp.bool_),
        )

        def one_step(state, action):
            current, returns, successes = state
            next_carry, _next_observation, output = env.step(current, action)
            return (next_carry, returns + output.reward, successes | output.terminated), None

        (carry, returns, successes), _ = jax.lax.scan(
            one_step, (carry, *initial), action_sequence
        )
        finite = jnp.logical_and(
            jnp.all(jnp.isfinite(carry.data.qpos), axis=-1),
            jnp.all(jnp.isfinite(carry.data.qvel), axis=-1),
        )
        return returns, successes, finite

    returns, successes, finite = jax.jit(rollout)(world_keys, actions)
    returns, successes, finite = map(np.asarray, (returns, successes, finite))
    print("random lanes:", RANDOM_BATCH_SIZE, "horizon:", RANDOM_HORIZON)
    print("random successes:", int(successes.sum()))
    print("random mean return:", float(returns.mean()))
    print("random finite lanes:", int(finite.sum()), "/", RANDOM_BATCH_SIZE)
    if successes.any() or returns.mean() >= 2.5 or not finite.all():
        raise RuntimeError("random-action comparison did not meet Step 7 acceptance")


def main() -> None:
    env = So101Env.from_scene(1, randomization=NO_RANDOMIZATION)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    carry, _ = reset(jax.random.split(jax.random.key(20260905), 1))
    host_model = mujoco.MjModel.from_xml_path("sim/scene_cube_tub.xml")

    # The calibrated proxy centers are symmetric when closed.  Close above the
    # cube, then descend on that shared centerline so both pads arrive together.
    targets = [
        (pad_midpoint_ik(host_model, np.array([0.22, 0.00, 0.091])), -0.174, 35),
        (pad_midpoint_ik(host_model, np.array([0.22, 0.00, 0.021])), -0.174, 40),
        (pad_midpoint_ik(host_model, np.array([0.22, 0.00, 0.120])), -0.174, 30),
        (pad_midpoint_ik(host_model, np.array([0.28, 0.15, 0.120])), -0.174, 50),
        (pad_midpoint_ik(host_model, np.array([0.28, 0.15, 0.025])), -0.174, 30),
        (pad_midpoint_ik(host_model, np.array([0.28, 0.15, 0.025])), 1.70, 40),
    ]
    total = 0.0
    output = None
    for stage, (arm_target, gripper_target, controls) in enumerate(targets, start=1):
        target = jnp.asarray(np.concatenate((arm_target, [gripper_target])), dtype=jnp.float32)
        carry, output, score = follow_target(env, step, carry, target, controls)
        total += score
        print(
            f"stage {stage}: reward={score:.3f} cube={np.asarray(carry.data.xpos[0, env.reward_ids.cube_body]).round(3)} "
            f"q={np.asarray(carry.data.qpos[0, :6]).round(3)} "
            f"ctrl={np.asarray(carry.command_target[0]).round(3)} "
            f"contacts=({bool(output.diagnostics.fixed_jaw_contact[0])}, "
            f"{bool(output.diagnostics.moving_jaw_contact[0])}) "
            f"secure={bool(output.diagnostics.secure_pickup[0])}"
        )
    print("scripted return:", total)
    print("scripted secure pickup:", bool(output.diagnostics.secure_pickup[0]))
    print("scripted success:", bool(output.terminated[0]))
    if not bool(output.terminated[0]) or total < 11.0:
        raise RuntimeError("scripted pick-place did not meet Step 7 acceptance")
    random_action_comparison()


if __name__ == "__main__":
    main()
