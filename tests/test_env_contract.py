"""Structural tests for the mechanics-only SO-101 environment contract."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from sim.autoreset import select_reset
from sim.env import Diagnostics, RewardTerms, So101Env, StepOutput, raise_if_nonfinite


class EnvironmentContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.batch_size = 2
        cls.env = So101Env.from_scene(batch_size=cls.batch_size)
        cls.keys = jax.random.split(jax.random.key(7), cls.batch_size)
        # Store these as static methods: a plain callable stored on a
        # ``unittest.TestCase`` class is a descriptor and would otherwise bind
        # ``self`` as an unintended first JAX argument.
        cls.reset = staticmethod(jax.jit(cls.env.reset))
        cls.step = staticmethod(jax.jit(cls.env.step))
        cls.carry, cls.observation = cls.reset(cls.keys)
        jax.block_until_ready(cls.observation["wrist"])

    def test_reset_is_deterministic_and_observation_contract_is_exact(self) -> None:
        carry_again, observation_again = self.reset(self.keys)
        jax.block_until_ready(observation_again["wrist"])
        self.assertEqual(set(self.observation), {"wrist", "overhead", "proprio"})
        self.assertEqual(self.observation["wrist"].shape, (2, 128, 128, 3))
        self.assertEqual(self.observation["overhead"].shape, (2, 128, 128, 3))
        self.assertEqual(self.observation["proprio"].shape, (2, 24))
        self.assertEqual(self.observation["wrist"].dtype, jnp.uint8)
        self.assertEqual(self.observation["overhead"].dtype, jnp.uint8)
        self.assertEqual(self.observation["proprio"].dtype, jnp.float32)
        np.testing.assert_array_equal(carry_again.data.qpos, self.carry.data.qpos)
        np.testing.assert_array_equal(observation_again["proprio"], self.observation["proprio"])

    def test_action_clipping_and_control_timing(self) -> None:
        saturated = jnp.full((2, 6), 2.0, dtype=jnp.float32)
        carry, _observation, output = self.step(self.carry, saturated)
        expected = jnp.clip(
            self.carry.command_target + self.env.action_scale,
            self.env.ctrl_low,
            self.env.ctrl_high,
        )
        np.testing.assert_allclose(carry.command_target, expected)
        np.testing.assert_allclose(carry.data.time - self.carry.data.time, 0.05, atol=1e-6)
        self.assertTrue(np.asarray(output.diagnostics.physics_finite).all())

    def test_cube_state_cannot_change_proprio(self) -> None:
        cube_joint = mujoco.mj_name2id(
            self.env.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free"
        )
        cube_start = int(self.env.mj_model.jnt_qposadr[cube_joint])
        altered_qpos = self.carry.data.qpos.at[:, cube_start].add(0.01)
        altered_data = self.carry.data.replace(qpos=altered_qpos)
        _data, altered_obs = self.env._render_observation(
            altered_data,
            self.carry.previous_action,
            self.carry.command_target,
        )
        jax.block_until_ready(altered_obs["proprio"])
        np.testing.assert_array_equal(altered_obs["proprio"], self.observation["proprio"])

    def test_zero_reward_and_precise_truncation(self) -> None:
        carry_at_limit = self.carry._replace(
            step=jnp.full((2,), 199, dtype=jnp.int32)
        )
        action = jnp.zeros((2, 6), dtype=jnp.float32)
        _carry, _observation, output = self.step(carry_at_limit, action)
        np.testing.assert_array_equal(output.reward, jnp.zeros((2,), dtype=jnp.float32))
        np.testing.assert_array_equal(
            output.reward_terms.mechanics_only, jnp.zeros((2,), dtype=jnp.float32)
        )
        self.assertFalse(np.asarray(output.terminated).any())
        self.assertTrue(np.asarray(output.truncated).all())

    def test_nonfinite_host_boundary_and_autoreset_selection(self) -> None:
        failed = StepOutput(
            reward=jnp.zeros((2,), dtype=jnp.float32),
            reward_terms=RewardTerms(mechanics_only=jnp.zeros((2,), dtype=jnp.float32)),
            terminated=jnp.zeros((2,), dtype=jnp.bool_),
            truncated=jnp.zeros((2,), dtype=jnp.bool_),
            diagnostics=Diagnostics(physics_finite=jnp.array([True, False])),
        )
        with self.assertRaises(FloatingPointError):
            raise_if_nonfinite(failed)
        selected = select_reset(
            jnp.array([False, True]),
            jnp.array([[10.0], [20.0]]),
            jnp.array([[1.0], [2.0]]),
        )
        np.testing.assert_array_equal(selected, jnp.array([[1.0], [20.0]]))


if __name__ == "__main__":
    unittest.main()
