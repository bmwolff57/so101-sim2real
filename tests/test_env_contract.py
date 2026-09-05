"""Structural tests for the Step 6 SO-101 randomization contract."""

from __future__ import annotations

from dataclasses import replace
import unittest

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from configs.domain_randomization import NO_RANDOMIZATION, STAGED_RANDOMIZATION
from sim.autoreset import select_reset
from sim.env import Diagnostics, RewardTerms, So101Env, StepOutput, raise_if_nonfinite
from sim.randomization import (
    MAX_ACTION_DELAY_STEPS,
    MAX_IMAGE_DELAY_STEPS,
    NOMINAL_CUBE_INERTIA,
    NOMINAL_CUBE_MASS,
    apply_params_to_model,
    sample_params,
)


class EnvironmentContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.batch_size = 2
        cls.env = So101Env.from_scene(
            batch_size=cls.batch_size, randomization=STAGED_RANDOMIZATION
        )
        cls.keys = jax.random.split(jax.random.key(7), cls.batch_size)
        cls.reset = staticmethod(jax.jit(cls.env.reset))
        cls.step = staticmethod(jax.jit(cls.env.step))
        cls.carry, cls.observation = cls.reset(cls.keys)
        jax.block_until_ready(cls.observation["wrist"])

    def test_same_keys_recreate_full_staged_world(self) -> None:
        carry_again, observation_again = self.reset(self.keys)
        jax.block_until_ready(observation_again["wrist"])
        self.assertEqual(set(self.observation), {"wrist", "overhead", "proprio"})
        self.assertEqual(self.observation["wrist"].shape, (2, 128, 128, 3))
        self.assertEqual(self.observation["overhead"].shape, (2, 128, 128, 3))
        self.assertEqual(self.observation["proprio"].shape, (2, 24))
        self.assertEqual(self.observation["wrist"].dtype, jnp.uint8)
        self.assertEqual(self.observation["overhead"].dtype, jnp.uint8)
        self.assertEqual(self.observation["proprio"].dtype, jnp.float32)
        np.testing.assert_array_equal(
            jax.random.key_data(carry_again.params.world_key),
            jax.random.key_data(self.carry.params.world_key),
        )
        for name in carry_again.params._fields[1:]:
            np.testing.assert_array_equal(
                getattr(carry_again.params, name), getattr(self.carry.params, name)
            )
        np.testing.assert_array_equal(carry_again.data.qpos, self.carry.data.qpos)
        np.testing.assert_array_equal(observation_again["wrist"], self.observation["wrist"])

    def test_no_randomization_preserves_nominal_parameters(self) -> None:
        env = So101Env.from_scene(batch_size=2, randomization=NO_RANDOMIZATION)
        params = sample_params(self.keys, NO_RANDOMIZATION, env.model, env.randomization_ids)
        np.testing.assert_allclose(params.cube_mass, NOMINAL_CUBE_MASS)
        np.testing.assert_allclose(params.cube_inertia, NOMINAL_CUBE_INERTIA)
        np.testing.assert_allclose(
            params.cube_position, np.tile([0.22, 0.0, 0.021], (2, 1))
        )
        np.testing.assert_allclose(params.overhead_camera_fovy, 43.0)
        np.testing.assert_allclose(params.wrist_camera_fovy, 81.0)
        np.testing.assert_array_equal(params.action_delay_steps, 0)
        np.testing.assert_array_equal(params.image_delay_steps, 0)

    def test_staged_ranges_and_cube_spawn_are_physical(self) -> None:
        params = self.carry.params
        self.assertTrue(np.all(np.asarray(params.cube_mass) >= 0.0276))
        self.assertTrue(np.all(np.asarray(params.cube_mass) <= 0.0414))
        self.assertTrue(np.all(np.asarray(params.cube_position[:, 0]) >= 0.16))
        self.assertTrue(np.all(np.asarray(params.cube_position[:, 0]) <= 0.28))
        self.assertTrue(np.all(np.asarray(params.cube_position[:, 1]) >= -0.10))
        self.assertTrue(np.all(np.asarray(params.cube_position[:, 1]) <= 0.02))
        np.testing.assert_allclose(params.cube_position[:, 2], 0.021)
        # The spawn rectangle ends 6 cm before the provisional tub's nearest
        # x/y extent, and its 20 mm half-height is clear of the table top.
        np.testing.assert_allclose(
            params.cube_inertia / NOMINAL_CUBE_INERTIA,
            np.broadcast_to(
                (params.cube_mass / NOMINAL_CUBE_MASS)[:, None], (2, 3)
            ),
        )
        np.testing.assert_array_equal(self.carry.data.qpos[:, -4:], [[1, 0, 0, 0]] * 2)

    def test_every_active_visual_axis_stays_in_its_declared_range(self) -> None:
        params = self.carry.params
        ids = self.env.randomization_ids
        config = STAGED_RANDOMIZATION
        overhead_delta = params.overhead_camera_position - self.env.model.cam_pos[ids.overhead_camera]
        wrist_delta = params.wrist_camera_position - self.env.model.cam_pos[ids.wrist_camera]
        light_delta = params.light_position - self.env.model.light_pos[ids.light]
        self.assertTrue(np.all(np.abs(np.asarray(overhead_delta)) <= config.overhead_position_radius))
        self.assertTrue(np.all(np.abs(np.asarray(wrist_delta)) <= config.wrist_position_radius))
        self.assertTrue(np.all(np.abs(np.asarray(light_delta)) <= config.light_position_radius))
        self.assertTrue(np.all(np.asarray(params.overhead_camera_fovy) >= 41.0))
        self.assertTrue(np.all(np.asarray(params.overhead_camera_fovy) <= 45.0))
        self.assertTrue(np.all(np.asarray(params.wrist_camera_fovy) >= 78.0))
        self.assertTrue(np.all(np.asarray(params.wrist_camera_fovy) <= 84.0))
        self.assertTrue(np.all(np.asarray(params.light_rgb_scale) >= 0.75))
        self.assertTrue(np.all(np.asarray(params.light_rgb_scale) <= 1.25))
        self.assertTrue(np.all(np.asarray(params.material_brightness) >= 0.85))
        self.assertTrue(np.all(np.asarray(params.material_brightness) <= 1.15))
        self.assertTrue(np.isin(np.asarray(params.table_material_index), [0, 1, 2]).all())

    def test_model_leaf_replacement_has_scoped_effects(self) -> None:
        params = self.carry.params
        model, _axes = apply_params_to_model(
            self.env.model, params, self.env.randomization_ids
        )
        ids = self.env.randomization_ids
        np.testing.assert_allclose(model.body_mass[:, ids.cube_body], params.cube_mass)
        np.testing.assert_allclose(model.body_inertia[:, ids.cube_body], params.cube_inertia)
        np.testing.assert_allclose(
            model.cam_fovy[:, ids.overhead_camera], params.overhead_camera_fovy
        )
        self.assertTrue(
            np.isin(
                np.asarray(model.geom_matid[:, ids.table_geom]),
                np.asarray(ids.table_material_ids),
            ).all()
        )
        np.testing.assert_allclose(
            model.geom_friction[:, ids.cube_geom],
            np.broadcast_to(self.env.model.geom_friction[ids.cube_geom], (2, 3)),
        )
        np.testing.assert_allclose(
            model.geom_solref[:, ids.cube_geom],
            np.broadcast_to(self.env.model.geom_solref[ids.cube_geom], (2, 2)),
        )

    def test_action_clipping_and_control_timing(self) -> None:
        saturated = jnp.full((2, 6), 2.0, dtype=jnp.float32)
        carry, _observation, output = self.step(self.carry, saturated)
        expected = jnp.clip(
            self.carry.command_target + self.env.action_scale,
            self.env.ctrl_low,
            self.env.ctrl_high,
        )
        np.testing.assert_allclose(carry.command_target, expected)
        np.testing.assert_allclose(carry.previous_action, jnp.ones((2, 6)))
        np.testing.assert_allclose(carry.data.time - self.carry.data.time, 0.05, atol=1e-6)
        self.assertTrue(np.asarray(output.diagnostics.physics_finite).all())

    def test_action_and_image_delay_selection_is_exact(self) -> None:
        requested = jnp.full((2, 6), 9.0)
        history = jnp.broadcast_to(jnp.arange(5, dtype=jnp.float32)[None, :, None], (2, 5, 6))
        for delay, expected in ((0, 9.0), (1, 4.0), (MAX_ACTION_DELAY_STEPS, 0.0)):
            selected, next_history = self.env._select_executed_action(
                requested, history, jnp.full((2,), delay, dtype=jnp.int32)
            )
            np.testing.assert_allclose(selected, expected)
            np.testing.assert_allclose(next_history[:, -1], 9.0)

        current = jnp.full((2, 128, 128, 3), 9, dtype=jnp.uint8)
        image_history = jnp.stack(
            (jnp.full_like(current, 1), jnp.full_like(current, 2)), axis=1
        )
        for delay, expected in ((0, 9), (1, 2), (MAX_IMAGE_DELAY_STEPS, 1)):
            selected, next_history = self.env._select_delayed_image(
                current, image_history, jnp.full((2,), delay, dtype=jnp.int32)
            )
            np.testing.assert_array_equal(selected, expected)
            np.testing.assert_array_equal(next_history[:, -1], 9)
        np.testing.assert_array_equal(self.carry.action_history, 0)
        np.testing.assert_array_equal(self.carry.camera_history.wrist[:, 0], self.observation["wrist"])

    def test_cube_state_cannot_change_proprio(self) -> None:
        cube_joint = mujoco.mj_name2id(
            self.env.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free"
        )
        cube_start = int(self.env.mj_model.jnt_qposadr[cube_joint])
        altered_data = self.carry.data.replace(
            qpos=self.carry.data.qpos.at[:, cube_start].add(0.01)
        )
        model, axes = apply_params_to_model(
            self.env.model, self.carry.params, self.env.randomization_ids
        )
        _data, altered_obs = self.env._render_observation(
            model, axes, altered_data, self.carry.previous_action, self.carry.command_target
        )
        jax.block_until_ready(altered_obs["proprio"])
        np.testing.assert_array_equal(altered_obs["proprio"], self.observation["proprio"])

    def test_zero_reward_precise_truncation_and_all_physical_axes_step(self) -> None:
        carry_at_limit = self.carry._replace(step=jnp.full((2,), 199, dtype=jnp.int32))
        _carry, _observation, output = self.step(carry_at_limit, jnp.zeros((2, 6)))
        np.testing.assert_array_equal(output.reward, jnp.zeros((2,), dtype=jnp.float32))
        np.testing.assert_array_equal(output.reward_terms.mechanics_only, 0.0)
        self.assertFalse(np.asarray(output.terminated).any())
        self.assertTrue(np.asarray(output.truncated).all())

        for field in ("cube_friction_enabled", "arm_damping_enabled", "actuator_force_enabled", "cube_contact_enabled"):
            config = replace(NO_RANDOMIZATION, name=f"test_{field}", **{field: True})
            env = So101Env.from_scene(batch_size=1, randomization=config)
            carry, _ = jax.jit(env.reset)(jax.random.split(jax.random.key(11), 1))
            carry, _obs, output = jax.jit(env.step)(carry, jnp.zeros((1, 6)))
            jax.block_until_ready(carry.data.qpos)
            self.assertTrue(np.asarray(output.diagnostics.physics_finite).all(), field)

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
        selected = select_reset(jnp.array([False, True]), jnp.array([[10.0], [20.0]]), jnp.array([[1.0], [2.0]]))
        np.testing.assert_array_equal(selected, jnp.array([[1.0], [20.0]]))


if __name__ == "__main__":
    unittest.main()
