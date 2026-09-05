"""Fast host-MuJoCo diagnostic for the two Step 7 jaw-pad proxies.

This is not a policy and not a reward test.  It places the cube at the
calculated closed-pad midpoint, verifies both touch sensors, then commands a
10 cm lift through the ordinary position actuators.  Use it after each small
pad XML adjustment before rerunning the full MJX reward smoke test.
"""

from __future__ import annotations

import mujoco
import numpy as np


SCENE = "sim/scene_cube_tub.xml"
CUBE_POSITION = np.array([0.22, 0.00, 0.040])
LIFT_HEIGHT = 0.10


def closed_pad_midpoint_ik(model: mujoco.MjModel, target: np.ndarray) -> np.ndarray:
    """Solve arm joints so the closed-proxy midpoint reaches `target`."""
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
        for joint in range(5):
            perturbed = q.copy()
            perturbed[joint] += 1e-5
            jacobian[:, joint] = (midpoint(perturbed) - point) / 1e-5
        q[:5] = np.clip(
            q[:5] + 0.5 * jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 1e-4 * np.eye(3), error
            ),
            lower,
            upper,
        )
    return q


def main() -> None:
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    fixed = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_pad")
    moving = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_pad")
    cube_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_adr = int(model.jnt_qposadr[cube_joint])

    closed = closed_pad_midpoint_ik(model, CUBE_POSITION)
    data.qpos[:6] = closed
    data.qpos[cube_adr : cube_adr + 3] = CUBE_POSITION
    data.qpos[cube_adr + 3 : cube_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.ctrl[:] = closed
    mujoco.mj_forward(model, data)
    fixed_force, moving_force = data.sensordata[:2]
    print("closed joint target:", np.round(closed, 5))
    print("fixed pad world position:", np.round(data.geom_xpos[fixed], 5))
    print("moving pad world position:", np.round(data.geom_xpos[moving], 5))
    print("touch forces [fixed, moving]:", np.round(data.sensordata[:2], 5))

    # Let the position servos/contact solver establish the closed pinch before
    # commanding the lift; otherwise the first lift impulse can eject a cube
    # that began in slight proxy penetration.
    for _ in range(100):
        mujoco.mj_step(model, data)
    lift_target = closed_pad_midpoint_ik(model, CUBE_POSITION + [0.0, 0.0, LIFT_HEIGHT])
    data.ctrl[:] = lift_target
    for _ in range(250):
        mujoco.mj_step(model, data)
    cube_height = float(data.xpos[cube_body, 2])
    held = fixed_force > 1e-6 and moving_force > 1e-6 and cube_height >= CUBE_POSITION[2] + 0.05
    print("cube height after commanded lift:", f"{cube_height:.5f} m")
    print("RESULT:", "PASS — opposing contact and retained lift" if held else "FAIL — tune jaw pads")
    if not held:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
