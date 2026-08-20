"""Render the project scene through a gripper-mounted wrist camera and an overhead camera.

After running this, you should understand how the camera ideas in this folder
apply to the SO-101 cube-and-tub scene without modifying the production XML.
Modify the camera pos, xyaxes, fovy, or robot joint qpos values and re-run to
compare simulated viewpoints with your real cameras.
"""

from pathlib import Path

import mujoco
from PIL import Image


IMAGE_SIZE = 128


def render_named_camera(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera_name: str,
    output_path: Path,
) -> None:
    """Render one named camera from the current SO-101 simulation state."""
    renderer.update_scene(data, camera=camera_name)
    rgb_pixels = renderer.render()
    Image.fromarray(rgb_pixels).save(output_path)


def main() -> None:
    tutorial_directory = Path(__file__).resolve().parent
    output_directory = tutorial_directory / "output"
    output_directory.mkdir(exist_ok=True)
    scene_path = tutorial_directory / "scene_with_cameras.xml"

    # This tutorial-local copy includes the untouched production calibration XML.
    # It adds only an overhead camera itself; this script adds the wrist camera
    # after loading because the gripper body comes from that included calibration.
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # The calibration scene's gripper body is named "gripper". Add a camera to
    # that existing body without editing sim/scene_cube_tub.xml or its include.
    gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    wrist_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    model.cam_bodyid[wrist_camera_id] = gripper_body_id

    # This local pose is expressed in the gripper body's frame. Its optical axis
    # points along local negative z, approximately away from the gripper mount.
    model.cam_pos[wrist_camera_id] = (-0.008, 0.0, -0.098)
    model.cam_quat[wrist_camera_id] = (0.707107, 0.0, 0.707107, 0.0)

    # 81 degrees is the vertical FOV corresponding to a 120-degree diagonal FOV
    # at a 16:9 aspect ratio. 43 degrees similarly approximates a 78-degree C920.
    model.cam_fovy[wrist_camera_id] = 81

    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, width=IMAGE_SIZE, height=IMAGE_SIZE)
    render_named_camera(renderer, data, "wrist_camera", output_directory / "08_wrist_camera.png")
    render_named_camera(renderer, data, "overhead_camera", output_directory / "08_overhead_camera.png")
    renderer.close()

    print(f"Saved {output_directory / '08_wrist_camera.png'}")
    print(f"Saved {output_directory / '08_overhead_camera.png'}")
    print("Compare the wrist and overhead views with your real cameras. Try changing a camera pose, fovy, or a robot joint qpos, then re-run.")


if __name__ == "__main__":
    main()
