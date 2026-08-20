"""Compare camera field of view and image resolution.

After running this, you should understand that fovy changes the amount of scene
visible through the same camera pose, while renderer width and height change the
number of image pixels. Modify the fovy list or resolution list and re-run.
"""

from pathlib import Path

import mujoco
from PIL import Image


SCENE_XML = """
<mujoco model="fovy_and_intrinsics">
  <worldbody>
    <light pos="0 -1 2" diffuse="1 1 1"/>
    <geom type="plane" size="3 3 0.1" rgba="0.28 0.28 0.28 1"/>
    <geom type="box" size="0.25 0.25 0.25" pos="0 0 0.25" rgba="0.2 0.6 0.9 1"/>
    <geom type="sphere" size="0.14" pos="0.7 0.25 0.14" rgba="0.9 0.2 0.15 1"/>
    <geom type="cylinder" size="0.12 0.3" pos="-0.45 -0.3 0.3" rgba="0.2 0.8 0.3 1"/>
    <camera name="view" pos="1.3 -1.3 1.0"
            xyaxes="0.707107 0.707107 0 -0.345874 0.345874 0.872872" fovy="60"/>
  </worldbody>
</mujoco>
"""


def render_to_file(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    width: int,
    height: int,
    output_path: Path,
) -> None:
    """Create a renderer at one resolution and save its view camera image."""
    renderer = mujoco.Renderer(model, width=width, height=height)
    renderer.update_scene(data, camera="view")
    rgb_pixels = renderer.render()
    renderer.close()
    Image.fromarray(rgb_pixels).save(output_path)


def main() -> None:
    tutorial_directory = Path(__file__).resolve().parent
    output_directory = tutorial_directory / "output"
    output_directory.mkdir(exist_ok=True)

    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "view")

    # All three images use the same resolution and camera position. Only fovy changes.
    for fovy_degrees in (30, 60, 90):
        model.cam_fovy[camera_id] = fovy_degrees
        mujoco.mj_forward(model, data)
        render_to_file(model, data, 256, 256, output_directory / f"06_fovy_{fovy_degrees}.png")

    # All three images use the same 60-degree field of view. Only pixel resolution changes.
    model.cam_fovy[camera_id] = 60
    mujoco.mj_forward(model, data)
    for resolution in (64, 128, 256):
        render_to_file(
            model,
            data,
            resolution,
            resolution,
            output_directory / f"06_resolution_{resolution}.png",
        )

    print(f"Saved six 06_fovy_*.png and 06_resolution_*.png images in {output_directory}")
    print("Compare how fovy changes framing and resolution changes pixel detail. Try another fovy or resolution, then re-run.")


if __name__ == "__main__":
    main()
