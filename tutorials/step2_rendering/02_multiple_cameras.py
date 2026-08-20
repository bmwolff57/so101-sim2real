"""Declare and render from multiple named MJCF cameras.

After running this, you should understand that cameras are named scene objects
and that Renderer.update_scene selects one by its name. Modify either camera's
pos, xyaxes, or fovy attributes and compare the two saved views.
"""

from pathlib import Path

import mujoco
from PIL import Image


SCENE_XML = """
<mujoco model="multiple_cameras">
  <worldbody>
    <light pos="0 0 2" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.25 0.25 0.25 1"/>
    <geom name="box" type="box" size="0.25 0.25 0.25" pos="0 0 0.25"
          rgba="0.2 0.6 0.9 1"/>

    <!-- This camera is fixed in worldbody and looks straight down. -->
    <camera name="overhead" pos="0 0 1.6" xyaxes="1 0 0 0 1 0" fovy="45"/>

    <!-- This camera is also fixed in worldbody, but looks toward the origin. -->
    <camera name="angled" pos="1.2 -1.2 0.9"
            xyaxes="0.707107 0.707107 0 -0.331295 0.331295 0.883452" fovy="45"/>
  </worldbody>
</mujoco>
"""


def render_named_camera(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera_name: str,
    output_path: Path,
) -> None:
    """Render the current simulation state from one named camera."""
    renderer.update_scene(data, camera=camera_name)
    rgb_pixels = renderer.render()
    Image.fromarray(rgb_pixels).save(output_path)


def main() -> None:
    tutorial_directory = Path(__file__).resolve().parent
    output_directory = tutorial_directory / "output"
    output_directory.mkdir(exist_ok=True)

    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, width=256, height=256)
    render_named_camera(renderer, data, "overhead", output_directory / "02_overhead.png")
    render_named_camera(renderer, data, "angled", output_directory / "02_angled.png")
    renderer.close()

    print(f"Saved {output_directory / '02_overhead.png'}")
    print(f"Saved {output_directory / '02_angled.png'}")
    print("Compare the overhead and angled views. Try moving one named <camera> in SCENE_XML, then re-run.")


if __name__ == "__main__":
    main()
