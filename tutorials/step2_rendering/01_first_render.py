"""Render one MuJoCo frame and save it as a PNG.

After running this, you should understand the smallest useful MuJoCo rendering
pipeline: make a model and data object, update a Renderer, then save its RGB
pixels. Modify the box's size, position, or rgba value and run it again to see
that the rendered image comes directly from the MJCF scene.
"""

from pathlib import Path

import mujoco
from PIL import Image


# Keeping the MJCF beside the Python code makes this tutorial fully standalone.
SCENE_XML = """
<mujoco model="first_render">
  <visual>
    <global offwidth="256" offheight="256"/>
  </visual>
  <worldbody>
    <light pos="0 0 2" diffuse="1 1 1"/>
    <geom name="box" type="box" size="0.25 0.25 0.25" rgba="0.2 0.6 0.9 1"/>
    <camera name="view" pos="1.2 -1.2 0.9"
            xyaxes="0.707107 0.707107 0 -0.331295 0.331295 0.883452"/>
  </worldbody>
</mujoco>
"""


def main() -> None:
    # These paths work even though the command is run from the project root.
    tutorial_directory = Path(__file__).resolve().parent
    output_directory = tutorial_directory / "output"
    output_directory.mkdir(exist_ok=True)
    output_path = output_directory / "01_first_render.png"

    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)

    # mj_forward computes camera and geometry positions for the current state.
    mujoco.mj_forward(model, data)

    # A Renderer owns an off-screen framebuffer. Width is written before height
    # because that is the order used by MuJoCo's Python API.
    renderer = mujoco.Renderer(model, width=256, height=256)
    renderer.update_scene(data, camera="view")
    rgb_pixels = renderer.render()
    renderer.close()

    Image.fromarray(rgb_pixels).save(output_path)

    print(f"Saved {output_path}")
    print(f"Image shape: {rgb_pixels.shape}")
    print(f"Image dtype: {rgb_pixels.dtype}")
    print("Look for one blue box. Try changing the box rgba or size in SCENE_XML, then re-run.")


if __name__ == "__main__":
    main()
