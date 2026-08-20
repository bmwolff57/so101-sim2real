"""Apply a built-in MuJoCo checkerboard texture through a material.

After running this, you should understand the MJCF texture-to-material-to-geom
pipeline: a texture asset is referenced by a material, and the material is used
by a geom. Modify rgb1, rgb2, texrepeat, or the box size and re-run.
"""

from pathlib import Path

import mujoco
from PIL import Image


SCENE_XML = """
<mujoco model="textures">
  <asset>
    <!-- A built-in texture needs no external image file. -->
    <texture name="checker_texture" type="2d" builtin="checker"
             rgb1="0.95 0.95 0.95" rgb2="0.1 0.25 0.7" width="512" height="512"/>
    <material name="checker_material" texture="checker_texture" texrepeat="4 4"/>
  </asset>
  <worldbody>
    <light pos="0 -1 2" diffuse="1 1 1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.3 0.3 0.3 1"/>
    <geom name="textured_box" type="box" size="0.35 0.35 0.35" pos="0 0 0.35"
          material="checker_material"/>
    <camera name="view" pos="1.3 -1.3 1.0"
            xyaxes="0.707107 0.707107 0 -0.345874 0.345874 0.872872" fovy="45"/>
  </worldbody>
</mujoco>
"""


def main() -> None:
    tutorial_directory = Path(__file__).resolve().parent
    output_directory = tutorial_directory / "output"
    output_directory.mkdir(exist_ok=True)
    output_path = output_directory / "05_checker_texture.png"

    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, width=256, height=256)
    renderer.update_scene(data, camera="view")
    rgb_pixels = renderer.render()
    renderer.close()
    Image.fromarray(rgb_pixels).save(output_path)

    print(f"Saved {output_path}")
    print("Look for the checkerboard on the box faces. Try changing checker_texture rgb1/rgb2 or checker_material texrepeat, then re-run.")


if __name__ == "__main__":
    main()
