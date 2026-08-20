"""Change lights and material colors, then compare their rendered effects.

After running this, you should understand that lights describe illumination while
materials describe how surfaces respond to it. Modify a light diffuse value,
light position, or material rgba value and re-run to build intuition.
"""

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


SCENE_XML = """
<mujoco model="lights_and_materials">
  <asset>
    <material name="box_material" rgba="0.2 0.6 0.9 1"/>
  </asset>
  <worldbody>
    <light name="scene_light" pos="0 -1 2" diffuse="1 1 1" specular="0.2 0.2 0.2"/>
    <geom type="plane" size="2 2 0.1" rgba="0.25 0.25 0.25 1"/>
    <geom name="box" type="box" size="0.3 0.3 0.3" pos="0 0 0.3" material="box_material"/>
    <camera name="view" pos="1.2 -1.2 0.9"
            xyaxes="0.707107 0.707107 0 -0.331295 0.331295 0.883452" fovy="45"/>
  </worldbody>
</mujoco>
"""


def save_current_render(renderer: mujoco.Renderer, data: mujoco.MjData, output_path: Path) -> None:
    """Render the current model parameters from the one fixed camera."""
    renderer.update_scene(data, camera="view")
    rgb_pixels = renderer.render()
    Image.fromarray(rgb_pixels).save(output_path)


def main() -> None:
    tutorial_directory = Path(__file__).resolve().parent
    output_directory = tutorial_directory / "output"
    output_directory.mkdir(exist_ok=True)

    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "scene_light")
    material_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "box_material")
    renderer = mujoco.Renderer(model, width=256, height=256)

    # First keep the blue material fixed and alter only the light's brightness/color.
    lighting_examples = {
        "dim": (0.25, 0.25, 0.25),
        "bright": (1.0, 1.0, 1.0),
        "colored": (1.0, 0.25, 0.25),
    }
    model.mat_rgba[material_id] = (0.2, 0.6, 0.9, 1.0)
    for name, diffuse_color in lighting_examples.items():
        model.light_diffuse[light_id] = diffuse_color
        mujoco.mj_forward(model, data)
        save_current_render(renderer, data, output_directory / f"04_light_{name}.png")

    # Next use one bright white light and alter only the material's surface color.
    model.light_diffuse[light_id] = (1.0, 1.0, 1.0)
    material_examples = {
        "red": (0.9, 0.15, 0.15, 1.0),
        "green": (0.15, 0.8, 0.2, 1.0),
        "blue": (0.2, 0.4, 0.95, 1.0),
    }
    for name, rgba in material_examples.items():
        model.mat_rgba[material_id] = np.array(rgba)
        mujoco.mj_forward(model, data)
        save_current_render(renderer, data, output_directory / f"04_material_{name}.png")

    renderer.close()

    print(f"Saved six 04_light_*.png and 04_material_*.png images in {output_directory}")
    print("Compare changing illumination with changing surface color. Try moving scene_light or editing a rgba tuple, then re-run.")


if __name__ == "__main__":
    main()
