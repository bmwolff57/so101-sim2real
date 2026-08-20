"""Manually randomize visual scene parameters and build a contact sheet.

After running this, you should understand that basic domain randomization is
often just a loop that mutates MuJoCo model arrays before rendering. Modify the
jitter ranges, color ranges, seed, or sample count and re-run to see a new
synthetic visual domain.
"""

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


SCENE_XML = """
<mujoco model="domain_randomization_basics">
  <asset>
    <material name="box_material" rgba="0.2 0.6 0.9 1"/>
  </asset>
  <worldbody>
    <light name="random_light" pos="0 -1 2" diffuse="1 1 1"/>
    <geom type="plane" size="3 3 0.1" rgba="0.28 0.28 0.28 1"/>
    <geom type="box" size="0.3 0.3 0.3" pos="0 0 0.3" material="box_material"/>
    <camera name="random_camera" pos="1.3 -1.3 1.0"
            xyaxes="0.707107 0.707107 0 -0.345874 0.345874 0.872872" fovy="50"/>
  </worldbody>
</mujoco>
"""

IMAGE_SIZE = 128
SAMPLE_COUNT = 10


def main() -> None:
    tutorial_directory = Path(__file__).resolve().parent
    output_directory = tutorial_directory / "output"
    output_directory.mkdir(exist_ok=True)

    # A fixed seed makes this first experiment repeatable. Change it for a new set.
    random_generator = np.random.default_rng(seed=7)

    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "random_light")
    material_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "box_material")
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "random_camera")

    # Start from this position every time so jitter is centered at one useful view.
    base_camera_position = model.cam_pos[camera_id].copy()
    rendered_images: list[Image.Image] = []
    renderer = mujoco.Renderer(model, width=IMAGE_SIZE, height=IMAGE_SIZE)

    for sample_number in range(SAMPLE_COUNT):
        # These assignments change the C-engine model directly; no framework is needed.
        model.light_pos[light_id] = random_generator.uniform(
            low=(-1.0, -1.0, 0.8),
            high=(1.0, 1.0, 2.2),
        )
        model.light_diffuse[light_id] = random_generator.uniform(low=0.25, high=1.0, size=3)
        model.mat_rgba[material_id] = (
            random_generator.uniform(0.1, 1.0),
            random_generator.uniform(0.1, 1.0),
            random_generator.uniform(0.1, 1.0),
            1.0,
        )
        camera_jitter = random_generator.uniform(low=-0.12, high=0.12, size=3)
        model.cam_pos[camera_id] = base_camera_position + camera_jitter

        # Forward recomputes derived camera and lighting state after model changes.
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="random_camera")
        rgb_pixels = renderer.render()

        # copy() keeps this sample independent from any later renderer buffer use.
        individual_image = Image.fromarray(rgb_pixels).copy()
        individual_path = output_directory / f"07_random_{sample_number:02d}.png"
        individual_image.save(individual_path)
        rendered_images.append(individual_image)

    renderer.close()

    # Paste ten same-sized images into a 2-row by 5-column contact sheet.
    grid_image = Image.new("RGB", (IMAGE_SIZE * 5, IMAGE_SIZE * 2))
    for image_index, individual_image in enumerate(rendered_images):
        column_index = image_index % 5
        row_index = image_index // 5
        paste_position = (column_index * IMAGE_SIZE, row_index * IMAGE_SIZE)
        grid_image.paste(individual_image, paste_position)
    grid_path = output_directory / "07_randomization_grid.png"
    grid_image.save(grid_path)

    print(f"Saved {SAMPLE_COUNT} individual 07_random_*.png images and {grid_path}")
    print("Look for changes in shadows, box color, and framing. Try widening one random range or changing the seed, then re-run.")


if __name__ == "__main__":
    main()
