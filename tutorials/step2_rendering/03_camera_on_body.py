"""Attach a camera to a freely moving body and render a frame sequence.

After running this, you should understand that a camera nested inside a body
inherits that body's position and orientation, unlike a camera in worldbody.
Modify the camera's local pos, the box's start height, or FRAME_COUNT and watch
how the sequence changes.
"""

from pathlib import Path

import mujoco
from PIL import Image


SCENE_XML = """
<mujoco model="camera_on_body">
  <option timestep="0.002" gravity="0 0 -1.81"/>
  <worldbody>
    <light pos="0 -1 2" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.35 0.35 0.35 1"/>
    <geom name="red_landmark" type="sphere" size="0.12" pos="0.65 0 0.12" rgba="0.9 0.1 0.1 1"/>
    <geom name="green_landmark" type="box" size="0.12 0.12 0.12" pos="-0.45 0.35 0.12" rgba="0.1 0.8 0.2 1"/>

    <body name="moving_box" pos="0 0 1.2">
      <freejoint/>
      <geom name="box" type="box" size="0.12 0.12 0.12" rgba="0.2 0.6 0.9 1"/>

      <!-- This is a body-attached camera. Its pos and xyaxes are local to the box. -->
      <!-- Its optical axis points down the box's local negative-z direction. -->
      <camera name="box_camera" pos="0 0 0.13" xyaxes="1 0 0 0 1 0" fovy="60"/>
    </body>
  </worldbody>
</mujoco>
"""

# 15 frames, 0.2 seconds apart, show roughly three seconds of falling and resting.
FRAME_COUNT = 15
STEPS_BETWEEN_FRAMES = 100


def main() -> None:
    tutorial_directory = Path(__file__).resolve().parent
    output_directory = tutorial_directory / "output"
    output_directory.mkdir(exist_ok=True)

    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=256, height=256)

    for frame_number in range(FRAME_COUNT):
        # Physics changes the moving_box pose. The nested camera follows automatically.
        for _ in range(STEPS_BETWEEN_FRAMES):
            mujoco.mj_step(model, data)

        renderer.update_scene(data, camera="box_camera")
        rgb_pixels = renderer.render()
        frame_path = output_directory / f"03_box_camera_{frame_number:02d}.png"
        Image.fromarray(rgb_pixels).save(frame_path)

    renderer.close()

    print(f"Saved {FRAME_COUNT} frames named 03_box_camera_00.png through 03_box_camera_14.png in {output_directory}")
    print("Watch the landmarks move in the box camera's view. Try changing the camera local pos or the box start height, then re-run.")


if __name__ == "__main__":
    main()
