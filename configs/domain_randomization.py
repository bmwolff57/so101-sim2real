"""Named domain-randomization presets for the SO-101 environment."""

from sim.randomization import RandomizationConfig


NO_RANDOMIZATION = RandomizationConfig(name="no_randomization")

STAGED_RANDOMIZATION = RandomizationConfig(
    name="staged_randomization",
    cube_mass_enabled=True,
    cube_spawn_enabled=True,
    overhead_camera_enabled=True,
    wrist_camera_enabled=True,
    light_enabled=True,
    material_color_enabled=True,
    table_material_enabled=True,
    # These mechanisms exist in the fixed-shape parameter contract but stay
    # dormant until measurements justify turning them on.
    cube_friction_enabled=False,
    arm_damping_enabled=False,
    actuator_force_enabled=False,
    cube_contact_enabled=False,
    action_delay_enabled=False,
    image_delay_enabled=False,
)
