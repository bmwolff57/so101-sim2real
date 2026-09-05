"""Static reward and success thresholds for the Step 7 task slice."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskRewardConfig:
    """Working-hypothesis reward parameters; all units are SI."""

    name: str = "step7_baseline"
    reach_scale_m: float = 0.060
    transport_scale_m: float = 0.100
    pickup_height_m: float = 0.045
    reach_weight: float = 0.010
    transport_weight: float = 0.020
    pickup_bonus: float = 1.0
    success_bonus: float = 10.0
    containment_half_extent_m: float = 0.042
    max_cube_center_height_m: float = 0.032
    max_linear_speed_mps: float = 0.050
    max_angular_speed_radps: float = 1.0
    settle_control_steps: int = 5


STEP7_REWARD = TaskRewardConfig()
