"""Static, typed project configuration presets."""

from configs.domain_randomization import NO_RANDOMIZATION, STAGED_RANDOMIZATION
from configs.task_reward import STEP7_REWARD, TaskRewardConfig

__all__ = ["NO_RANDOMIZATION", "STAGED_RANDOMIZATION", "STEP7_REWARD", "TaskRewardConfig"]
