# pylint: disable=missing-module-docstring, wrong-import-position, ungrouped-imports, too-few-public-methods, consider-using-enumerate, line-too-long, line-too-long, too-many-function-args, too-many-arguments

import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)


from bad.rewardshape_setting import RewardShapeSetting

class BadSetting:
    """bad setting"""
    def __init__(self, with_reward_shaping: bool, batch_size:int, epoch_size: int, gamma:float, learning_rate: float, rewardshape_setting: RewardShapeSetting, reward_threshold:int) -> None:
        """init"""
        self.epoch_size = epoch_size
        self.with_reward_shaping = with_reward_shaping
        self.batch_size: int = batch_size
        self.gamma: float = gamma
        self.learning_rate = learning_rate
        self.rewardshape_setting = rewardshape_setting
        self.reward_threshold = reward_threshold
