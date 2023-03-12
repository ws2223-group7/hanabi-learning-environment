# pylint: disable=missing-module-docstring, wrong-import-position, ungrouped-imports, too-few-public-methods, line-too-long, consider-using-enumerate, unused-variable, too-many-locals

import sys
import os
import numpy as np

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.collect_batch_results import CollectBatchResults
from bad.rewards_to_go_episode_calculation_result import RewardsCalculationResult
from bad.rewards_to_go_calculation_result import RewardsToGoCalculationResult
from bad.reward_shape_converter import RewardShapeConverter
from bad.game_buffer import GameBuffer
from bad.bad_setting import BadSetting
from bad.rewardshape_setting import RewardShapeSetting


class RewardToGoCalculation:
    ''''calculate reward to go'''
    def __init__(self, bad_setting: BadSetting) -> None:
        self.gamma = bad_setting.gamma
        self.with_reward_shaping = bad_setting.with_reward_shaping

    def calculate(self, buffer: GameBuffer, result: RewardsCalculationResult, rewardshape_setting: RewardShapeSetting) -> None:
        ''''calculate episode'''

        reward_shape_converter = RewardShapeConverter()

        for index in range(len(buffer.bayesian_actions)): # über jede aktion (pro spiel)
            reward_shape = reward_shape_converter.convert(buffer.reward_shapes[index], rewardshape_setting)
            # hier rewards verändern
            reward_to_go_vom_hanabi_framework = float(np.sum(buffer.rewards[index:]))
            reward_vom_reward_shaping = reward_shape.get_sum()

            reward_shaping_array = np.empty(len(buffer.bayesian_actions) - index)
            reward_shaping_array.fill(reward_vom_reward_shaping)
            reward_vom_reward_shaping_to_go = reward_shaping_array.sum() if self.with_reward_shaping is True else 0

            reward_to_go = reward_to_go_vom_hanabi_framework
            discounted_reward_to_go = (reward_to_go +reward_vom_reward_shaping_to_go) * np.power(self.gamma, index + 1)

            reward_vom_hanabi_framework = buffer.rewards[index]

            observation = buffer.observation[index]
            bayesian_actions = buffer.bayesian_actions[index]

            result.append(bayesian_actions.sampled_action, discounted_reward_to_go, reward_vom_hanabi_framework, observation)

    def execute(self,collected_batch_results: CollectBatchResults, rewardshape_setting: RewardShapeSetting) -> RewardsToGoCalculationResult:
        """execute"""
        episodes_result = RewardsToGoCalculationResult(collected_batch_results.get_games_played())

        for batch_result in collected_batch_results.results:
            reward_calculation_result = RewardsCalculationResult()
            episodes_result.append(reward_calculation_result)

            buffer = batch_result.buffer
            self.calculate(buffer, reward_calculation_result, rewardshape_setting)

        return episodes_result
