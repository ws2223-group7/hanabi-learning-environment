# pylint: disable=missing-module-docstring, wrong-import-position, ungrouped-imports, too-few-public-methods, consider-using-enumerate, line-too-long, line-too-long


import os
import sys

import numpy as np

from bad.action_network import ActionNetwork
from bad.rewards_to_go_calculation_result import RewardsToGoCalculationResult

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

class BackPropagation:
    """backpropagation"""
    def __init__(self, network:ActionNetwork) -> None:
        self.network = network

    def execute(self, calc_result: RewardsToGoCalculationResult) -> float:
        """execute"""
        actions = np.empty(0, int)
        logprob = np.empty(0, float)
        rewards_to_go = np.empty(0, int)
        observation_array_array = []
        baseline =  calc_result.get_baseline()

        for episode_result in calc_result.results:
            for action_index in range(len(episode_result.observation)):
                current_observation = episode_result.observation[action_index].to_array()
                observation_array_array.append(current_observation)
                actions = np.append(actions, episode_result.actions[action_index])
                logprob = np.append(logprob, episode_result.logprob[action_index])
                rewards_to_go = np.append(rewards_to_go, episode_result.rewards_to_go[action_index])

        return self.network.backpropagation(observation_array_array, actions, logprob, rewards_to_go, baseline)
