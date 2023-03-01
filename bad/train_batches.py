# pylint: disable=missing-module-docstring, wrong-import-position, ungrouped-imports, too-few-public-methods, consider-using-enumerate, line-too-long, line-too-long

import sys
import os
import numpy as np

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.collect_episode_data import CollectEpisodeData
from bad.collect_episode_data_result import CollectEpisodeDataResult
from bad.collect_episodes_data_results import CollectEpisodesDataResults
from bad.constants import Constants
from hanabi_learning_environment import pyhanabi, rl_env
from bad.action_network import ActionNetwork
from bad.encoding.observationconverter import ObservationConverter
from bad.set_extra_observation import SetExtraObservation
from bad.reward_to_go_calculation import RewardToGoCalculation
from bad.train_batch_result import TrainBatchResult
from bad.rewards_to_go_calculation_result import RewardsToGoCalculationResult
import gc

class TrainBatches:
    """train batch"""
    def __init__(self, network: ActionNetwork, hanabi_environment: rl_env.HanabiEnv, players: int) -> None:
        """init"""
        self.network = network
        self.hanabi_environment = hanabi_environment
        self.players = players

    def collect_data(self, batch_size:int) -> CollectEpisodesDataResults:
        """collect data"""
        collect_batch_episodes_result = CollectEpisodesDataResults()

        while collect_batch_episodes_result.get_batch_size() < batch_size:

            hanabi_observation = self.hanabi_environment.reset()
            max_moves: int = self.hanabi_environment.game.max_moves() + 1
            max_actions = max_moves + 1 # 0 index based

            seo = SetExtraObservation()
            seo.set_extra_observation(hanabi_observation, max_moves, max_actions, \
                self.hanabi_environment.state.legal_moves_int())
            observation_converter: ObservationConverter = ObservationConverter()
            self.network.build(observation_converter.convert(hanabi_observation), max_actions)

            ce_data = CollectEpisodeData(hanabi_observation, self.hanabi_environment, self.network)
            episode_data_result: CollectEpisodeDataResult = ce_data.collect() # hier werden die Daten für eine episode gesammelt

            collect_batch_episodes_result.add(episode_data_result)

            print(f"collected episoden aktionen: {collect_batch_episodes_result.get_batch_size()} von batch size {batch_size}")

        gc.collect()
        return collect_batch_episodes_result

    def calculation(self, collected_data: CollectEpisodesDataResults, \
        gamma: float) -> RewardsToGoCalculationResult:
        '''reward to go calculation'''
        reward_to_go_calculation = RewardToGoCalculation(gamma)
        return reward_to_go_calculation.run(collected_data)

    def backpropagation(self, calc_result: RewardsToGoCalculationResult) -> float:
        '''backpropagation'''
        actions = np.empty(0, int)
        logprob = np.empty(0, float)
        rewards_to_go = np.empty(0, int)
        observation_array_array = []
        baseline =  calc_result.get_baseline()

        for episode_result in calc_result.results:
            for action_index in range(len(episode_result.observation)):
                current_observation = episode_result.observation[action_index].to_one_hot_vec()
                observation_array_array.append(current_observation)
                actions = np.append(actions, episode_result.actions[action_index])
                logprob = np.append(logprob, episode_result.logprob[action_index])
                rewards_to_go = np.append(rewards_to_go, episode_result.rewards_to_go[action_index])

        return self.network.backpropagation(observation_array_array, actions, logprob, rewards_to_go, baseline)

    def run(self, batch_size: int, gamma: float) -> TrainBatchResult:
        '''init'''
        print('training')

        print('collecting data')
        collected_data = self.collect_data(batch_size)

        print('reward calculation')
        calculation_result = self.calculation(collected_data, gamma)

        print('backpropagation')
        loss = self.backpropagation(calculation_result)

        return TrainBatchResult(loss, calculation_result.get_reward_sum())
