# pylint: disable=missing-module-docstring, wrong-import-position, ungrouped-imports, too-few-public-methods, consider-using-enumerate, line-too-long, line-too-long

import sys
import os

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

class TrainBatches:
    '''train batch'''
    def __init__(self, network: ActionNetwork) -> None:
        '''init'''
        self.network = network

    def collect_data(self, batch_size:int, players: int) -> CollectEpisodesDataResults:
        '''collect data'''
        collect_batch_episodes_result = CollectEpisodesDataResults()
        constants = Constants()
        hanabi_environment = rl_env.make(constants.environment_name, players, \
        pyhanabi.AgentObservationType.SEER)

        while collect_batch_episodes_result.get_n() < batch_size:

            hanabi_observation = hanabi_environment.reset()
            max_moves: int = hanabi_environment.game.max_moves() + 1
            max_actions = max_moves + 1 # 0 index based

            seo = SetExtraObservation()
            seo.set_extra_observation(hanabi_observation, max_moves, max_actions, \
                hanabi_environment.state.legal_moves_int())
            observation_converter: ObservationConverter = ObservationConverter()
            self.network.build(observation_converter.convert(hanabi_observation), max_actions)

            ce_data = CollectEpisodeData(hanabi_observation, hanabi_environment, self.network)
            episode_data_result: CollectEpisodeDataResult = ce_data.collect() # hier werden die Daten für eine episode gesammelt

            collect_batch_episodes_result.add(episode_data_result)

            print(f"collected episoden aktionen: {collect_batch_episodes_result.get_n()} von batch size {batch_size}")

        return collect_batch_episodes_result

    def reward_to_go_calculation(self, collected_data: CollectEpisodesDataResults, \
        gamma: float) -> RewardsToGoCalculationResult:
        '''reward to go calculation'''
        reward_to_go_calculation = RewardToGoCalculation(gamma)
        return reward_to_go_calculation.run(collected_data)

    def backpropagation(self, calc_result: RewardsToGoCalculationResult) -> None:
        '''backpropagation'''

        for calc_result_result in calc_result.results:
            for index in range(len(calc_result_result.observation)):
                self.network.backpropagation(calc_result_result.observation[index], calc_result_result.rewards[index], calc_result_result.losses[index])

    def run(self, batch_size: int, gamma: float) -> TrainBatchResult:
        '''init'''
        print('training')
        players:int = 2

        print('collecting data')
        collected_data = self.collect_data(batch_size, players)

        print('reward calculation')
        reward_calculation_result = self.reward_to_go_calculation(collected_data, gamma)

        print('backpropagation')
        self.backpropagation(reward_calculation_result)

        return TrainBatchResult()
