# pylint: disable=missing-module-docstring, wrong-import-position, ungrouped-imports, too-few-public-methods, consider-using-enumerate, line-too-long, line-too-long


import os
import sys

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.action_network import ActionNetwork
from bad.collect_episode_data import CollectEpisodeData
from bad.collect_episode_data_result import CollectEpisodeDataResult
from bad.collect_episodes_data_results import CollectEpisodesDataResults
from bad.constants import Constants
from bad.encoding.observationconverter import ObservationConverter
from bad.set_extra_observation import SetExtraObservation
from hanabi_learning_environment import pyhanabi, rl_env


class CollectData:
    """collect data"""
    def __init__(self, network: ActionNetwork, players: int) -> None:
        """init"""
        self.network = network
        constants = Constants()
        self.hanabi_environment = rl_env.make(constants.environment_name, players, pyhanabi.AgentObservationType.SEER)

    async def execute_async(self, batch_size: int, epoch: int) -> CollectEpisodesDataResults:
        """execute"""
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

            print(f"epoch: {epoch} collected episoden aktionen: {collect_batch_episodes_result.get_batch_size()} von batch size {batch_size}")

        return collect_batch_episodes_result
