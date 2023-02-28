# pylint: disable=missing-module-docstring, wrong-import-position, no-name-in-module, unused-variable, unused-variable, line-too-long

import asyncio
import os
import random
import sys
import numpy as np
import tensorflow as tf

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.self_play import SelfPlay
from bad.action_network import ActionNetwork
from bad.reward_to_go_calculation import RewardToGoCalculation
from bad.collect_data import CollectData
from bad.collect_episodes_data_results import CollectEpisodesDataResults
from bad.rewards_to_go_calculation_result import RewardsToGoCalculationResult
from bad.backpropagation import BackPropagation

async def main() -> None:
    '''main'''
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.keras.utils.set_random_seed(seed)  # sets seeds for base-python, numpy and tf
    tf.config.experimental.enable_op_determinism()

    batch_size: int = 100
    epoch_size: int = 100

    episodes_running: int = 100
    gamma: float = 1.0

    model_path = 'model'
    players: int = 2

    do_load_and_save_model: bool = False

    print(f'welcome to bad agent with tf version: {tf.__version__}')
    print(f'running {episodes_running} episodes')

    network: ActionNetwork = ActionNetwork(model_path)

    if do_load_and_save_model && os.path.exists(model_path):
        network.load()

    collected_results: list[CollectEpisodesDataResults] = []
    tasks = []

    for epoch in range(epoch_size):
        print('')
        print(f'start running epoch: {epoch}')
        collect_data = CollectData(network, players)
        tasks.append(asyncio.create_task(collect_data.execute_async(batch_size, epoch)))

    await asyncio.gather(*tasks)

    print('finished all running epochs')

    for task in tasks:
        collected_results.append(task.result())

    print('start calculating results')

    calculation_results: list[RewardsToGoCalculationResult] = []
    tasks = []
    for collected_data in collected_results:
        reward_to_go_calculation = RewardToGoCalculation(gamma)
        tasks.append(asyncio.create_task(reward_to_go_calculation.execute_async(collected_data)))

    await asyncio.gather(*tasks)

    print('finished calculating results')

    for task in tasks:
        calculation_results.append(task.result())

    print('running backpropagation')

    for calculation_result in calculation_results:
        backpropagation = BackPropagation(network)
        backpropagation.execute(calculation_result)
        print(f"reward: {calculation_result.get_reward_sum() / batch_size}")

    if do_load_and_save_model:
        network.save()

    #self_play = SelfPlay(network)
    #self_play.run(episodes_running)

    print("finish with everything")
if __name__ == "__main__":
    asyncio.run(main())
