#pylint: disable=missing-module-docstring, wrong-import-position, import-error, unnecessary-comprehension
import os
import sys

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

import matplotlib.pyplot as plt
from print_plot_log.Logger import Logger

class PlotTraining:
    'Plot Trainings Rewards'

    def plot(self, reward_shaping: bool) -> None:
        "Plot"
        self.plot_reward(reward_shaping)

    def plot_reward(self, reward_shaping: bool) -> None:
        """Plot the reward of the model with reward shaping if 
           reward_shaping is True, else plot the reward of the model 
           without reward shaping"""

        if reward_shaping:
            logger = Logger('models_with_reward_shaping')
        else:
            logger = Logger('models_without_reward_shaping')

        results = logger.get_all_rewards()
        x_axis = [x for x in range(len(results))]

        directory = 'diagramms'
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.ylabel('Rewards  pro Epoche')
        plt.xlabel('Epoche')
        plt.title('Rewards during Training')
        plt.plot(x_axis, results)
        plt.savefig(f'{directory}/Train.png')

    def plot_reward_shaping_vs_no_reward_shaping(self):
        """Plot the reward of the model with reward shaping 
           and the model without reward shaping"""

        logger_reward_shaping = Logger('models_with_reward_shaping')
        logger_no_reward_shaping = Logger('models_without_reward_shaping')

        results_reward_shaping = logger_reward_shaping.get_all_rewards()
        results_no_reward_shaping = logger_no_reward_shaping.get_all_rewards()

        x_axis = [x for x in range(len(results_reward_shaping))]

        plt.ylabel('Rewards pro Epoche')
        plt.xlabel('Epoche')
        plt.title('Rewards during Training')
        plt.plot(x_axis, results_reward_shaping, label='reward shaping')
        plt.plot(x_axis, results_no_reward_shaping, label='no reward shaping')
        plt.legend()
        plt.savefig('diagramms/Reward Shaping vs No Reward Shaping.png')


def main():
    'Main'
    reward_shaping = True
    plot_training = PlotTraining()
    plot_training.plot_reward(reward_shaping)


if __name__ == "__main__":
    main()
