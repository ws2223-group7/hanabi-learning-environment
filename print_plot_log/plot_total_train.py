import os
import sys

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from print_plot_log.Logger import Logger
import matplotlib.pyplot as plt

class PlotTraining:
    def plot(self, reward_shaping: bool) -> None:
        self.plot_reward(reward_shaping)

    def plot_reward(self, reward_shaping: bool) -> None:
        
        if reward_shaping:
            logger = Logger('models_with_reward_shaping')
        else:
            logger = Logger('models_without_reward_shaping')

        results = logger.get_all_rewards()
        x_axis = [x for x in range(len(results))]

        plt.ylabel('Rewards  pro Epoche')
        plt.xlabel('Epoche')
        plt.title('Rewards during Training')
        plt.plot(x_axis, results)
        plt.savefig('diagramms/Train.png')
    
    def plot_reward_shaping_vs_no_reward_shaping():
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
    plot_training = PlotTraining()
    plot_training.plot_reward()


if __name__ == "__main__":
    main()
