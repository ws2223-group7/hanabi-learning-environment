#pylint: disable=wrong-import-position, import-error, invalid-name, logging-fstring-interpolation, missing-module-docstring, broad-exception-raised

import logging
import re
import os
import sys

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from print_plot_log.helpfunction import read_epoch_number

class Logger:
    'Logger'

    def __init__(self, modelpath, level=logging.DEBUG):
        'Init Logger'
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
        self.modelpath = modelpath
        self.filename = 'log_reward_shaping.txt' \
            if modelpath == 'models_with_reward_shaping' else 'log_no_reward_shaping.txt'
        handler = logging.FileHandler(self.filename)
        handler.setLevel(level)

        self.logger.addHandler(handler)

    def log_reward(self, reward: float):
        'Log Reward'
        epoch_number = read_epoch_number(self.modelpath)
        self.logger.log(
            logging.DEBUG, f'Rewards in Epoch {epoch_number}: {reward}')

    def get_reward_for_epoch(self, episode):
        'Get Reward for Epoch'
        with open(self.filename, encoding="utf-8") as file:
            for line in file:
                match = re.match(r'^Rewards in Epoch (\d+): ([\d\.]+)$', line)
                if match:
                    epoch = int(match.group(1))
                    reward = float(match.group(2))
                    if epoch == episode:
                        return reward
        raise Exception('no reward for epoch')
    def get_all_rewards(self):
        'Get All Rewards'
        rewards = []
        with open(self.filename, encoding="utf-8") as file:
            for line in file:
                match = re.match(r'^Rewards in Epoch \d+: ([\d\.]+)$', line)
                if match:
                    reward = float(match.group(1))
                    rewards.append(reward)
        return rewards


def main():
    'Main'
    logger = Logger('models_with_reward_shaping')
    all_rewards = logger.get_all_rewards()
    logger.log_reward(0)
    print(all_rewards)


if __name__ == "__main__":
    main()
