# pylint: disable=missing-module-docstring, wrong-import-position, import-error, no-member, no-name-in-module, too-many-function-args, unnecessary-pass, too-few-public-methods
import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.run_episode_result import RunEpisodeResult
from hanabi_learning_environment.rl_env import HanabiEnv


class PrintEpisodeSelfPlay:
    def __init__(self, run_episode_result: RunEpisodeResult) -> None:
        '''init'''
        self.run_episode_result = run_episode_result

    def print(self) -> None:
        '''print'''
        print(f"episode {self.run_episode_result.number} result:")
        print(f"episode reward: {self.run_episode_result.reward}")
