# pylint: disable=missing-module-docstring too-few-public-methods, pointless-string-statement,wrong-import-position, fixme, broad-exception-raised
import sys
import os

import numpy as np
import torch
from torch.distributions.categorical import Categorical

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.bayesian_action_result import BayesianActionResult


class BayesianAction:
    '''Bayesian Action'''
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = torch.as_tensor(actions, dtype=torch.float32) 

        if len(actions) == 0:
            print()
            raise Exception('no actions')

    def sample_action(self) -> BayesianActionResult:
        '''returns a choice'''
        all_action_probs_distribution = Categorical(probs=self.actions)
        sampled_action:int = int(all_action_probs_distribution.sample().numpy())
        return BayesianActionResult(sampled_action)

