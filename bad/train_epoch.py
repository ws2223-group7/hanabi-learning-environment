# pylint: disable=missing-module-docstring, wrong-import-position, ungrouped-imports, too-few-public-methods, consider-using-enumerate, line-too-long, line-too-long

import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.action_network import ActionNetwork
from bad.train_batches import TrainBatches
from bad.train_epoch_result import TrainEpochResult
class TrainEpoch:
    """train epoch"""
    def __init__(self, network: ActionNetwork) -> None:
        """init"""
        self.network = network

    def train(self, batch_size: int, gamma: float) -> TrainEpochResult:
        """train"""
        train_batches = TrainBatches(self.network)
        result = train_batches.run(batch_size=batch_size, gamma=gamma)
        return TrainEpochResult(result.loss)
