# pylint: disable=missing-module-docstring, wrong-import-position, unused-variable, unused-argument, not-callable, invalid-name, fixme, unreachable, line-too-long, consider-using-enumerate, too-many-locals
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.observation import Observation
from bad.bayesian_action import BayesianAction
from bad.action_provider import ActionProvider
from bad.baseline import Baseline


class ActionNetwork(ActionProvider):
    ''' action network '''

    def __init__(self, path) -> None:
        self.model = None
        self.path = path
        self.optimizer_policy_net = None

    # def load(self, ) -> None:
    #     """load"""
    #     self.model = tf.keras.models.load_model(self.path)

    # def save(self):
    #     """save"""
    #     self.model.save(self.path)

    def build(self, observation: Observation, max_action: int, \
              public_belief = None) -> None:
        '''build'''
        observation = len(observation.to_one_hot_vec())
        if self.model is None:
            layers = []
            layers = [nn.Linear(observation, 384), nn.ReLU()]
            layers += [nn.Linear(384, 384), nn.ReLU()]
            layers += [nn.Linear(384, max_action), nn.Softmax()]
            self.model = nn.Sequential(*layers)
            self.optimizer_policy_net = Adam(self.model.parameters(), lr=0.001)

    # def print_summary(self):
    #     '''print summary'''
    #     self.model.summary()

    # def print_summary(self):
    #     '''print summary'''
    #     self.model.summary()

    def get_model_input(self, observation: Observation, publicBelief=None):
        '''get model input'''
        network_input = torch.as_tensor(observation.to_one_hot_vec(), dtype=torch.float32)       

        # Input muss noch angepasst werden
        # network_input = publicBelief.to_one_hot_vec() + observation.to_one_hot_vec()

        return network_input

    def get_action(self, observation: Observation, legal_moves_as_int: list, \
                   public_belief = None) -> BayesianAction:
        '''get action'''
        result = self.model(self.get_model_input(observation, public_belief))
        result_list = result.detach().numpy().tolist()
        result_filtered = [elem_in_res if (elem_idx in legal_moves_as_int) else 0
                           for elem_idx, elem_in_res in enumerate(result_list)]
        result_filtered_sliced = result_filtered[:20]
        return BayesianAction(np.array(result_filtered_sliced))

    
    def calculate_loss_policy(self, observation, action: np.ndarray, rewards_to_go: np.ndarray)-> float:
        """Calculate loss policy"""
        action = torch.as_tensor(action, dtype=torch.float32)
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
        rewards_to_go_tensor = torch.as_tensor(rewards_to_go, dtype=torch.float32)

        log_probs = self.get_policy(observation_tensor).log_prob(action)
        return -(log_probs * rewards_to_go_tensor).mean()


    def get_policy(self, observation)-> float:
        """get policy"""
        model = self.model
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
        logits = model(observation_tensor)
        return Categorical(logits=logits)
    
    
    def backpropagation(self, observation, actions: np.ndarray, rewards_to_go: np.ndarray, baseline: Baseline) -> float:
        """train step"""
        self.optimizer_policy_net.zero_grad()
        batch_loss_policy = self.calculate_loss_policy(observation, actions, rewards_to_go)
        batch_loss_policy.backward()
        self.optimizer_policy_net.step()
        return batch_loss_policy.detach().numpy()
