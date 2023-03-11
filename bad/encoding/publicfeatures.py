# pylint: disable=missing-module-docstring, wrong-import-position, too-few-public-methods, too-many-instance-attributes
import sys
import os
import numpy as np

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.fireworkrank import FireworkRank
from bad.encoding.legal_actions import LegalActions
from bad.encoding.hands import Hands

class PublicFeatures:
    '''public features'''
    def __init__(self, observation: dict) -> None:
        '''init'''
        self.observation = observation
        self.curr_player = observation['current_player']

        self.life_tokens_left = self.convert_life_tokens()
        self.hint_tokens_left = self.convert_information_tokens()
        self.current_player = self.convert_current_player()
        self.last_action = self.convert_last_action()
        self.firework = FireworkRank(observation)
        self.legal_actions = LegalActions(observation)
        self.hand = Hands(observation)

    def convert_life_tokens(self) -> np.ndarray:
        ''' convert life tokens '''
        life_tokens = self.observation['player_observations'][self.curr_player]['life_tokens']
        return np.eye(16)[life_tokens]  

    def convert_information_tokens(self) -> np.ndarray:
        ''' convert information tokens '''
        in_to = self.observation['player_observations'][self.curr_player]['information_tokens']
        return np.eye(16)[in_to] 
        
    def convert_current_player(self) -> np.ndarray:
        '''converts current player'''
        return np.eye(2)[self.curr_player]

    def convert_last_action(self) -> np.ndarray:
        '''convert last action'''
        max_action: int = self.observation['max_action']
        last_action: int = self.observation['last_action']
        # print(last_action)
        return np.eye(max_action)[last_action]
