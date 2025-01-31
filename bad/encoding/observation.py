# pylint: disable=missing-module-docstring, wrong-import-position, too-few-public-methods
import sys
import os
import numpy as np

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
parentPath2 = os.path.dirname(parentPath)
sys.path.append(parentPath2)

from bad.encoding.publicfeatures import PublicFeatures
from bad.encoding.privatefeatures import PrivateFeatures

class Observation:
    '''observation'''
    def __init__(self, observation: dict):
        self.public_features = PublicFeatures(observation)
        self.private_features = PrivateFeatures(observation)

    def to_one_hot_vec(self) -> list:
        '''one single vecor'''
        result = np.concatenate(( \
            self.public_features.life_tokens_left, \
            self.public_features.hint_tokens_left, \
            self.public_features.firework.red, \
            self.public_features.firework.yellow, \
            self.public_features.firework.green, \
            self.public_features.firework.white, \
            self.public_features.firework.blue, \
            self.public_features.last_action, \
            self.public_features.legal_actions.vector, \
            self.public_features.current_player, \
            self.public_features.hand.own_cards,
            self.private_features.hands.other_cards
            ))
        return result

    def public_features_to_one_hot_vec(self) -> list:
        '''one single vecor'''
        result = np.concatenate(( \
            self.public_features.life_tokens_left, \
            self.public_features.hint_tokens_left, \
            self.public_features.firework.red, \
            self.public_features.firework.yellow, \
            self.public_features.firework.green, \
            self.public_features.firework.white, \
            self.public_features.firework.blue, \
            self.public_features.last_action, \
            self.public_features.legal_actions.vector, \
            self.public_features.current_player
            ))
        return result
