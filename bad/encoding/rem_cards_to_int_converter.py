# pylint: disable=missing-module-docstring, wrong-import-position, too-few-public-methods

import numpy as np

class NumRemCardsToIntConverter:
    ''' rank converter'''
    def convert(self, rem_cards:int) -> np.ndarray:
        '''convert'''
        return np.eye(6)[rem_cards]
