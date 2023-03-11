# pylint: disable=missing-module-docstring, wrong-import-position, too-few-public-methods
import numpy as np

class RankToIntConverter:
    ''' rank converter'''
    def convert(self, rank:int) -> np.ndarray:
        '''convert'''
        return np.eye(6)[rank]
