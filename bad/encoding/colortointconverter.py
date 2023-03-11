# pylint: disable=missing-module-docstring, wrong-import-position, too-few-public-methods

import string
import numpy as np


class ColorToIntConverter:
    '''color to int converter'''

    def __init__(self) -> None:
        '''init'''
        self.colors = {'None': 0, 'R': 1, 'Y': 2, 'G': 3, 'W': 4, 'B': 5}

    def convert(self, col: string) -> np.ndarray:
        '''convert'''
        return np.eye(6)[self.convert_color_to_int(col)] 

    def convert_color_to_int(self, col: string) -> int:
        '''convert color to int'''
        return self.colors[col]
