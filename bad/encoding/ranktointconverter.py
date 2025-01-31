# pylint: disable=missing-module-docstring, wrong-import-position, too-few-public-methods
import numpy as np
import tensorflow as tf


class RankToIntConverter:
    ''' rank converter'''
    def convert(self, rank:int) -> np.ndarray:
        '''convert'''
        return tf.keras.utils.to_categorical(rank, num_classes=6, dtype=int)
