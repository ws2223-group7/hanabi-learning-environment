# pylint: disable=missing-module-docstring, wrong-import-position, import-error
import unittest
import sys
import os
import numpy as np
from bad.encoding.fireworkrank import FireworkRank

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.publicfeatures import PublicFeatures

class ObservationConverterEncodingTest(unittest.TestCase):
    ''' observation converter '''
    def test_convert_public_features_haseverything(self):
        ''' test convert everything '''
        observation = {}
        observation['life_tokens'] = 3
        observation['information_tokens'] = 8

        observation['fireworks'] = {}
        observation['fireworks']['B'] = 5
        observation['fireworks']['G'] = 5
        observation['fireworks']['R'] = 5
        observation['fireworks']['W'] = 5
        observation['fireworks']['Y'] = 5

        observation['current_player'] = 1

        expected_lt_left = np.zeros(shape=4, dtype=int)
        expected_lt_left[3] = 1

        expected_ht_left = np.zeros(shape=9, dtype=int)
        expected_ht_left[8] = 1

        expected_fw_red = np.zeros(shape=6, dtype=int)
        expected_fw_red[5] = 1

        expected_fw_green = np.zeros(shape=6, dtype=int)
        expected_fw_green[5] = 1

        expected_fw_blue = np.zeros(shape=6, dtype=int)
        expected_fw_blue[5] = 1

        expected_fw_white = np.zeros(shape=6, dtype=int)
        expected_fw_white[5] = 1

        expected_fw_yellow = np.zeros(shape=6, dtype=int)
        expected_fw_yellow[5] = 1

        expected_cur_player = np.zeros(shape=2, dtype=int)
        expected_cur_player[1] = 1

        public_features = PublicFeatures(observation)

        self.assertTrue(np.array_equal(public_features.life_tokens_left, \
            expected_lt_left))
        self.assertTrue(np.array_equal(public_features.hint_tokens_left, \
            expected_ht_left))

        self.assertTrue(np.array_equal(public_features.firework.red, \
            expected_fw_red))
        self.assertTrue(np.array_equal(public_features.firework.green, \
            expected_fw_green))
        self.assertTrue(np.array_equal(public_features.firework.blue, \
            expected_fw_blue))
        self.assertTrue(np.array_equal(public_features.firework.white, \
            expected_fw_white))
        self.assertTrue(np.array_equal(public_features.firework.yellow, \
            expected_fw_yellow))

        self.assertTrue(np.array_equal(public_features.current_player, expected_cur_player))

    def test_test_convert_public_features_hasnothing(self):
        ''' convert has nothing '''
        observation = {}
        observation['life_tokens'] = 0
        observation['information_tokens'] = 0

        observation['fireworks'] = {}
        observation['fireworks']['B'] = 0
        observation['fireworks']['G'] = 0
        observation['fireworks']['R'] = 0
        observation['fireworks']['W'] = 0
        observation['fireworks']['Y'] = 0

        observation['current_player'] = 0

        expected_lt_left = np.zeros(shape=4, dtype=int)
        expected_lt_left[0] = 1

        expected_ht_left = np.zeros(shape=9, dtype=int)
        expected_ht_left[0] = 1

        expectedfireworkrankred = np.zeros(shape=6, dtype=int)
        expectedfireworkrankred[0] = 1

        expectedfireworkrankgreen = np.zeros(shape=6, dtype=int)
        expectedfireworkrankgreen[0] = 1

        expectedfireworkrankblue = np.zeros(shape=6, dtype=int)
        expectedfireworkrankblue[0] = 1

        expectedfireworkrankwhite = np.zeros(shape=6, dtype=int)
        expectedfireworkrankwhite[0] = 1

        expectedfireworkrankyellow = np.zeros(shape=6, dtype=int)
        expectedfireworkrankyellow[0] = 1

        expected_cur_player = np.zeros(shape=2, dtype=int)
        expected_cur_player[0] = 1

        public_features = PublicFeatures(observation)

        self.assertTrue(np.array_equal(public_features.life_tokens_left, \
            expected_lt_left))
        self.assertTrue(np.array_equal(public_features.hint_tokens_left, \
            expected_ht_left))

        self.assertTrue(np.array_equal(public_features.firework.red, \
            expectedfireworkrankred))
        self.assertTrue(np.array_equal(public_features.firework.green, \
            expectedfireworkrankgreen))
        self.assertTrue(np.array_equal(public_features.firework.blue, \
            expectedfireworkrankblue))
        self.assertTrue(np.array_equal(public_features.firework.white, \
            expectedfireworkrankwhite))
        self.assertTrue(np.array_equal(public_features.firework.yellow, \
            expectedfireworkrankyellow))

        self.assertTrue(np.array_equal(public_features.current_player, expected_cur_player))

    def test_differentfireworks(self):
        ''' test different fireworks '''
        observation = {}
        observation['life_tokens'] = 3
        observation['information_tokens'] = 8

        observation['fireworks'] = {}
        observation['fireworks']['B'] = 1
        observation['fireworks']['G'] = 2
        observation['fireworks']['R'] = 3
        observation['fireworks']['W'] = 4
        observation['fireworks']['Y'] = 5

        observation['current_player'] = 1

        expectedfireworkrankblue = np.zeros(shape=6, dtype=int)
        expectedfireworkrankblue[1] = 1

        expected_fw_green = np.zeros(shape=6, dtype=int)
        expected_fw_green[2] = 1

        expected_fw_red = np.zeros(shape=6, dtype=int)
        expected_fw_red[3] = 1

        expectedfireworkrankwhite = np.zeros(shape=6, dtype=int)
        expectedfireworkrankwhite[4] = 1

        expectedfireworkrankyellow = np.zeros(shape=6, dtype=int)
        expectedfireworkrankyellow[5] = 1

        firework = FireworkRank(observation)

        self.assertTrue(np.array_equal(firework.red, expected_fw_red))
        self.assertTrue(np.array_equal(firework.green, expected_fw_green))
        self.assertTrue(np.array_equal(firework.blue, expectedfireworkrankblue))
        self.assertTrue(np.array_equal(firework.white, expectedfireworkrankwhite))
        self.assertTrue(np.array_equal(firework.yellow, expectedfireworkrankyellow))

if __name__ == '__main__':
    unittest.main()
