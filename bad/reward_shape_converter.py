# pylint: disable=missing-module-docstring, wrong-import-position, too-few-public-methods, too-many-arguments, line-too-long

import sys
import os

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.reward_shape_result import RewardShapeResult
from bad.reward_shape import RewardShape
from bad.rewardshape_setting import RewardShapeSetting

class RewardShapeConverter:
    """reward shape converter"""
    def convert(self, reward_shape: RewardShape, rewardshape_setting: RewardShapeSetting) -> RewardShapeResult:
        """convert"""

        lost_one_life_token = rewardshape_setting.lost_one_life_token_true if reward_shape.lost_one_life_token is True else rewardshape_setting.lost_one_life_token_false
        lost_all_life_tokens: float = rewardshape_setting.lost_all_life_tokens_true if reward_shape.lost_all_life_tokens is True else rewardshape_setting.lost_all_life_tokens_false
        successfully_played_a_card: float = rewardshape_setting.successfully_played_a_card_true if reward_shape.successfully_played_a_card is True else rewardshape_setting.successfully_played_a_card_false

        discard: float = rewardshape_setting.discard_true if reward_shape.discard is True else rewardshape_setting.discard_false
        discard_playable: float = 0 if reward_shape.discard_playable is None else rewardshape_setting.discard_playable_true if reward_shape.discard_playable is True else rewardshape_setting.discard_playable_false
        discard_unique: float = 0 if reward_shape.discard_unique is None else rewardshape_setting.discard_unique_true if reward_shape.discard_unique is True else rewardshape_setting.discard_unique_false
        discard_useless: float = 0 if reward_shape.discard_useless is None else rewardshape_setting.discard_useless_true if reward_shape.discard_useless is True else rewardshape_setting.discard_useless_false

        hint: float = rewardshape_setting.hint_true if reward_shape.hint is True else rewardshape_setting.hint_false
        play: float = rewardshape_setting.play_true if reward_shape.play is True else rewardshape_setting.play_false

        return RewardShapeResult(lost_one_life_token= lost_one_life_token,
                                 lost_all_life_tokens= lost_all_life_tokens,
                                 successfully_played_a_card= successfully_played_a_card,
                                 discard= discard,
                                 discard_playable= discard_playable,
                                 discard_unique= discard_unique,
                                 discard_useless= discard_useless,
                                 hint= hint,
                                 play= play
                                 )
