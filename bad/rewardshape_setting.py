# pylint: disable=missing-module-docstring, wrong-import-position, no-name-in-module, unused-variable, unused-variable, line-too-long, ungrouped-imports, too-many-locals, invalid-name, too-many-instance-attributes, too-many-arguments, too-few-public-methods

class RewardShapeSetting:
    """reward shape setting"""
    def __init__(self, lost_one_life_token_true: float, lost_one_life_token_false: float,
                 lost_all_life_tokens_true:float, lost_all_life_tokens_false:float,
                 successfully_played_a_card_true:float, successfully_played_a_card_false: float,
                 discard_true: float, discard_false:float,
                 discard_playable_true: float, discard_playable_false: float,
                 discard_unique_true: float, discard_unique_false: float,
                 discard_useless_true: float, discard_useless_false: float,
                 hint_true: float, hint_false: float,
                 play_true:float, play_false: float
                    ) -> None:
        self.lost_one_life_token_true = lost_one_life_token_true
        self.lost_one_life_token_false = lost_one_life_token_false
        self.lost_all_life_tokens_true = lost_all_life_tokens_true
        self.lost_all_life_tokens_false = lost_all_life_tokens_false
        self.successfully_played_a_card_true = successfully_played_a_card_true
        self.successfully_played_a_card_false = successfully_played_a_card_false
        self.discard_true = discard_true
        self.discard_false = discard_false
        self.discard_playable_true = discard_playable_true
        self.discard_playable_false = discard_playable_false
        self.discard_unique_true = discard_unique_true
        self.discard_unique_false = discard_unique_false
        self.discard_useless_true = discard_useless_true
        self.discard_useless_false = discard_useless_false
        self.hint_true = hint_true
        self.hint_false = hint_false
        self.play_true = play_true
        self.play_false = play_false
