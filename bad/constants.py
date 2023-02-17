# pylint: disable=missing-module-docstring, wrong-import-position, no-name-in-module, too-few-public-methods

class Constants:
    '''constants'''

    def __init__(self) -> None:
        '''init'''
        self.hanabi_env = None
        self.num_ply = None
        self.num_ranks = None
        self.hand_size = None
        self.num_cards_per_rank = None
        self.num_colors = None
        self.colors = None
        self.environment_name = 'Hanabi-Group-7'

    def update(self, hanabi_env) -> None:
        '''update'''
        # rank infos needs to update munally
        all_colors = ['R', 'Y', 'G', 'W', 'B']
        self.hanabi_env = hanabi_env
        self.num_ply = hanabi_env.game.num_players()
        self.hand_size = hanabi_env.game.hand_size()
        self.num_ranks = hanabi_env.game.num_ranks()
        self.num_colors = hanabi_env.game.num_colors()
        self.colors = [all_colors[i] for i in range(self.num_colors)]
        self.num_cards_per_rank = [hanabi_env.game.num_cards('R', rank)
                                   for rank in range(self.num_ranks + 1)]

    def action_int_to_move_type(self, action: int):
        return self.hanabi_env.game.get_move(action)
