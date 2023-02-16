import sys
import os

from public_belief_player import PublicBeliefPlayer
from hint_matrix_global import HintMatrix
from ftpubvec import RemaingCards
from likelihood_global import Likelihood
from build_hanabi_env import get_hanabi_env

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
parentPath2 = os.path.dirname(parentPath)
sys.path.append(parentPath2)


from hanabi_learning_environment.rl_env import HanabiEnv
from constants import Constants
from encoding.observation import Observation


class PublicBelief(list):
    
    def __init__(self, hanabi_env: HanabiEnv, constants: Constants, 
                 action_network, pre_observation_enc: Observation= None,  last_action=None,
                 pre_hanabi_env: HanabiEnv=None, public_belief= None):
        """Initialize / Update the PublicBelief
        - create 
            RemaningCard (FtpubVec in Paper)
            HintMatrix
            likelihoof
        
        Creating the publicBelief the first time the parameter
            observation, last_action, pre_hanabi_env and public_belief are None!!!

        Args:
            constants (HanabiEnv): Hanabi environment 
            act_network (ActionNetwork): Network that predicts the last_action
            last_act (_type_, None): the last action taken by the agent
            pre_observation (Observation): Observation from the previous step
            pub_belief (Likelihood, None): current public_belief
            pre_hint_matrix (HintMatrix, None): Hint matrix from the previous step
        Returns:
            likelihood (Likelihood): Likelihood of the current step
        """


        # Init rem_cards,hint_matrix and hanabi_env due to dubug purpose 
        self.rem_cards = RemaingCards(hanabi_env)
        self.hint_matrix = HintMatrix(constants, self.rem_cards)
        self.likelehood = Likelihood(constants, action_network, pre_hanabi_env, 
                                     pre_observation_enc, last_action, public_belief)
        
        # Init hanabi_env due to dubug purpose
        self.hanabi_env = hanabi_env
        
        super().__init__(self.init(constants, self.rem_cards,
                         self.hint_matrix, self.likelihood))
        
    def init(self, constants, rem_cards,
                    hint_matrix, likelihood):

        pub_belf = [PublicBeliefPlayer(constants, idx_ply, rem_cards,
                    hint_matrix[idx_ply], likelihood[idx_ply]) 
                    for idx_ply in range(constants.num_ply)]
        return pub_belf 


if __name__ == "__main__":
    hanabi_env = get_hanabi_env()
    pub_belf = PublicBelief(hanabi_env)
    print()        
            
