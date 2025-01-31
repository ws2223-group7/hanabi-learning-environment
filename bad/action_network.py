# pylint: disable=missing-module-docstring, wrong-import-position, unused-variable, unused-argument, not-callable, invalid-name, fixme, unreachable, line-too-long, consider-using-enumerate, too-many-locals
import sys
import os
import numpy as np
import tensorflow as tf

currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

from bad.encoding.observation import Observation
from bad.bayesian_action import BayesianAction
from bad.action_provider import ActionProvider
from bad.baseline import Baseline


class ActionNetwork(ActionProvider):
    ''' action network '''

    def __init__(self, basepath, learning_rate) -> None:
        self.model = None
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.basepath = basepath

    def exists(self) -> bool:
        """exists"""
        return os.path.exists(self.basepath)

    def load(self) -> None:
        """load"""
        path = self.get_current_model_path()

        self.model = tf.keras.models.load_model(path)

    def get_current_model_path(self):
        """get current model path"""
        return self.get_model_path(0)

    def get_next_model_path(self):
        """get next model path"""
        return self.get_model_path(1)

    def get_model_path(self, inkrement: int):
        """gets next model path"""
        if not self.exists():
            return f"{self.basepath}/01"

        next_path = self.get_last_epoch_number()+inkrement
        return f"{self.basepath}/{next_path:02d}"

    def get_last_epoch_number(self) -> int:
        """get last epoch number"""
        if not self.exists():
            return 0

        networks_paths = os.listdir(self.basepath)
        networks_path_as_int = [int(networks_path) for networks_path in networks_paths]
        return int(np.max(networks_path_as_int))

    def save(self) -> None:
        """save"""
        self.model.save(self.get_next_model_path())

    def build(self, observation: Observation, max_action: int, \
              public_belief = None) -> None:
        '''build'''
        if self.model is None:
            shape = observation.to_one_hot_vec().shape

            self.model = tf.keras.Sequential([
                tf.keras.Input(shape=shape, name="input"),
                tf.keras.layers.Dense(384, activation="relu", name="layer1"),
                tf.keras.layers.Dense(384, activation="relu", name="layer2"),
                tf.keras.layers.Dense(max_action, activation='softmax', name='Output_Layer')
            ])
            self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer)

    def print_summary(self) -> None:
        '''print summary'''
        self.model.summary()

    def get_model_input(self, observation: Observation, publicBelief=None):
        '''get model input'''
        network_input = observation.to_one_hot_vec()

        # Input muss noch angepasst werden
        # network_input = publicBelief.to_one_hot_vec() + observation.to_one_hot_vec()

        reshaped = tf.reshape(network_input, [1, network_input.shape[0]])
        return reshaped

    def get_action(self, observation: Observation, legal_moves_as_int: list, \
                   max_moves: int, \
                   public_belief = None) -> BayesianAction:
        '''get action'''
        result = self.model(self.get_model_input(observation, public_belief))
        result_list = result.numpy()[0].tolist()
        result_filtered = [elem_in_res if (elem_idx in legal_moves_as_int) else 0
                           for elem_idx, elem_in_res in enumerate(result_list)]

        result_filtered_filtered = result_filtered[:max_moves]
        return BayesianAction(np.array(result_filtered_filtered))

    def backpropagation(self, observation, actions: np.ndarray, rewards_to_go: np.ndarray, baseline: Baseline) -> float:
        """train step"""
        model = self.model
        batch_size = len(observation)
        observation_length = len(observation[0])
        observation_tensor = tf.reshape(observation, [batch_size, observation_length])
        rewards_to_go_tensor = tf.convert_to_tensor(rewards_to_go, dtype=float)

        with tf.GradientTape() as tape:
            logits = model(observation_tensor)
            log_logprobs = tf.nn.log_softmax(logits)

            row_indices= tf.range(len(actions))
            indices = tf.transpose([row_indices, actions])
            logprob = tf.gather_nd(log_logprobs, indices)

            loss = -(tf.reduce_mean(logprob * rewards_to_go_tensor))
            print(f'current loss: {loss.numpy()}')

        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss.numpy()
