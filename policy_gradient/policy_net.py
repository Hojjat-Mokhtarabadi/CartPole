from typing import Tuple

from tensorflow.keras.layers import Dense
import tensorflow as tf


class ReinforcePolicyNet(tf.keras.Model):

    def __init__(self, action_count, states_count):
        self._action_count = action_count
        self._states_count = states_count

        super(ReinforcePolicyNet, self).__init__()
        self._dense1 = Dense(units=128, activation='relu', input_shape=(4,))
        # self._dense2 = Dense(units=128, activation='relu')
        self.actor = Dense(units=self._action_count, activation='linear')
        self.critic = Dense(1, activation='linear')

    def call(self, inputs: tf.Tensor, training=True, mask=None) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self._dense1(inputs)
        # x = self._dense2(x)
        # x = self._dense2(x)
        return self.actor(x), self.critic(x)

    def get_config(self):
        pass


class ReinforcePolicyModel:
    def __init__(self, net: ReinforcePolicyNet):
        self._net = net

    @staticmethod
    def policy(actions):
        return tf.nn.softmax(actions)

    def predict(self, state):
        actor_prediction, critic_prediction = self._net(state)
        return self.policy(actor_prediction), critic_prediction

    def choose_action(self, action_prob):
        action = tf.random.categorical(action_prob, 1)[0][0]
        return action
