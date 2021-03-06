from typing import Tuple, List
import tensorflow as tf
import gym
import numpy as np


class Environment:
    def __init__(self, env=''):
        self.env = gym.make(env)

    @staticmethod
    def preprocess_state(state):
        return np.reshape(state, (1, 4))

    def spaces_count(self):
        return self.env.action_space.n, self.env.observation_space.shape[0]

    def reset_env(self):
        return self.env.reset()

    '''
        tf.numpy function allows us to pass a python function to computational graph as a Tensor op
    '''
    def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        state, reward, done, _ = self.env.step(action)
        return state.astype(dtype=np.float32), np.array(reward, dtype=np.int32), np.array(done, dtype=np.bool)

    def tf_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(func=self.env_step, inp=[action], Tout=[tf.float32, tf.int32, tf.bool])
