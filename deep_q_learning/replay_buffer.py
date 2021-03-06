from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self):
        self._memory = deque(maxlen=10000)

    @staticmethod
    def preprocess_state(state):
        return np.reshape(state, (1, 4))

    def remember(self, state, action, reward, next_state):
        if next_state is not None:
            next_state = self.preprocess_state(next_state)
        state = self.preprocess_state(state)
        self._memory.append((state, action, reward, next_state))

    def get_memory(self):
        return self._memory

    @property
    def buffer_size(self):
        return len(self._memory)
