import math
import random
import numpy as np
from q_net import DQNet
from replay_buffer import ReplayBuffer
import gym
import matplotlib.pyplot as plt


class DQAgent:
    def __init__(self, env='', episodes=1000, alpha=0.01, gamma=0.99, min_epsilon=0.1, max_epsilon=1.0,
                 epsilon_decay_rate=0.1, alpha_decay_rate=0.9, batch_size=64):
        self.env = gym.make(env)
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.alpha_decay_rate = alpha_decay_rate
        self.batch_size = batch_size
        self.state_count = self.env.observation_space.shape[0]
        self.action_count = self.env.action_space.n

        self.q_net = DQNet(action_count=self.action_count, state_count=self.state_count,
                           alpha=alpha, lr_decay_rate=alpha_decay_rate, steps=episodes)
        self.memory = ReplayBuffer()

    def choose_epsilon_greedy_action(self, state):
        rnd = random.random()
        if rnd >= self.max_epsilon:
            state = self.memory.preprocess_state(state)
            return np.argmax(self.q_net.predict_on_one(state))
        else:
            return self.env.action_space.sample()

    def decay_epsilon(self, episode):
        self.max_epsilon = max(self.min_epsilon,
                               min(self.max_epsilon, 1.0 - math.log10((episode + 1) * self.epsilon_decay_rate)))

    def fit_on_batch(self):
        size = min(self.memory.buffer_size, self.batch_size)
        mini_batch = random.sample(self.memory.get_memory(), k=size)
        states = np.array([val[0][0] for val in mini_batch])
        next_states = np.array(
            [(np.zeros(self.state_count, ) if val[3] is None else val[3][0]) for val in mini_batch])
        q_s_a = self.q_net.predict_on_batch(states)
        q_s_a_prime = self.q_net.predict_on_batch(next_states)

        x = np.zeros((size, self.state_count))
        y = np.zeros((size, self.action_count))
        for cnt, item in enumerate(mini_batch):
            state, action, reward, next_state = item[0], item[1], item[2], item[3]
            current_q = q_s_a[cnt]
            current_q[action] = reward if next_state is None else reward + (self.gamma * max(q_s_a_prime[cnt]))

            x[cnt] = state
            y[cnt] = current_q

        self.q_net.train_on_batch(states=x, true_q_values=y, epoch=1)

    @staticmethod
    def plot(total, avg, cnt):
        plt.clf()
        plt.plot(cnt, total, label='rewards')
        plt.plot(cnt, avg, label='average reward')
        plt.legend()
        plt.pause(0.01)

    def run(self):
        total_reward = []
        avg_reward = []
        count = []
        for episode in range(self.episodes):
            current_state = self.env.reset()
            done = False
            epsd_rwd = 0
            while not done:
                self.env.render()
                current_action = self.choose_epsilon_greedy_action(current_state)
                next_state, reward, done, _ = self.env.step(current_action)
                epsd_rwd += reward
                if done:
                    next_state = None

                self.memory.remember(current_state, current_action, reward, next_state)
                self.fit_on_batch()
                current_state = next_state

            self.decay_epsilon(episode)

            total_reward.append(epsd_rwd)
            count.append(episode)
            avg = sum(total_reward) / len(count)
            avg_reward.append(avg)
            self.plot(total=total_reward, cnt=count, avg=avg_reward)
            print("episode {}/{}, reward: {} , avg_reward: {}".format(episode, self.episodes, epsd_rwd, avg))

        plt.show()
        self.q_net.save('my_model')
        self.env.close()
