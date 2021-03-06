import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from environment import Environment
from policy_net import ReinforcePolicyNet, ReinforcePolicyModel
from reinforce_policy_agent import ReinforcePolicyAgent
import matplotlib.pyplot as plt


class TrainAgent:
    def __init__(self, env: '', episodes=1000, alpha=0.01, gamma=0.9, alpha_decay_rate=0.9):
        self.env = Environment(env=env)
        self.episodes = episodes
        self.lr = ExponentialDecay(alpha, episodes, alpha_decay_rate)
        self.optimizer = Adam(self.lr)
        self.action_count, self.states_count = self.env.spaces_count()
        self.gamma = gamma
        self._net = ReinforcePolicyNet(action_count=self.action_count, states_count=self.states_count)
        self._model = ReinforcePolicyModel(self._net)
        self._agent = ReinforcePolicyAgent(env=self.env, model=self._model, gamma=gamma)
        self.huber_loss = Huber(reduction=tf.keras.losses.Reduction.SUM)

    def compute_loss(self, action_prob, epi_return, values):
        """
            actually action prob is our policy that give each action a probability over a distribution.
            here the actor loss is -mean(pi(a|s) * Bt) which should be minimized, 'Bt ' refers to the Baseline that we use
            with the purpose of reducing the variance
        """
        advantage = epi_return - values
        prob = tf.math.log(action_prob + 1e-30)
        actor_loss = -tf.math.reduce_mean(prob * advantage)

        critic_loss = self.huber_loss(values, epi_return)

        return critic_loss + actor_loss

    @tf.function
    def train_step(self, init_state: tf.Tensor):
        with tf.GradientTape() as tape:
            episode_return, action_probs, rewards, values = self._agent.run(max_steps=200, init_state=init_state)
            loss = self.compute_loss(action_prob=action_probs, epi_return=episode_return, values=values)

        grads = tape.gradient(loss, self._net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self._net.trainable_variables))

        episode_rewards = tf.math.reduce_sum(rewards)
        return episode_rewards

    @staticmethod
    def plot_me(total, avg, cnt):
        plt.clf()
        plt.plot(cnt, total, label='rewards')
        plt.plot(cnt, avg, label='average reward')
        plt.legend()
        plt.pause(0.01)

    def run(self):
        e_r = []
        count = []
        avg_reward = []
        for episode in range(self.episodes):
            init_state = tf.constant(self.env.reset_env(), dtype=tf.float32)
            e_r.append(int(self.train_step(init_state)))
            count.append(episode)
            avg = sum(e_r) / len(count)
            avg_reward.append(avg)
            self.plot_me(e_r,avg_reward, count)
            print(f"episode {episode}/{self.episodes}, reward: {e_r[episode]}")
