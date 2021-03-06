from dq_agent import DQAgent
from train import TrainAgent
import tensorflow as tf

ENVIRONMENT1 = 'CartPole-v0'
ENVIRONMENT2 = 'CartPole-v1'
GAMMA = 0.9
ALPHA = 0.01
ALPHA_DECAY_RATE = 0.9
EPISODES = 10000
BATCH_SIZE = 128

dq_agent = DQAgent(env=ENVIRONMENT1, episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, batch_size=BATCH_SIZE,
                   alpha_decay_rate=ALPHA_DECAY_RATE)

reinforce_pg_agent = TrainAgent(env=ENVIRONMENT1, alpha=ALPHA, alpha_decay_rate=ALPHA_DECAY_RATE, gamma=GAMMA,
                                episodes=EPISODES)

if __name__ == "__main__":
    # tf.compat.v1.disable_eager_execution()

    reinforce_pg_agent.run()
    # dq_agent.run()
