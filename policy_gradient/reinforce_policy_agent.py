import tensorflow as tf
from environment import Environment
from policy_net import ReinforcePolicyModel


class ReinforcePolicyAgent:
    def __init__(self, env: Environment, model: ReinforcePolicyModel, gamma=0.9):
        self.gamma = gamma
        self.env = env
        self._policy_model = model

    def get_return(self, epi_rewards: tf.Tensor) -> tf.TensorArray:

        # --n shows total number of steps we took in a single episode
        n = tf.shape(epi_rewards)[0]

        # --episode return size is equal to number of steps
        episode_returns = tf.TensorArray(tf.float32, size=n, dynamic_size=True)

        '''
            because as we take more steps we get closer to end of the episode,
            the last reward in rewards list should incur less discount, so we reverse the list
        '''
        epi_rewards = tf.cast(epi_rewards[::-1], dtype=tf.float32)

        discounted_reward = tf.constant(0.0)
        discounted_reward_shape = discounted_reward.get_shape()

        for i in tf.range(n):
            '''
                we reversed the rewards list, it means we are going to start from the last steps and
                go backwards to the start, with this approach its possible to update whole values we've
                seen in episode at the end of it
            '''
            discounted_reward = epi_rewards[i] + self.gamma * discounted_reward
            discounted_reward.set_shape(discounted_reward_shape)
            episode_returns = episode_returns.write(i, discounted_reward)

        episode_returns = episode_returns.stack()[::-1]
        return episode_returns

    def run(self, max_steps, init_state: tf.Tensor):
        actions_prob = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        values = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        episode_return = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        current_state = init_state
        initial_state_shape = init_state.shape

        for i in tf.range(max_steps):

            '''
                expand states dimensions to be a appropriate input for model
            '''
            current_state = tf.expand_dims(current_state, 0)

            '''
                our model is a fully connected neural net, which we get two outputs from it 
                so in other words, the actor and critic both share the same parameters.
                the policy that we have chosen is "Softmax policy", thus the model predictions for 
                actions is a set of probability distributions, which with 'choose_action' method we try to choose a 
                random action from this distribution and push up its probability if its good and vice versa.             
            '''
            current_action_probs, current_value = self._policy_model.predict(current_state)
            current_action = self._policy_model.choose_action(current_action_probs)

            next_state, reward, done = self.env.tf_env_step(current_action)
            current_state = next_state
            current_state.set_shape(initial_state_shape)

            current_value = tf.squeeze(current_value, 0)
            values = values.write(i, current_value)
            actions_prob = actions_prob.write(i, current_action_probs[0][current_action])
            rewards = rewards.write(i, reward)

            if done:
                break

        actions_prob = actions_prob.stack()
        rewards = rewards.stack()
        values = values.stack()
        episode_return = self.get_return(rewards)

        return episode_return, actions_prob, rewards, values
