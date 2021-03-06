import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, schedules


class DQNet:
    def __init__(self, action_count, state_count, alpha=0.1, lr_decay_rate=0.9, steps=1000):
        self._state_count = state_count
        self._action_count = action_count
        self._model = self._build_model()
        # self._model.build()
        self._lr = schedules.ExponentialDecay(initial_learning_rate=alpha, decay_rate=lr_decay_rate, decay_steps=steps,
                                              staircase=True)
        self._optimizer = Adam(learning_rate=self._lr)
        self._model.compile(optimizer=self._optimizer, loss='mse')

    def _build_model(self):
        # tf.compat.v1.disable_eager_execution()
        model = Sequential()
        model.add(Dense(units=self._state_count, activation='relu', input_shape=(4,)))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=128, activation='relu'))
        # model.add(Dense(units=64, activation='relu'))
        # model.add(Dense(units=64, activation='relu'))
        # model.add(Dense(units=128, activation='relu'))
        # model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=self._action_count, activation='linear'))
        return model

    def predict_on_one(self, state):
        return self._model.predict(state)

    def predict_on_batch(self, state):
        return self._model.predict_on_batch(state)

    def train_on_batch(self, states, true_q_values, epoch=1):
        self._model.fit(x=states, y=true_q_values, epochs=epoch, verbose=0)

    def save(self, path):
        self._model.save(path)

    # @tf.function
    # def train_one_step(self, state, true_q):
    #     with tf.GradientTape() as tape:
    #         current_q = self._model(state)
    #         loss = MSE(true_q, current_q)
    #     grads = tape.gradient(loss, self._model.trainable_variables)
    #     self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
    #
    #     return loss
    #
    # @tf.function
    # def train_on_batch(self, states, true_q_values, epoch=1, batch_size=1):
    #     for step in range(epoch):
    #         # for x, y in zip(states, true_q_values):
    #         # x = tf.convert_to_tensor(states)
    #         # y = tf.convert_to_tensor(true_q_values)
    #         self.train_one_step(state=states, true_q=true_q_values)
    #         # step += 1
    #
    #     return step
