import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class DQNAgent:
    def __init__(self, env, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.env = env
        self.state_size = env.state_description().shape[0]
        self.action_size = env.TOTAL_ACTIONS  # Utiliser la constante de l'environnement
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.path = "models/dqn.h5"

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.model = self.create_model(input_shape=(self.state_size,), action_space=self.action_size, learning_rate=self.learning_rate)

    def create_model(self, input_shape, action_space, layer_sizes=[128, 128], learning_rate=0.001):
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=input_shape))
        for size in layer_sizes:
            model.add(layers.Dense(size, activation='relu'))
        model.add(layers.Dense(action_space, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mean_squared_error')
        return model

    def choose_action(self, state):
        action_mask = self.env.action_mask()
        valid_actions = np.where(action_mask == 1)[0]
        if np.random.rand() < self.epsilon:
            action = np.random.choice(valid_actions)
        else:
            state = np.expand_dims(state, axis=0)
            q_values = self.model.predict(state, verbose=0)[0]
            masked_q_values = np.full_like(q_values, -np.inf)
            masked_q_values[valid_actions] = q_values[valid_actions]
            action = np.argmax(masked_q_values)
        return action

    def learn(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)

        action_mask_next = self.env.action_mask()
        valid_actions_next = np.where(action_mask_next == 1)[0]

        q_values = self.model.predict(state, verbose=0)
        q_values_next = self.model.predict(next_state, verbose=0)[0]

        if done or len(valid_actions_next) == 0:
            target = reward
        else:
            masked_q_values_next = q_values_next[valid_actions_next]
            target = reward + self.gamma * np.max(masked_q_values_next)

        q_values[0][action] = target

        self.model.fit(state, q_values, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self):
        self.model.save(self.path)
