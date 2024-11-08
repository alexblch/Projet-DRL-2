import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers

class DQNAgentWithReplay:
    def __init__(self, env, memory_size=2000, batch_size=64, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.env = env
        self.state_size = env.state_description().shape[0]
        self.action_size = env.TOTAL_ACTIONS  # Utiliser la constante de l'environnement
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.path = "models/dqn_with_replay.h5"

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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        action_mask = self.env.action_mask()
        if np.random.rand() < self.epsilon:
            valid_actions = np.where(action_mask == 1)[0]
            return np.random.choice(valid_actions)
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state, verbose=0)[0]
        # Appliquer le masque d'actions
        masked_q_values = np.full_like(q_values, -np.inf)
        masked_q_values[action_mask == 1] = q_values[action_mask == 1]
        return np.argmax(masked_q_values)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([sample[0] for sample in minibatch])
        targets = np.zeros((self.batch_size, self.action_size))

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            q_values = self.model.predict(state, verbose=0)
            q_values_next = self.model.predict(next_state, verbose=0)
            target = reward if done else reward + self.gamma * np.max(q_values_next[0])
            q_values[0][action] = target
            targets[i] = q_values[0]

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self):
        self.model.save(self.path)
