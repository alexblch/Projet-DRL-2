from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
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
        model.add(layers.Dense(layer_sizes[0], activation='relu'))
        model.add(layers.Dense(layer_sizes[1], activation='relu'))
        
        model.add(layers.Dense(action_space, activation='linear'))
    
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mean_squared_error')
        
        return model

    def choose_action(self, state, available_actions, action_mask):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(available_actions)
        else:
            state = np.expand_dims(state, axis=0)
            q_values = self.model.predict(state, verbose=0)[0]
            masked_q_values = np.copy(q_values)
            masked_q_values[action_mask == 0] = -np.inf
            action = np.argmax(masked_q_values)
        return action

    def learn(self, state, action, reward, next_state, done, action_mask_next):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)

        q_values = self.model.predict(state, verbose=0)
        q_values_next = self.model.predict(next_state, verbose=0)
        masked_q_values_next = q_values_next + (action_mask_next - 1) * 1e9

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(masked_q_values_next[0])

        q_values[0][action] = target

        self.model.fit(state, q_values, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
            
    def save(self):
        self.model.save(self.path)
