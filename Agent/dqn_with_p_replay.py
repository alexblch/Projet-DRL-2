from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random

class PrioritizedExperienceReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def add(self, error, transition):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + 1e-5

class DQNAgentWithPrioritizedReplay:
    def __init__(self, env, memory_size=2000, batch_size=64, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, alpha=0.6, beta_start=0.4, beta_increase=1e-3):
        self.env = env
        self.state_size = env.state_description().shape[0]
        self.action_size = env.action_mask().shape[0]
        self.memory = PrioritizedExperienceReplayBuffer(memory_size, alpha)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.path = "models/dqn_with_p_replay.h5"

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.beta = beta_start
        self.beta_increase = beta_increase

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
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)

        q_values = self.model.predict(state, verbose=0)
        q_values_next = self.model.predict(next_state, verbose=0)
        target = reward if done else reward + self.gamma * np.max(q_values_next[0])
        td_error = abs(q_values[0][action] - target)
        
        self.memory.add(td_error, (state[0], action, reward, next_state[0], done))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.env.available_actions_ids())
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state, verbose=0)[0]
        mask = self.env.action_mask()
        q_values = np.where(mask == 1, q_values, -np.inf)  # Apply action mask to restrict invalid actions
        return np.argmax(q_values)

    def replay(self):
        if len(self.memory.buffer) < self.batch_size:
            return

        minibatch, indices, weights = self.memory.sample(self.batch_size, beta=self.beta)
        self.beta = min(1.0, self.beta + self.beta_increase)

        states, targets, td_errors = [], [], []
        for (state, action, reward, next_state, done), weight in zip(minibatch, weights):
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)

            q_values = self.model.predict(state, verbose=0)
            q_values_next = self.model.predict(next_state, verbose=0)
            target = reward if done else reward + self.gamma * np.max(q_values_next[0])

            td_error = abs(q_values[0][action] - target)
            td_errors.append(td_error)
            q_values[0][action] = target
            states.append(state[0])
            targets.append(q_values[0])

        self.memory.update_priorities(indices, td_errors)
        states, targets, weights = np.array(states), np.array(targets), np.array(weights)
        self.model.fit(states, targets, sample_weight=weights, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes):
        for e in range(episodes):
            state = self.env.state_description()
            self.env.reset()
            total_reward = 0
            done = False

            while not done:
                available_actions = self.env.available_actions_ids()
                action_mask = self.env.action_mask()
                action = self.choose_action(state)
                
                try:
                    self.env.step(action)
                except ValueError as e:
                    print(f"Action invalide: {e}")
                    done = True
                    reward = -1
                    self.remember(state, action, reward, state, done)
                    break

                next_state = self.env.state_description()
                reward = self.env.score()
                done = self.env.is_game_over()
                self.remember(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward

            print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon}")
            self.replay()
            
    def save(self):
        self.model.save(self.path)
