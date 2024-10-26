import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class REINFORCEAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma


        self.model = self.create_model(input_shape=(self.state_size,), action_space=self.action_size, learning_rate=self.learning_rate)
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    def create_model(self, input_shape, action_space, layer_sizes=[128, 128], learning_rate=0.001):
        model = tf.keras.Sequential()
        
        model.add(layers.InputLayer(input_shape=input_shape))
        model.add(layers.Dense(layer_sizes[0], activation='relu'))
        model.add(layers.Dense(layer_sizes[1], activation='relu'))
        
        model.add(layers.Dense(action_space, activation='softmax'))
    
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy')
        
        return model

    def choose_action(self, state):
        state = np.expand_dims(state, axis=0)
        action_probs = self.model.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_size, p=action_probs)
        return action

    def store_transition(self, state, action, reward):
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def discount_rewards(self):
        discounted_rewards = np.zeros_like(self.episode_rewards, dtype=np.float32)
        cumulative = 0
        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * self.gamma + self.episode_rewards[t]
            discounted_rewards[t] = cumulative

        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards) + 1e-8
        return discounted_rewards

    def train(self):
        discounted_rewards = self.discount_rewards()
        states = np.array(self.episode_states)
        actions = np.array(self.episode_actions)
        advantages = discounted_rewards


        action_masks = np.zeros((len(actions), self.action_size))
        action_masks[np.arange(len(actions)), actions] = 1


        self.model.fit(states, action_masks, sample_weight=advantages, epochs=1, verbose=0)


        self.episode_states, self.episode_actions, self.episode_rewards = [], [], []
