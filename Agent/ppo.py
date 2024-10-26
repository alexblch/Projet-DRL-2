import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class PPOAgent:
    def __init__(self, state_size, action_size, learning_rate=0.0003, gamma=0.99, clip_ratio=0.2, 
                 update_epochs=10, lambda_=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.lambda_ = lambda_

        # Initialiser les modèles de politique et de critique
        self.policy_model = self.create_policy_model(input_shape=(self.state_size,), action_space=self.action_size)
        self.value_model = self.create_value_model(input_shape=(self.state_size,))
        
        # Stockage pour les trajectoires de l'agent
        self.states, self.actions, self.rewards, self.dones, self.log_probs, self.values = [], [], [], [], [], []

    def create_policy_model(self, input_shape, action_space, layer_sizes=[64, 64]):
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=input_shape))
        model.add(layers.Dense(layer_sizes[0], activation='relu'))
        model.add(layers.Dense(layer_sizes[1], activation='relu'))
        model.add(layers.Dense(action_space, activation='softmax'))
        return model

    def create_value_model(self, input_shape, layer_sizes=[64, 64]):
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=input_shape))
        model.add(layers.Dense(layer_sizes[0], activation='relu'))
        model.add(layers.Dense(layer_sizes[1], activation='relu'))
        model.add(layers.Dense(1, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
        return model

    def choose_action(self, state):
        state = np.expand_dims(state, axis=0)
        action_probs = self.policy_model.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_size, p=action_probs)
        
        log_prob = np.log(action_probs[action])
        value = self.value_model.predict(state, verbose=0)[0][0]

        return action, log_prob, value

    def store_transition(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(self, next_value):
        values = self.values + [next_value]
        gae = 0
        returns = []
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_ * (1 - self.dones[t]) * gae
            returns.insert(0, gae + values[t])
        return returns

    def train(self, next_value):
        returns = self.compute_gae(next_value)
        states = np.array(self.states)
        actions = np.array(self.actions)
        old_log_probs = np.array(self.log_probs)
        returns = np.array(returns)
        values = np.array(self.values)

        advantages = returns - values

        for _ in range(self.update_epochs):
            with tf.GradientTape() as tape:
                action_probs = self.policy_model(states, training=True)
                selected_action_probs = tf.gather_nd(action_probs, np.vstack((np.arange(len(actions)), actions)).T)
                
                log_probs = tf.math.log(selected_action_probs)
                ratios = tf.exp(log_probs - old_log_probs)

                clipped_ratios = tf.clip_by_value(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
                surrogate_loss = -tf.reduce_mean(tf.minimum(ratios * advantages, clipped_ratios * advantages))

                value_preds = self.value_model(states, training=True)
                value_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(value_preds)))

                loss = surrogate_loss + 0.5 * value_loss

            grads = tape.gradient(loss, self.policy_model.trainable_variables + self.value_model.trainable_variables)
            tf.keras.optimizers.Adam(learning_rate=self.learning_rate).apply_gradients(
                zip(grads, self.policy_model.trainable_variables + self.value_model.trainable_variables)
            )

        self.states, self.actions, self.rewards, self.dones, self.log_probs, self.values = [], [], [], [], [], []
