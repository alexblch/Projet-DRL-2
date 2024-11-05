import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class PPOAgent:
    def __init__(self, env, learning_rate=0.0003, gamma=0.99, clip_ratio=0.2, update_epochs=10, lambda_=0.95):
        self.env = env
        self.state_size = env.state_description().shape[0]
        self.action_size = env.action_mask().shape[0]
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.lambda_ = lambda_
        self.policy_model_path = "models/ppo/policy_model.h5"
        self.value_model_path = "models/ppo/value_model.h5"

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
        # Vérifie si `state` est valide
        if state is None:
            raise ValueError("L'état (state) est invalide ou non défini.")

        state = np.expand_dims(state, axis=0)
        action_probs = self.policy_model.predict(state, verbose=0)[0]

        # Applique le masque d'action
        mask = self.env.action_mask()
        action_probs *= mask  # Applique le masque d'action
        if np.sum(action_probs) == 0:
            raise ValueError("Le masque d'action a annulé toutes les probabilités. Vérifiez la validité des actions.")
        
        action_probs /= np.sum(action_probs)  # Normalise pour obtenir une distribution de probabilité valide

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

        # Réinitialisation des trajectoires
        self.states, self.actions, self.rewards, self.dones, self.log_probs, self.values = [], [], [], [], [], []

    def run_episode(self):
        state = self.env.state_description()
        self.env.reset()
        done = False
        total_reward = 0

        while not done:
            action, log_prob, value = self.choose_action(state)
            try:
                self.env.step(action)
            except ValueError as e:
                print(f"Action invalide: {e}")
                done = True
                reward = -1
                self.store_transition(state, action, reward, done, log_prob, value)
                break

            next_state = self.env.state_description()
            reward = self.env.score()
            done = self.env.is_game_over()
            self.store_transition(state, action, reward, done, log_prob, value)
            state = next_state
            total_reward += reward

        next_value = self.value_model.predict(np.expand_dims(state, axis=0), verbose=0)[0][0]
        self.train(next_value)
        print(f"Total Reward: {total_reward}")
        
    def save(self):
        self.policy_model.save(self.policy_model_path)
        self.value_model.save(self.value_model_path)
        
