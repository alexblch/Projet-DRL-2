# Agent/reinforce_with_learned_critic.py

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class REINFORCEWithBaselineLearnedCriticAgent:
    def __init__(self, env, learning_rate=0.001, gamma=0.99):
        self.env = env
        self.state_size = env.state_description().shape[0]
        self.action_size = env.action_mask().shape[0]
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model_path = "models/reinforce_with_baseline_learned_critic.h5"

        # Policy model
        self.policy_model = self.create_policy_model(
            input_shape=(self.state_size,), action_space=self.action_size
        )
        
        # Critic model
        self.critic_model = self.create_critic_model(input_shape=(self.state_size,))

        # Lists to store episode transitions
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

        # Configure GPU usage if available
        self.configure_gpu()

    def configure_gpu(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU: {physical_devices}")
            except RuntimeError as e:
                print(e)
        else:
            print("No GPU found. Using CPU.")

    def create_policy_model(self, input_shape, action_space, layer_sizes=[64, 64]):
        inputs = layers.Input(shape=input_shape)
        x = layers.Dense(layer_sizes[0], activation='relu')(inputs)
        x = layers.Dense(layer_sizes[1], activation='relu')(x)
        outputs = layers.Dense(action_space, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=self.policy_loss
        )
        return model

    def create_critic_model(self, input_shape, layer_sizes=[64, 64]):
        inputs = layers.Input(shape=input_shape)
        x = layers.Dense(layer_sizes[0], activation='relu')(inputs)
        x = layers.Dense(layer_sizes[1], activation='relu')(x)
        outputs = layers.Dense(1, activation='linear')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error'
        )
        return model

    def policy_loss(self, y_true, y_pred):
        return tf.reduce_mean(-y_true * tf.math.log(y_pred + 1e-8))

    def choose_action(self, state):
        state = np.expand_dims(state, axis=0)
        action_probs = self.policy_model(state, training=False).numpy()[0]

        # Apply action mask
        action_mask = self.env.action_mask().astype(np.float32)
        action_probs *= action_mask
        if np.sum(action_probs) == 0:
            raise ValueError("Action mask has zero probability for all actions.")
        action_probs /= np.sum(action_probs)
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
        return discounted_rewards

    def calculate_advantages(self, discounted_rewards):
        states = np.array(self.episode_states, dtype=np.float32)
        baseline_values = self.critic_model.predict(states).flatten()
        advantages = discounted_rewards - baseline_values
        return advantages

    @tf.function
    def train_step(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            action_probs = self.policy_model(states, training=True)
            indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            selected_action_probs = tf.gather_nd(action_probs, indices)
            log_probs = tf.math.log(selected_action_probs + 1e-8)
            policy_loss = -tf.reduce_mean(log_probs * advantages)
        policy_grads = tape.gradient(policy_loss, self.policy_model.trainable_variables)
        self.policy_model.optimizer.apply_gradients(zip(policy_grads, self.policy_model.trainable_variables))
        return policy_loss

    def train(self):
        discounted_rewards = self.discount_rewards()
        advantages = self.calculate_advantages(discounted_rewards)

        states = np.array(self.episode_states, dtype=np.float32)
        actions = np.array(self.episode_actions, dtype=np.int32)
        advantages = advantages.astype(np.float32)

        # Convert to tensors
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        advantages = tf.convert_to_tensor(advantages)

        # Train policy model
        policy_loss = self.train_step(states, actions, advantages)

        # Train critic model
        discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)
        self.critic_model.fit(states, discounted_rewards, verbose=0)

        # Reset episode data
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        return policy_loss.numpy()

    def save(self):
        self.policy_model.save(self.model_path + "_policy")
        self.critic_model.save(self.model_path + "_critic")

    def load(self):
        self.policy_model = tf.keras.models.load_model(self.model_path + "_policy")
        self.critic_model = tf.keras.models.load_model(self.model_path + "_critic")
