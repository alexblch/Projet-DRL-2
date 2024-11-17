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
        self.action_size = env.TOTAL_ACTIONS  # Use the constant from the environment
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.path = "models/dqn_with_replay.keras"

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Create the model with a simpler architecture for faster computation
        self.model = self.create_model(input_shape=(self.state_size,), action_space=self.action_size)

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

    def create_model(self, input_shape, action_space, layer_sizes=[64, 64]):
        inputs = layers.Input(shape=input_shape)
        x = layers.Dense(layer_sizes[0], activation='relu')(inputs)
        x = layers.Dense(layer_sizes[1], activation='relu')(x)
        outputs = layers.Dense(action_space, activation='linear')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
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
        q_values = self.model(state, training=False).numpy()[0]
        # Apply the action mask
        masked_q_values = np.full_like(q_values, -np.inf)
        masked_q_values[action_mask == 1] = q_values[action_mask == 1]
        return np.argmax(masked_q_values)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([sample[0] for sample in minibatch], dtype=np.float32)
        actions = np.array([sample[1] for sample in minibatch], dtype=np.int32)
        rewards = np.array([sample[2] for sample in minibatch], dtype=np.float32)
        next_states = np.array([sample[3] for sample in minibatch], dtype=np.float32)
        dones = np.array([sample[4] for sample in minibatch], dtype=np.float32)

        # Batch prediction for states and next_states
        q_values = self.model(states, training=False).numpy()
        q_values_next = self.model(next_states, training=False).numpy()

        # Compute target Q-values
        max_q_values_next = np.max(q_values_next, axis=1)
        targets = q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * max_q_values_next[i]

        # Train the model
        self.model.fit(states, targets, epochs=1, verbose=0)

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self):
        self.model.save(self.path)

    def load(self):
        self.model = tf.keras.models.load_model(self.path)
