import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class A2CAgent:
    def __init__(
        self,
        env,
        learning_rate=0.0007,
        gamma=0.99,
        lambda_=0.95,
        entropy_coeff=0.01,
        value_coeff=0.5,
        batch_size=64,
        n_steps=5  # Number of steps to collect before updating
    ):
        self.env = env
        self.state_size = env.state_description().shape[0]
        self.action_size = env.action_mask().shape[0]
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.policy_model_path = "models/a2c/policy_model.h5"
        self.value_model_path = "models/a2c/value_model.h5"

        # Initialize policy and value models with simplified architectures
        self.policy_model = self.create_policy_model(
            input_shape=(self.state_size,), action_space=self.action_size, layer_sizes=[32, 32]
        )
        self.value_model = self.create_value_model(
            input_shape=(self.state_size,), layer_sizes=[32, 32]
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Storage for agent trajectories
        self.reset_storage()

        # Configure GPU usage if available
        self.configure_gpu()

    def reset_storage(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

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

    def create_policy_model(self, input_shape, action_space, layer_sizes=[32, 32]):
        inputs = layers.Input(shape=input_shape)
        x = layers.Dense(layer_sizes[0], activation="relu")(inputs)
        x = layers.Dense(layer_sizes[1], activation="relu")(x)
        outputs = layers.Dense(action_space, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def create_value_model(self, input_shape, layer_sizes=[32, 32]):
        inputs = layers.Input(shape=input_shape)
        x = layers.Dense(layer_sizes[0], activation="relu")(inputs)
        x = layers.Dense(layer_sizes[1], activation="relu")(x)
        outputs = layers.Dense(1, activation="linear")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def choose_action(self, state, action_mask):
        if state is None:
            raise ValueError("L'état (state) est invalide ou non défini.")

        state = np.expand_dims(state, axis=0)
        action_probs = self.policy_model(state, training=False).numpy()[0]

        # Ensure action_mask is float32 and same dtype as action_probs
        action_mask = action_mask.astype(np.float32)
        action_probs *= action_mask

        if np.sum(action_probs) == 0:
            raise ValueError(
                "Le masque d'action a annulé toutes les probabilités. Vérifiez la validité des actions."
            )

        action_probs /= np.sum(action_probs)  # Normalize to get a valid probability distribution

        action = np.random.choice(self.action_size, p=action_probs)
        log_prob = np.log(action_probs[action] + 1e-8)  # Add epsilon for numerical stability
        value = self.value_model(state, training=False).numpy()[0][0]

        return action, log_prob, value

    def store_transition(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    @tf.function  # Compile the function into a TensorFlow graph
    def train_step(self, states, actions, returns, advantages):
        with tf.GradientTape() as tape:
            # Forward pass
            action_probs = self.policy_model(states, training=True)
            values_pred = self.value_model(states, training=True)
            values_pred = tf.squeeze(values_pred)

            # Compute new log_probs
            indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            selected_action_probs = tf.gather_nd(action_probs, indices)
            new_log_probs = tf.math.log(selected_action_probs + 1e-8)

            # Policy loss
            policy_loss = -tf.reduce_mean(new_log_probs * advantages)

            # Value loss
            value_loss = self.value_coeff * tf.reduce_mean(tf.square(returns - values_pred))

            # Entropy bonus
            entropy = -tf.reduce_mean(
                tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1)
            )
            entropy_loss = -self.entropy_coeff * entropy

            # Total loss
            total_loss = policy_loss + value_loss + entropy_loss

        # Compute gradients and update parameters
        grads = tape.gradient(
            total_loss,
            self.policy_model.trainable_variables + self.value_model.trainable_variables,
        )
        self.optimizer.apply_gradients(
            zip(
                grads,
                self.policy_model.trainable_variables + self.value_model.trainable_variables,
            )
        )
        return total_loss

    def train(self, next_value):
        # Convert lists to numpy arrays
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)

        # Compute returns and advantages using vectorized operations
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lambda_ * (1 - dones[i]) * gae
            advantages[i] = gae
            returns[i] = gae + values[i]
            next_value = values[i]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        returns = tf.convert_to_tensor(returns)
        advantages = tf.convert_to_tensor(advantages)

        # Perform a single optimization step
        total_loss = self.train_step(states, actions, returns, advantages)

        # Reset storage
        self.reset_storage()

        # Return the total loss for logging
        return total_loss.numpy()

    def save(self):
        self.policy_model.save(self.policy_model_path)
        self.value_model.save(self.value_model_path)

    def load(self):
        self.policy_model = tf.keras.models.load_model(self.policy_model_path)
        self.value_model = tf.keras.models.load_model(self.value_model_path)
