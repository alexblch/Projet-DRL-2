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
        n_steps=5  # Nombre d'étapes avant mise à jour
    ):
        self.env = env
        self.state_size = env.state_description().shape[0]
        self.action_size = env.action_mask().shape[0]
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.n_steps = n_steps
        self.policy_model_path = "models/a2c/policy_model.h5"
        self.value_model_path = "models/a2c/value_model.h5"

        # Initialiser les modèles de politique et de valeur
        self.policy_model = self.create_policy_model()
        self.value_model = self.create_value_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Stockage des trajectoires
        self.reset_storage()

        # Configurer l'utilisation du GPU si disponible
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
                print(f"Utilisation du GPU: {physical_devices}")
            except RuntimeError as e:
                print(e)
        else:
            print("Aucun GPU trouvé. Utilisation du CPU.")

    def create_policy_model(self):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(128, activation="relu")(inputs)
        x = layers.Dense(128, activation="relu")(x)
        outputs = layers.Dense(self.action_size, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def create_value_model(self):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(128, activation="relu")(inputs)
        x = layers.Dense(128, activation="relu")(x)
        outputs = layers.Dense(1, activation="linear")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def choose_action(self, state):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        action_probs = self.policy_model(state).numpy()[0]

        # Appliquer le masque d'action
        action_mask = self.env.action_mask().astype(np.float32)
        action_probs *= action_mask

        if np.sum(action_probs) == 0:
            # Si toutes les actions sont invalides, choisir aléatoirement parmi les actions valides
            valid_actions = np.where(action_mask == 1)[0]
            action = np.random.choice(valid_actions)
            log_prob = np.log(1.0 / len(valid_actions))
        else:
            action_probs /= np.sum(action_probs)
            action = np.random.choice(self.action_size, p=action_probs)
            log_prob = np.log(action_probs[action] + 1e-8)

        value = self.value_model(state).numpy()[0][0]

        return action, log_prob, value

    def store_transition(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns_and_advantages(self, next_value):
        rewards = np.array(self.rewards + [next_value], dtype=np.float32)
        dones = np.array(self.dones + [0], dtype=np.float32)
        values = np.array(self.values + [next_value], dtype=np.float32)

        advantages = np.zeros_like(rewards[:-1])
        gae = 0
        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_ * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return returns, advantages

    @tf.function
    def train_step(self, states, actions, returns, advantages):
        with tf.GradientTape() as tape:
            action_probs = self.policy_model(states, training=True)
            values = self.value_model(states, training=True)
            values = tf.squeeze(values, axis=1)

            # Obtenir les log-probabilités des actions sélectionnées
            action_indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            selected_action_probs = tf.gather_nd(action_probs, action_indices)
            log_probs = tf.math.log(selected_action_probs + 1e-8)

            # Calcul des pertes
            policy_loss = -tf.reduce_mean(log_probs * advantages)
            value_loss = self.value_coeff * tf.reduce_mean(tf.square(returns - values))
            entropy_loss = -self.entropy_coeff * tf.reduce_mean(
                tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1)
            )

            total_loss = policy_loss + value_loss + entropy_loss

        grads = tape.gradient(total_loss, self.policy_model.trainable_variables + self.value_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_model.trainable_variables + self.value_model.trainable_variables))
        return total_loss

    def train(self, next_state, done):
        next_state = np.expand_dims(next_state, axis=0).astype(np.float32)
        next_value = self.value_model(next_state).numpy()[0][0] if not done else 0

        returns, advantages = self.compute_returns_and_advantages(next_value)
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int32)

        # Normaliser les avantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convertir en tenseurs
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        returns = tf.convert_to_tensor(returns)
        advantages = tf.convert_to_tensor(advantages)

        # Entraîner
        loss = self.train_step(states, actions, returns, advantages)

        # Réinitialiser le stockage
        self.reset_storage()

        return loss.numpy()

    def save(self):
        self.policy_model.save(self.policy_model_path)
        self.value_model.save(self.value_model_path)

    def load(self):
        self.policy_model = tf.keras.models.load_model(self.policy_model_path)
        self.value_model = tf.keras.models.load_model(self.value_model_path)
