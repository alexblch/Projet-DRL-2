import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import os

class DQNAgent:
    def __init__(self, env, learning_rate=0.0001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.env = env
        self.state_size = env.state_description().shape[0]
        self.action_size = env.TOTAL_ACTIONS
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Paramètres pour la politique epsilon-greedy
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Chemin pour sauvegarder le modèle
        self.path = "models/optimized_dqn.h5"

        # Création du modèle principal
        self.model = self.create_model(input_shape=(self.state_size,), action_space=self.action_size)

        # Configuration du GPU pour TensorFlow si disponible
        self.configure_gpu()

    def configure_gpu(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Utilisation du GPU : {[gpu.name for gpu in physical_devices]}")
            except RuntimeError as e:
                print(e)
        else:
            print("Aucun GPU trouvé. Utilisation du CPU.")

    def create_model(self, input_shape, action_space, layer_sizes=[128, 128]):
        """Crée un modèle DQN."""
        inputs = layers.Input(shape=input_shape)
        x = inputs
        for size in layer_sizes:
            x = layers.Dense(size, activation='relu')(x)
        outputs = layers.Dense(action_space, activation='linear')(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mean_squared_error')
        return model

    def choose_action(self, state):
        """Choisit une action en utilisant une politique epsilon-greedy, en masquant les actions invalides."""
        action_mask = self.env.action_mask()
        valid_actions = np.where(action_mask == 1)[0]

        if np.random.rand() < self.epsilon:
            # Action aléatoire parmi les actions valides
            return np.random.choice(valid_actions)
        else:
            # Prédire les Q-values
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            q_values = self.model(state_tensor)[0].numpy()
            # Masquer les actions invalides en les réglant à -inf
            q_values[~action_mask.astype(bool)] = -np.inf
            return np.argmax(q_values)

    @tf.function
    def train_step(self, state, action, target):
        with tf.GradientTape() as tape:
            q_values = self.model(state, training=True)
            # Extraction de la Q-value pour l'action prise
            q_value = tf.reduce_sum(q_values * tf.one_hot(action, self.action_size), axis=1)
            # Calcul de la perte
            loss = tf.reduce_mean(tf.square(target - q_value))
        # Calcul et application des gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def learn(self, state, action, reward, next_state, done):
        """Met à jour le modèle après chaque étape."""
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        next_state_tensor = tf.convert_to_tensor([next_state], dtype=tf.float32)

        # Prédire les Q-values pour le prochain état
        next_q_values = self.model(next_state_tensor)[0]
        action_mask_next = self.env.action_mask()
        valid_actions_next = tf.where(action_mask_next == 1)[0]

        if done or len(valid_actions_next) == 0:
            target = tf.constant([reward], dtype=tf.float32)
        else:
            max_next_q_value = tf.reduce_max(tf.gather(next_q_values, valid_actions_next))
            target = reward + self.gamma * max_next_q_value

        # Entraîner le modèle
        action_tensor = tf.constant([action], dtype=tf.int32)
        loss = self.train_step(state_tensor, action_tensor, target)

        # Réduire epsilon pour diminuer l'exploration au fil du temps
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

    def save(self):
        """Sauvegarde le modèle entraîné."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.model.save(self.path)
        print(f"Modèle sauvegardé à {self.path}")

    def load(self):
        """Charge un modèle existant."""
        if os.path.exists(self.path):
            self.model = tf.keras.models.load_model(self.path)
            print(f"Modèle chargé depuis {self.path}")
        else:
            print(f"Aucun modèle trouvé à {self.path}. Utilisation du modèle actuel.")
