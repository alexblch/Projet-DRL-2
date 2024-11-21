# Agent/DQNAgent.py

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import os

class DQNAgent:
    def __init__(self, env, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.env = env
        self.state_size = env.state_description().shape[0]
        self.action_size = env.TOTAL_ACTIONS
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Epsilon parameters for epsilon-greedy policy
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Path to save the model
        self.path = "models/optimized_dqn.h5"

        # Create the main model
        self.model = self.create_model(input_shape=(self.state_size,), action_space=self.action_size)

        # Configure GPU for TensorFlow if available
        self.configure_gpu()

    def configure_gpu(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU: {[gpu.name for gpu in physical_devices]}")
            except RuntimeError as e:
                print(e)
        else:
            print("No GPU found. Using CPU.")

    def create_model(self, input_shape, action_space, layer_sizes=[128, 128]):
        """Crée un modèle DQN rapide avec une architecture simplifiée."""
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        for size in layer_sizes:
            model.add(layers.Dense(size, activation='relu'))
        model.add(layers.Dense(action_space, activation='linear'))
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mean_squared_error')
        return model

    def choose_action(self, state):
        """Choisir une action avec epsilon-greedy, masquant les actions invalides."""
        action_mask = self.env.action_mask()
        valid_actions = np.where(action_mask == 1)[0]

        if np.random.rand() < self.epsilon:
            # Action aléatoire parmi les actions valides
            return np.random.choice(valid_actions)
        else:
            # Prédiction des Q-values
            state = np.expand_dims(state, axis=0)  # Ajouter une dimension batch
            q_values = self.model.predict(state, verbose=0)[0]
            # Masquer les actions invalides en attribuant une très basse valeur
            masked_q_values = np.full_like(q_values, -np.inf)
            masked_q_values[valid_actions] = q_values[valid_actions]
            return np.argmax(masked_q_values)

    def learn(self, state, action, reward, next_state, done):
        """Met à jour le modèle après chaque étape."""
        state = np.expand_dims(state, axis=0)  # Ajouter une dimension batch
        next_state = np.expand_dims(next_state, axis=0)  # Ajouter une dimension batch

        # Prédire les Q-values pour l'état actuel
        q_values = self.model.predict(state, verbose=0)

        if done:
            target = reward
        else:
            # Prédire les Q-values pour le prochain état
            q_values_next = self.model.predict(next_state, verbose=0)[0]
            # Masquer les actions invalides pour le prochain état
            action_mask_next = self.env.action_mask()
            valid_actions_next = np.where(action_mask_next == 1)[0]
            if len(valid_actions_next) == 0:
                target = reward
            else:
                target = reward + self.gamma * np.max(q_values_next[valid_actions_next])

        # Mettre à jour la Q-value de l'action prise
        q_values[0][action] = target

        # Entraîner le modèle sur les Q-values mises à jour
        self.model.fit(state, q_values, epochs=1, verbose=0, batch_size=1)

        # Réduire epsilon pour diminuer l'exploration au fil du temps
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self):
        """Sauvegarder le modèle entraîné."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.model.save(self.path)
        print(f"Modèle sauvegardé à {self.path}")

    def load(self):
        """Charger un modèle existant."""
        if os.path.exists(self.path):
            self.model = tf.keras.models.load_model(self.path)
            print(f"Modèle chargé depuis {self.path}")
        else:
            print(f"Aucun modèle trouvé à {self.path}. Utilisation du modèle actuel.")
