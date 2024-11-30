import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class AlphaZeroAgent:
    def __init__(self, env, n_simulations=100, c_puct=1.4):
        self.env = env
        self.state_size = env.state_description().shape[0]
        self.action_size = env.TOTAL_ACTIONS
        self.n_simulations = n_simulations
        self.c_puct = c_puct

        # Données pour l'entraînement
        self.training_data = []

        # Initialiser le réseau de neurones
        self.model = self.create_model()

        # Configurer le GPU
        self.configure_gpu()

    def configure_gpu(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    def create_model(self):
        # Modèle simplifié
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)

        # Tête de politique
        policy_logits = layers.Dense(self.action_size, activation='linear', name='policy')(x)

        # Tête de valeur
        value = layers.Dense(1, activation='tanh', name='value')(x)

        model = models.Model(inputs=inputs, outputs=[policy_logits, value])
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                      loss={'policy': 'categorical_crossentropy', 'value': 'mean_squared_error'})
        return model

    def policy_value_fn(self, state):
        """
        Fonction qui prend un état et retourne les probabilités d'action et la valeur estimée.
        """
        state_desc = state.state_description()
        policy_logits, value = self.model.predict(np.array([state_desc]), verbose=0)
        policy_probs = self._softmax(policy_logits[0])

        # Appliquer le masque d'actions valides
        valid_actions = state.available_actions_ids()
        mask = np.zeros_like(policy_probs)
        mask[valid_actions] = 1
        policy_probs *= mask
        policy_probs /= np.sum(policy_probs)

        # Créer un dictionnaire {action: probabilité}
        action_probs = {a: policy_probs[a] for a in valid_actions}
        return action_probs, value[0][0]

    def choose_action(self):
        state = self.env.clone()
        mcts = MCTS(self.policy_value_fn, self.n_simulations, self.c_puct)
        action_probs = mcts.get_action_probs(state)

        # Convertir en vecteur pour l'entraînement
        action_probs_vec = np.zeros(self.action_size)
        for action, prob in action_probs.items():
            action_probs_vec[action] = prob

        # Choisir l'action
        action = np.random.choice(list(action_probs.keys()), p=list(action_probs.values()))

        # Stocker les données pour l'entraînement
        state_desc = state.state_description()
        self.training_data.append((state_desc, action_probs_vec, 0))  # Valeur temporaire

        return action

    def train(self, winner):
        # Mettre à jour les valeurs dans les données d'entraînement
        for i in range(len(self.training_data)):
            state_desc, action_probs, _ = self.training_data[i]
            # Du point de vue du joueur actuel
            value = winner if i % 2 == 0 else -winner
            self.training_data[i] = (state_desc, action_probs, value)

        # Préparer les données
        states = np.array([d[0] for d in self.training_data])
        target_policies = np.array([d[1] for d in self.training_data])
        target_values = np.array([d[2] for d in self.training_data])

        # Entraîner le modèle
        self.model.fit(states, {'policy': target_policies, 'value': target_values},
                       batch_size=32, epochs=1, verbose=0)

        # Vider les données d'entraînement
        self.training_data = []

    def _softmax(self, x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def save(self, path="models/alphazero.h5"):
        self.model.save(path)

    def load(self, path="models/alphazero.h5"):
        self.model = models.load_model(path)
