import numpy as np
from tensorflow.keras import layers, models



class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Paramètres pour la stratégie epsilon-greedy
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Créer le modèle de réseau neuronal
        self.model = self._build_model()

    def _build_model(self):
        """Construit le réseau neuronal DQN."""
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def choose_action(self, state, available_actions, action_mask):
        """Choisir une action avec stratégie epsilon-greedy."""
        if np.random.rand() < self.epsilon:
            # Choisir une action aléatoire (exploration)
            return np.random.choice(available_actions)
        else:
            # Exploitation : choisir l'action avec la plus grande valeur Q parmi les actions disponibles
            state = np.expand_dims(state, axis=0)
            q_values = self.model.predict(state, verbose=0)[0]
            masked_q_values = np.copy(q_values)
            masked_q_values[action_mask == 0] = -np.inf  # Ignorer les actions non disponibles
            return np.argmax(masked_q_values)

    def learn(self, state, action, reward, next_state, done, action_mask_next):
        """Apprentissage basé sur la transition courante."""
        # Calculer la valeur cible (target)
        target = reward
        if not done:
            next_state = np.expand_dims(next_state, axis=0)
            target += self.gamma * np.max(self.model.predict(next_state, verbose=0)[0])

        # Appliquer la mise à jour des Q-valeurs
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state, verbose=0)
        q_values[0][action] = target

        # Entraîner le modèle sur cet exemple spécifique
        self.model.fit(state, q_values, epochs=1, verbose=0)

        # Réduire epsilon (stratégie epsilon-greedy)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
