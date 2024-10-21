import numpy as np
import random
from tensorflow.keras import layers, models


from Environnements.grid import GridWorld


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # Facteur de discount pour les récompenses futures

        # Paramètres pour la stratégie epsilon-greedy
        self.epsilon = epsilon_start  # Valeur initiale de epsilon
        self.epsilon_min = epsilon_end  # Valeur minimale de epsilon
        self.epsilon_decay = epsilon_decay  # Taux de décroissance de epsilon

        # Créer le modèle de réseau neuronal
        self.model = self._build_model()

    def _build_model(self):
        """Construire un modèle simple de réseau neuronal pour le DQN."""
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def choose_action(self, state, available_actions):
        """Choisir une action avec la stratégie epsilon-greedy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(available_actions)
        else:
            state = np.expand_dims(state, axis=0)
            q_values = self.model.predict(state, verbose=0)[0]
            masked_q_values = np.copy(q_values)
            masked_q_values[~np.isin(range(self.action_size), available_actions)] = -np.inf
            return np.argmax(masked_q_values)

    def learn(self, state, action, reward, next_state, done):
        """Entraîne le modèle avec une seule transition."""
        target = reward
        if not done:
            next_state = np.expand_dims(next_state, axis=0)
            target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])

        state = np.expand_dims(state, axis=0)
        target_f = self.model.predict(state, verbose=0)
        target_f[0][action] = target

        self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(episodes=1000):
    env = GridWorld()  # Initialiser l'environnement GridWorld
    state_size = env.reset().shape[0]  # Taille de l'état (taille de la grille aplatie)
    action_size = env.action_space  # Nombre d'actions possibles
    agent = DQNAgent(state_size, action_size)

    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Obtenir les actions disponibles à partir de la position actuelle
            available_actions = env.available_actions()

            # Choisir une action avec la stratégie epsilon-greedy
            action = agent.choose_action(state, available_actions)

            # Exécuter l'action dans l'environnement
            next_state, reward, done, _ = env.step(action)

            # Entraîner l'agent avec la transition actuelle
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            # Afficher la grille et les récompenses pour mieux comprendre
            env.render()

        env.list_scores.append(total_reward)
        print(f"Épisode {e + 1}/{episodes}, Récompense totale: {total_reward}, Epsilon: {agent.epsilon}")

    # Afficher le graphique des scores après l'entraînement
    env.graph_scores()

# Lancer l'entraînement
train_dqn(episodes=1000)
