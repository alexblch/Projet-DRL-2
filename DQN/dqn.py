from collections import deque
from DQN.neuralnetwork import create_model
import numpy as np
import random

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Mémoire de rejouabilité avec taille limitée
        self.gamma = gamma    # Discount rate
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min  # Minimal exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.batch_size = batch_size
        self.model = create_model((self.state_size,), self.action_size)  # Réseau principal
        self.target_model = create_model((self.state_size,), self.action_size)  # Réseau cible
        self.update_target_model()  # Initialisation du modèle cible

    def remember(self, state, action, reward, next_state, done):
        """Stocke les expériences dans la mémoire"""
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        """Copie les poids du modèle principal dans le modèle cible"""
        self.target_model.set_weights(self.model.get_weights())

    def save(self, name):
        """Sauvegarde les poids du modèle"""
        self.model.save_weights(name)

    def load(self, name):
        """Charge les poids du modèle"""
        self.model.load_weights(name)


    def act(self, state, env):
        """Pas de Numba ici, car cela implique l'usage de modèles et d'environnements complexes"""
        for _ in range(self.action_size):
            if np.random.rand() <= self.epsilon:
                action = random.randrange(self.action_size)
            else:
                act_values = self.model.predict(state)
                action = np.argmax(act_values[0])

            row, col = divmod(action, env.cols)
            trefle_number = env.get_random_trefle()
            if env.is_valid_action(row, col, trefle_number):
                return action

        return random.randrange(self.action_size)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([transition[0] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])

        states = np.squeeze(states, axis=1)
        next_states = np.squeeze(next_states, axis=1)

        targets = self.model.predict(states)
        target_next = self.target_model.predict(next_states)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                targets[i][action] = reward
            else:
                targets[i][action] = reward + self.gamma * np.amax(target_next[i])

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
