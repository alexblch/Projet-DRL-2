import numpy as np
import random
from collections import deque
from neuralnetwork import create_model  # Assurez-vous que le fichier est correctement importé

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma    # Discount rate
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min  # Minimal exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.batch_size = batch_size
        self.model = create_model((self.state_size,), self.action_size)  # Réseau principal avec mse
        self.target_model = create_model((self.state_size,), self.action_size)  # Réseau cible avec mse
        self.update_target_model()  # Initialisation du modèle cible

    def update_target_model(self):
        """Copie les poids du modèle principal dans le modèle cible"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Stocke les expériences dans la mémoire"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, env):
        """Choisit une action en fonction de la politique d'exploration/exploitation."""
        for _ in range(self.action_size):  # Essayer jusqu'à trouver une action valide
            if np.random.rand() <= self.epsilon:
                action = random.randrange(self.action_size)  # Choix aléatoire (exploration)
            else:
                act_values = self.model.predict(state)
                action = np.argmax(act_values[0])  # Action avec la plus haute valeur Q (exploitation)

            row, col = divmod(action, env.cols)
            trefle_number = env.get_random_trefle()
            if env.is_valid_action(row, col, trefle_number):  # Vérifie si l'action est valide
                return action  # Si l'action est valide, la retourne

        # Si aucune action valide n'est trouvée (peu probable), choisir au hasard
        return random.randrange(self.action_size)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        
        # Extraire les états et les états suivants
        states = np.array([transition[0] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])

        # Suppression des dimensions supplémentaires
        states = np.squeeze(states, axis=1)  # Reformater en (batch_size, state_size)
        next_states = np.squeeze(next_states, axis=1)  # Reformater en (batch_size, state_size)

        # Prédictions pour les états actuels et suivants
        targets = self.model.predict(states)
        target_next = self.target_model.predict(next_states)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            # Créer les cibles pour l'action choisie
            if done:
                targets[i][action] = reward
            else:
                targets[i][action] = reward + self.gamma * np.amax(target_next[i])

        # Entraînement du modèle sur le batch entier
        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """Charge les poids d'un modèle"""
        self.model.load_weights(name)

    def save(self, name):
        """Sauvegarde les poids du modèle"""
        self.model.save_weights(name)

    def train(self, episodes=1000, update_target_every=10):
        """Entraîne l'agent sur un nombre spécifié d'épisodes"""
        for e in range(episodes):
            state = np.reshape(env.reset(), [1, self.state_size])
            total_reward = 0
            done = False
            for time in range(500):  # Limiter le nombre de tours par épisode
                action = self.act(state, env)  # Choix de l'action par l'agent
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                # Mémoriser cette transition
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    print(f"Episode {e+1}/{episodes} - Reward: {total_reward}, Epsilon: {self.epsilon}")
                    break

                # Entraînement périodique avec les expériences stockées
                self.replay()

            # Mise à jour du modèle cible tous les 'update_target_every' épisodes
            if e % update_target_every == 0:
                self.update_target_model()
