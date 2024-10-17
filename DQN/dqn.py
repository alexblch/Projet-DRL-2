from collections import deque
from models.neuralnetwork import create_model
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_size = state_size  # Dimension du vecteur d'état
        self.action_size = action_size  # Nombre total d'actions possibles
        self.learning_rate = learning_rate
        self.gamma = gamma  # Facteur de discount pour les récompenses futures

        # Paramètres pour la stratégie epsilon-greedy
        self.epsilon = epsilon_start  # Valeur initiale de epsilon
        self.epsilon_min = epsilon_end  # Valeur minimale de epsilon
        self.epsilon_decay = epsilon_decay  # Taux de décroissance de epsilon

        # Construire le modèle de réseau neuronal
        self.model = self.create_model(input_shape=(self.state_size,), action_space=self.action_size, learning_rate=self.learning_rate)

    def create_model(self, input_shape, action_space, layer_sizes=[128, 128], learning_rate=0.001):
        """
        Crée un modèle de réseau neuronal pour DQN classique avec MSE comme fonction de perte.
        
        :param input_shape: La forme de l'état (ex: (16,) pour une grille 4x4).
        :param action_space: Nombre d'actions possibles (taille de l'espace d'actions).
        :param layer_sizes: Liste définissant le nombre de neurones dans chaque couche cachée.
        :param learning_rate: Taux d'apprentissage pour l'optimiseur Adam.
        :return: Modèle Keras compilé.
        """
        model = tf.keras.Sequential()
        
        # Couche d'entrée
        model.add(layers.InputLayer(input_shape=input_shape))
        model.add(layers.Dense(layer_sizes[0], activation='relu'))
        model.add(layers.Dense(layer_sizes[1], activation='relu'))
        # Si nécessaire, vous pouvez ajouter une troisième couche cachée :
        # model.add(layers.Dense(layer_sizes[0], activation='relu'))
        
        # Couche de sortie pour les valeurs Q
        model.add(layers.Dense(action_space, activation='linear'))
    
        # Compilation du modèle avec Adam
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mean_squared_error')
        
        return model

    def choose_action(self, state, available_actions, action_mask):
        if np.random.rand() < self.epsilon:
            # Exploration : choisir une action disponible au hasard
            action = np.random.choice(available_actions)
        else:
            # Exploitation : choisir l'action avec la valeur Q la plus élevée parmi les actions disponibles
            state = np.expand_dims(state, axis=0)
            q_values = self.model.predict(state, verbose=0)[0]
            # Appliquer le masque d'actions
            masked_q_values = np.copy(q_values)
            masked_q_values[action_mask == 0] = -np.inf  # Ignorer les actions non disponibles
            action = np.argmax(masked_q_values)
        return action


    def learn(self, state, action, reward, next_state, done, action_mask_next):
        """
        Met à jour les poids du réseau neuronal basé sur la transition.

        Paramètres:
            state (np.ndarray): L'état précédent.
            action (int): L'action prise.
            reward (float): La récompense reçue.
            next_state (np.ndarray): L'état suivant après l'action.
            done (bool): True si l'épisode est terminé.
            action_mask_next (np.ndarray): Masque des actions pour l'état suivant.
        """
        state = np.expand_dims(state, axis=0)  # Ajouter la dimension batch
        next_state = np.expand_dims(next_state, axis=0)

        # Obtenir les valeurs Q prédites pour l'état actuel
        q_values = self.model.predict(state, verbose=0)

        # Obtenir les valeurs Q prédites pour l'état suivant
        q_values_next = self.model.predict(next_state, verbose=0)

        # Masquer les actions non disponibles dans l'état suivant
        masked_q_values_next = q_values_next + (action_mask_next - 1) * 1e9

        # Calculer la valeur cible Q
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(masked_q_values_next[0])

        # Mettre à jour la valeur Q pour l'action prise
        q_values[0][action] = target

        # Entraîner le réseau
        self.model.fit(state, q_values, epochs=1, verbose=0)

        # Décroître epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
