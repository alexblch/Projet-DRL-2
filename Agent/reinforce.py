import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class REINFORCEAgent:
    def __init__(self, env, learning_rate=0.001, gamma=0.99):
        self.env = env
        self.state_size = env.state_description().shape[0]
        self.action_size = env.action_mask().shape[0]
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Modèle de politique pour l'agent
        self.model = self.create_model(input_shape=(self.state_size,), action_space=self.action_size, learning_rate=self.learning_rate)
        
        # Listes pour stocker les transitions de l'épisode
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    def create_model(self, input_shape, action_space, layer_sizes=[128, 128], learning_rate=0.001):
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=input_shape))
        model.add(layers.Dense(layer_sizes[0], activation='relu'))
        model.add(layers.Dense(layer_sizes[1], activation='relu'))
        model.add(layers.Dense(action_space, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy')
        return model

    def choose_action(self, state):
        state = np.expand_dims(state, axis=0)
        action_probs = self.model.predict(state, verbose=0)[0]
        
        # Appliquer le masque d'action
        mask = self.env.action_mask()
        action_probs *= mask  # Mettre à zéro les probabilités des actions non valides
        action_probs /= np.sum(action_probs)  # Normaliser pour obtenir une distribution valide

        # Choisir une action en fonction des probabilités filtrées
        action = np.random.choice(self.action_size, p=action_probs)
        return action

    def store_transition(self, state, action, reward):
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def discount_rewards(self):
        discounted_rewards = np.zeros_like(self.episode_rewards, dtype=np.float32)
        cumulative = 0
        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * self.gamma + self.episode_rewards[t]
            discounted_rewards[t] = cumulative

        # Normalisation des récompenses pour la stabilité
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= (np.std(discounted_rewards) + 1e-8)
        return discounted_rewards

    def train(self):
        discounted_rewards = self.discount_rewards()
        states = np.array(self.episode_states)
        actions = np.array(self.episode_actions)
        advantages = discounted_rewards

        # Créer un masque d'actions pour entraîner la politique uniquement sur les actions choisies
        action_masks = np.zeros((len(actions), self.action_size))
        action_masks[np.arange(len(actions)), actions] = 1

        # Entraîner le modèle en pondérant les actions par les avantages
        self.model.fit(states, action_masks, sample_weight=advantages, epochs=1, verbose=0)

        # Réinitialiser les trajectoires après l'entraînement
        self.episode_states, self.episode_actions, self.episode_rewards = [], [], []

    def run_episode(self):
        state = self.env.state_description()
        self.env.reset()
        done = False
        total_reward = 0

        while not done:
            action = self.choose_action(state)
            try:
                self.env.step(action)
            except ValueError as e:
                print(f"Action invalide: {e}")
                done = True
                reward = -1
                self.store_transition(state, action, reward)
                break

            # Récupérer les informations après avoir pris une action
            next_state = self.env.state_description()
            reward = self.env.score()
            done = self.env.is_game_over()
            self.store_transition(state, action, reward)

            # Passer à l'état suivant
            state = next_state
            total_reward += reward

        # Entraîner l'agent à la fin de chaque épisode
        self.train()
        print(f"Total Reward for Episode: {total_reward}")
