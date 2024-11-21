# Agent/neuralmcts.py

import numpy as np
from .mcts import MCTSNode
import tensorflow as tf
from tensorflow.keras import layers, models

class NeuralMCTSAgent:
    def __init__(self, env, n_iterations=1000):
        self.env = env
        self.n_iterations = n_iterations
        self.memory = []
        self.policy_network = self.build_policy_network()
        self.value_network = self.build_value_network()
        self.root = None

    def build_policy_network(self):
        model = models.Sequential()
        model.add(layers.Dense(128, activation='relu', input_shape=(self.env.observation_space.shape[0],)))
        model.add(layers.Dense(self.env.action_space.n, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model

    def build_value_network(self):
        model = models.Sequential()
        model.add(layers.Dense(128, activation='relu', input_shape=(self.env.observation_space.shape[0],)))
        model.add(layers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def initialize_tree(self, state):
        self.root = MCTSNode(state, None)

    def choose_action(self, env):
        for _ in range(self.n_iterations):
            node = self.tree_policy(self.root)
            reward = self.default_policy(node.state)
            self.backup(node, reward)

        # Choisir l'action avec le plus grand nombre de visites
        action = max(self.root.children.items(), key=lambda item: item[1].visit_count)[0]
        return action

    def tree_policy(self, node):
        while not self.is_terminal(node.state):
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = node.best_child()
        return node

    def expand(self, node):
        action = node.untried_actions.pop()
        next_state, reward, done, _ = self.env.step(action)
        child_node = MCTSNode(next_state, parent=node)
        node.children[action] = child_node
        return child_node

    def best_child(self, node, c_param=1.4):
        choices_weights = [
            (child.q_value / child.visit_count) + c_param * np.sqrt((2 * np.log(node.visit_count) / child.visit_count))
            for child in node.children.values()
        ]
        return list(node.children.values())[np.argmax(choices_weights)]

    def default_policy(self, state):
        # Utiliser le réseau de valeur pour évaluer l'état
        state_array = np.array(state).reshape(1, -1)
        value = self.value_network.predict(state_array)[0][0]
        return value

    def backup(self, node, reward):
        while node is not None:
            node.visit_count += 1
            node.q_value += reward
            node = node.parent

    def is_terminal(self, state):
        # Vérifier si l'état est terminal
        return self.env.is_game_over()

    def train(self):
        if not self.memory:
            print("Pas de données pour l'entraînement.")
            return

        # Exemple d'entraînement basique
        states, actions, rewards, next_states, dones = zip(*self.memory)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Entraînement du réseau de valeur
        self.value_network.fit(states, rewards, epochs=1, verbose=0)

        # Entraînement du réseau de politique
        # Convertir les actions en one-hot si nécessaire
        actions_one_hot = tf.keras.utils.to_categorical(actions, num_classes=self.env.action_space.n)
        self.policy_network.fit(states, actions_one_hot, epochs=1, verbose=0)

        # Vider la mémoire après l'entraînement
        self.memory = []
