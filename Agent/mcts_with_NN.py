import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers

class Node:
    __slots__ = ['state', 'parent', 'action', 'children', 'visits', 'value', 'untried_actions']

    def __init__(self, state, parent=None, action=None):
        self.state = state  # État de l'environnement à ce nœud
        self.parent = parent
        self.action = action  # Action qui a mené à cet état
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = state.available_actions_ids().tolist()  # Initialiser ici

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.clone()
        next_state.step(action)
        child = Node(state=next_state, parent=self, action=action)
        self.children.append(child)
        return child

    def best_child(self, c_param=1.4):
        total_visits = self.visits
        sqrt_total_visits = np.sqrt(total_visits)
        best_score = -np.inf
        best_child = None

        for child in self.children:
            exploitation = child.value / child.visits
            exploration = c_param * sqrt_total_visits / (1 + child.visits)
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def update(self, reward):
        self.visits += 1
        self.value += reward

class MCTSWithNN:
    def __init__(self, env, n_iterations=1000, max_depth=10):
        self.env = env
        self.n_iterations = n_iterations
        self.max_depth = max_depth
        self.value_model = self.create_value_model(input_shape=(env.state_description().shape[0],))
        self.configure_gpu()

    def configure_gpu(self):
        # Configuration du GPU une seule fois lors de l'initialisation
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    def create_value_model(self, input_shape, layer_sizes=[128, 128]):
        inputs = layers.Input(shape=input_shape)
        x = inputs
        for size in layer_sizes:
            x = layers.Dense(size, activation='relu')(x)
        outputs = layers.Dense(1, activation='linear')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error'
        )
        return model

    def choose_action(self):
        root = Node(state=self.env.clone())

        for _ in range(self.n_iterations):
            node = root
            state = self.env.clone()
            depth = 0

            # Sélection et expansion
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
                state.step(node.action)
                depth += 1

            if not state.is_game_over() and depth < self.max_depth:
                if not node.is_fully_expanded():
                    node = node.expand()
                    state.step(node.action)
                    depth += 1

            # Simulation avec réseau de neurones
            reward = self.evaluate_state(state)

            # Rétropropagation
            while node is not None:
                node.update(reward)
                reward = -reward  # Inversion pour l'adversaire
                node = node.parent

        # Retourne l'action avec le plus grand nombre de visites
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    def evaluate_state(self, state):
        """Évalue l'état en utilisant le modèle de valeur."""
        state_desc = state.state_description()
        state_tensor = tf.convert_to_tensor([state_desc], dtype=tf.float32)
        value = self.value_model(state_tensor, training=False)[0][0].numpy()
        return value

    def train_value_model(self, states, targets, epochs=10):
        """Entraîne le modèle de valeur."""
        states = np.array(states, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        self.value_model.fit(states, targets, epochs=epochs, verbose=0)  # Désactiver l'affichage pour la vitesse
