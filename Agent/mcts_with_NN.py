import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # État de l'environnement à ce nœud
        self.parent = parent
        self.action = action  # Action qui a mené à cet état
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def expand(self):
        if self.untried_actions is None:
            self.untried_actions = self.state.available_actions_ids().tolist()

    def add_child(self, action, next_state):
        child = Node(state=next_state, parent=self, action=action)
        self.children.append(child)
        self.untried_actions.remove(action)
        return child

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visits) +
            c_param * np.sqrt(2 * np.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

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
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU: {physical_devices}")
            except RuntimeError as e:
                print(e)
        else:
            print("No GPU found. Using CPU.")

    def create_value_model(self, input_shape, layer_sizes=[64, 64]):
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

    def choose_action(self, env):
        root = Node(state=env.clone())
        for _ in range(self.n_iterations):
            node = root
            state = env.clone()
            depth = 0

            # Sélection
            while not state.is_game_over() and node.is_fully_expanded() and node.children:
                node = node.best_child()
                state = node.state  # Synchronisation de l'état
                depth += 1

            # Expansion
            if not state.is_game_over() and depth < self.max_depth:
                node.expand()
                if node.untried_actions:
                    action = random.choice(node.untried_actions)
                    next_state = state.clone()
                    next_state.step(action)
                    node = node.add_child(action, next_state)
                    state = next_state  # Mettre à jour l'état pour la simulation

            # Simulation avec réseau de neurones
            reward = self.evaluate_state(state)

            # Rétropropagation
            while node is not None:
                node.update(reward)
                node = node.parent

        # Retourne l'action avec le plus grand nombre de visites
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    def evaluate_state(self, state):
        """Évalue l'état en utilisant le modèle de valeur."""
        state_desc = state.state_description()
        state_tensor = tf.convert_to_tensor([state_desc], dtype=tf.float32)
        value = self.value_model(state_tensor)[0][0].numpy()
        return value

    def train_value_model(self, states, targets, epochs=10):
        """Entraîne le modèle de valeur."""
        states = np.array(states, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        self.value_model.fit(states, targets, epochs=epochs, verbose=1)
