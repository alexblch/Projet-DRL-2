import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class AlphaZeroNode:
    def __init__(self, parent, prior_prob):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior_prob = prior_prob

    def is_leaf(self):
        return len(self.children) == 0

    def value(self):
        if self.visit_count == 0:
            return 0
        else:
            return self.value_sum / self.visit_count

class AlphaZeroAgent:
    def __init__(self, env, n_simulations=100, c_puct=1.0):
        self.env = env
        self.state_size = env.state_description().shape[0]
        self.action_size = env.TOTAL_ACTIONS
        self.n_simulations = n_simulations
        self.c_puct = c_puct

        # Données pour l'entraînement
        self.training_data = []

        # Optimiseur
        self.optimizer = optimizers.Adam(learning_rate=0.001)

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
                print(f"Utilisation du GPU : {physical_devices}")
            except RuntimeError as e:
                print(e)
        else:
            print("Aucun GPU trouvé. Utilisation du CPU.")

    def create_model(self):
        # Modèle simple avec couches partagées
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(128, activation='relu')(x)

        # Tête de politique
        policy_logits = layers.Dense(self.action_size, activation='linear', name='policy')(x)

        # Tête de valeur
        value = layers.Dense(1, activation='tanh', name='value')(x)

        model = models.Model(inputs=inputs, outputs=[policy_logits, value])
        model.compile(optimizer=self.optimizer, loss=['categorical_crossentropy', 'mean_squared_error'])
        return model

    def choose_action(self):
        root = AlphaZeroNode(parent=None, prior_prob=1.0)
        state = self.env.clone()

        # Obtenir la politique et la valeur initiales du réseau
        state_desc = state.state_description()
        policy_logits, _ = self.model.predict(np.array([state_desc]), verbose=0)
        policy_probs = self.softmax(policy_logits[0])
        valid_actions = state.available_actions_ids()
        policy = np.zeros(self.action_size)
        policy[valid_actions] = policy_probs[valid_actions]
        policy /= np.sum(policy[valid_actions])

        for a in valid_actions:
            root.children[a] = AlphaZeroNode(parent=root, prior_prob=policy[a])

        # Simulations MCTS
        for _ in range(self.n_simulations):
            node = root
            state_sim = state.clone()

            # Sélection et expansion
            while not node.is_leaf():
                action, node = self.select_child(node)
                state_sim.step(action)

            # Évaluer le nœud feuille
            state_desc = state_sim.state_description()
            policy_logits, value = self.model.predict(np.array([state_desc]), verbose=0)
            policy_probs = self.softmax(policy_logits[0])

            valid_actions = state_sim.available_actions_ids()
            policy = np.zeros(self.action_size)
            policy[valid_actions] = policy_probs[valid_actions]
            policy /= np.sum(policy[valid_actions])

            # Vérifier si l'état est terminal
            if state_sim.is_game_over():
                leaf_value = state_sim.score()
            else:
                leaf_value = value[0][0]

            # Expansion
            for a in valid_actions:
                node.children[a] = AlphaZeroNode(parent=node, prior_prob=policy[a])

            # Rétropropagation
            self.backpropagate(node, leaf_value)

        # Choisir l'action
        # Pour l'entraînement, nous utilisons les visites comme politique cible
        action_visits = np.zeros(self.action_size)
        for action, child in root.children.items():
            action_visits[action] = child.visit_count

        action_probs = action_visits / np.sum(action_visits)

        # Stocker les données pour l'entraînement
        self.training_data.append((state_desc, action_probs, 0))  # La valeur sera mise à jour après la fin du jeu

        # Choisir l'action avec le plus grand nombre de visites
        best_action = np.argmax(action_visits)
        return best_action

    def select_child(self, node):
        # Utiliser la formule PUCT
        total_visit = np.sum([child.visit_count for child in node.children.values()]) + 1
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            q_value = child.value()
            u_value = self.c_puct * child.prior_prob * np.sqrt(total_visit) / (1 + child.visit_count)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def backpropagate(self, node, value):
        # Mettre à jour les valeurs des nœuds et propager vers le haut
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Pour les jeux à somme nulle à deux joueurs
            node = node.parent

    def train(self, winner):
        # winner: 1 si l'agent a gagné, -1 s'il a perdu, 0 pour un match nul
        # Mettre à jour la valeur dans les données d'entraînement
        for i in range(len(self.training_data)):
            state_desc, action_probs, _ = self.training_data[i]
            # Du point de vue du joueur actuel
            value = winner
            if i % 2 == 1:
                value = -winner
            self.training_data[i] = (state_desc, action_probs, value)

        # Préparer les données
        states = np.array([d[0] for d in self.training_data])
        target_policies = np.array([d[1] for d in self.training_data])
        target_values = np.array([d[2] for d in self.training_data])

        # Entraîner le réseau
        self.model.fit(states, [target_policies, target_values], epochs=1, verbose=0)

        # Vider les données d'entraînement
        self.training_data = []

    def save(self, path="models/alphazero.h5"):
        self.model.save(path)

    def load(self, path="models/alphazero.h5"):
        self.model = models.load_model(path)

    def softmax(self, x):
        x = x - np.max(x)  # Pour la stabilité numérique
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
