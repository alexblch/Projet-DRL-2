import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class MuZeroNode:
    def __init__(self, parent, prior_prob, hidden_state):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior_prob = prior_prob
        self.hidden_state = hidden_state  # État caché du modèle dynamique

    def is_leaf(self):
        return len(self.children) == 0

    def value(self):
        if self.visit_count == 0:
            return 0
        else:
            return self.value_sum / self.visit_count

class MuZeroAgent:
    def __init__(self, env, n_simulations=100, c_puct=1.0):
        self.env = env
        self.state_size = env.state_description().shape[0]
        self.action_size = env.TOTAL_ACTIONS
        self.n_simulations = n_simulations
        self.c_puct = c_puct

        # Initialiser les réseaux de neurones
        self.representation_model = self.create_representation_model()
        self.dynamics_model = self.create_dynamics_model()
        self.prediction_model = self.create_prediction_model()

        # Données pour l'entraînement
        self.training_data = []

        # Optimiseur
        self.optimizer = optimizers.Adam(learning_rate=0.001)

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

    def create_representation_model(self):
        # Convertit l'état initial en état caché
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(128, activation='relu')(inputs)
        hidden_state = layers.Dense(128, activation='relu')(x)
        model = models.Model(inputs=inputs, outputs=hidden_state)
        return model

    def create_dynamics_model(self):
        # Prend l'état caché et une action, et prédit le nouvel état caché et la récompense
        hidden_inputs = layers.Input(shape=(128,))
        action_inputs = layers.Input(shape=(self.action_size,))
        x = layers.concatenate([hidden_inputs, action_inputs])
        x = layers.Dense(128, activation='relu')(x)
        next_hidden_state = layers.Dense(128, activation='relu')(x)
        reward = layers.Dense(1, activation='linear')(x)
        model = models.Model(inputs=[hidden_inputs, action_inputs], outputs=[next_hidden_state, reward])
        return model

    def create_prediction_model(self):
        # Prend l'état caché et prédit la politique et la valeur
        hidden_inputs = layers.Input(shape=(128,))
        x = layers.Dense(128, activation='relu')(hidden_inputs)
        policy_logits = layers.Dense(self.action_size, activation='linear')(x)
        value = layers.Dense(1, activation='tanh')(x)
        model = models.Model(inputs=hidden_inputs, outputs=[policy_logits, value])
        return model

    def choose_action(self):
        state = self.env.clone()
        state_desc = state.state_description()

        # Obtenir l'état caché initial
        hidden_state = self.representation_model.predict(np.array([state_desc]), verbose=0)[0]

        root = MuZeroNode(parent=None, prior_prob=1.0, hidden_state=hidden_state)

        # Obtenir la politique initiale du modèle de prédiction
        policy_logits, _ = self.prediction_model.predict(np.array([hidden_state]), verbose=0)
        policy_probs = self.softmax(policy_logits[0])
        valid_actions = state.available_actions_ids()
        policy = np.zeros(self.action_size)
        policy[valid_actions] = policy_probs[valid_actions]
        policy /= np.sum(policy[valid_actions])

        for a in valid_actions:
            root.children[a] = MuZeroNode(parent=root, prior_prob=policy[a], hidden_state=None)

        # Simulations MCTS
        for _ in range(self.n_simulations):
            node = root
            search_path = [node]

            # Sélection
            while not node.is_leaf():
                action, node = self.select_child(node)
                search_path.append(node)

            # Évaluation et expansion
            if node.hidden_state is None:
                action_one_hot = np.zeros(self.action_size)
                action_one_hot[action] = 1
                # Utiliser le modèle dynamique pour obtenir le nouvel état caché et la récompense
                next_hidden_state, reward = self.dynamics_model.predict([np.array([node.parent.hidden_state]), np.array([action_one_hot])], verbose=0)
                next_hidden_state = next_hidden_state[0]
                reward = reward[0][0]
                node.hidden_state = next_hidden_state
                # Obtenir la politique et la valeur
                policy_logits, value = self.prediction_model.predict(np.array([next_hidden_state]), verbose=0)
                policy_probs = self.softmax(policy_logits[0])
                # Ici, nous devrions mettre à jour l'état du jeu, mais pour simplifier, nous supposons que les actions sont toujours valides
                valid_actions = self.env.available_actions_ids()
                policy = np.zeros(self.action_size)
                policy[valid_actions] = policy_probs[valid_actions]
                policy /= np.sum(policy[valid_actions])

                # Expansion
                for a in valid_actions:
                    node.children[a] = MuZeroNode(parent=node, prior_prob=policy[a], hidden_state=None)
            else:
                value = 0  # Valeur terminale

            # Rétropropagation
            self.backpropagate(search_path, value)

        # Choisir l'action avec le plus grand nombre de visites
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

    def backpropagate(self, search_path, value):
        # Mettre à jour les valeurs des nœuds et propager vers le haut
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Pour les jeux à somme nulle à deux joueurs

    def train(self, winner):
        # Pour simplifier, nous n'implémenterons pas l'entraînement complet de MuZero ici
        # Entraînement des modèles basé sur les données collectées
        pass

    def save(self, path="models/muzero"):
        self.representation_model.save(path + "_representation.h5")
        self.dynamics_model.save(path + "_dynamics.h5")
        self.prediction_model.save(path + "_prediction.h5")

    def load(self, path="models/muzero"):
        self.representation_model = models.load_model(path + "_representation.h5")
        self.dynamics_model = models.load_model(path + "_dynamics.h5")
        self.prediction_model = models.load_model(path + "_prediction.h5")

    def softmax(self, x):
        x = x - np.max(x)  # Pour la stabilité numérique
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
