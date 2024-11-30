import numpy as np
import threading

class Node:
    def __init__(self, state, parent=None, prior_prob=1.0, action=None):
        self.state = state  # État de l'environnement à ce nœud
        self.parent = parent
        self.action = action  # Action qui a mené à ce nœud depuis le parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob
        self.lock = threading.Lock()  # Pour les opérations thread-safe

    def is_leaf(self):
        # Pas besoin de verrou ici car nous ne modifions pas les enfants
        return len(self.children) == 0

    def expand(self, action_probs):
        with self.lock:
            for action, prob in action_probs.items():
                if action not in self.children:
                    next_state = self.state.clone()
                    next_state.step(action)
                    self.children[action] = Node(
                        state=next_state,
                        parent=self,
                        prior_prob=prob,
                        action=action  # Stocker l'action menant à cet enfant
                    )

    def update(self, value):
        with self.lock:
            self.visit_count += 1
            self.value_sum += value

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select(self, c_puct):
        """
        Sélectionne l'enfant avec le score UCB le plus élevé.
        """
        best_score = -np.inf
        best_child = None

        total_visits = sum(child.visit_count for child in self.children.values()) + 1

        for child in self.children.values():
            q_value = child.value()
            u_value = c_puct * child.prior_prob * np.sqrt(total_visits) / (1 + child.visit_count)
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_child = child

        return best_child



import numpy as np
import random

class MCTSWithRandomRollouts:
    def __init__(self, env, n_simulations=1000, c_puct=1.4):
        self.env = env
        self.n_simulations = n_simulations
        self.c_puct = c_puct

    def choose_action(self):
        root = Node(state=self.env.clone())
        for _ in range(self.n_simulations):
            node = root
            state = self.env.clone()

            # Sélection
            path = []
            while not node.is_leaf() and not state.is_game_over():
                node = node.select(self.c_puct)
                action = node.action
                state.step(action)
                path.append(node)

            # Expansion
            if not state.is_game_over():
                valid_actions = state.available_actions_ids()
                # Probabilités uniformes pour les actions valides
                prob = 1.0 / len(valid_actions)
                action_probs = {action: prob for action in valid_actions}
                node.expand(action_probs)

                # Simulation par random rollout
                reward = self.random_rollout(state)
            else:
                reward = state.score()

            # Rétropropagation
            self.backpropagate(node, reward)

        # Choisir l'action avec le plus de visites depuis la racine
        best_child = max(root.children.values(), key=lambda n: n.visit_count)
        return best_child.action

    def random_rollout(self, state):
        """Effectue un rollout aléatoire depuis l'état donné jusqu'à la fin du jeu."""
        current_state = state.clone()
        while not current_state.is_game_over():
            actions = current_state.available_actions_ids()
            action = random.choice(actions)
            current_state.step(action)
        reward = current_state.score()
        return reward

    def backpropagate(self, node, reward):
        """Met à jour les nœuds le long du chemin avec le résultat du rollout."""
        while node is not None:
            node.update(reward)
            reward = -reward  # Inversion du résultat pour l'adversaire
            node = node.parent
