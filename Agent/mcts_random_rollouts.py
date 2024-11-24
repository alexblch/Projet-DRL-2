import numpy as np
import random

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

class MCTSWithRandomRollouts:
    def __init__(self, n_iterations=1000):
        self.n_iterations = n_iterations

    def choose_action(self, env):
        root = Node(state=env.clone())
        for _ in range(self.n_iterations):
            node = root
            state = env.clone()

            # Sélection
            while not state.is_game_over() and node.is_fully_expanded() and node.children:
                node = node.best_child()
                state = node.state  # Synchronisation de l'état

            # Expansion
            if not state.is_game_over():
                node.expand()
                if node.untried_actions:
                    action = random.choice(node.untried_actions)
                    next_state = state.clone()
                    next_state.step(action)
                    node = node.add_child(action, next_state)
                    state = next_state  # Mettre à jour l'état pour la simulation

            # Simulation par Random Rollout
            reward = self.random_rollout(state)

            # Rétropropagation
            while node is not None:
                node.update(reward)
                node = node.parent

        # Retourne l'action avec le plus grand nombre de visites
        best_child = max(root.children, key=lambda c: c.visits)
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
