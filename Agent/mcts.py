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
        self.untried_actions = state.available_actions_ids().tolist()  # Initialiser ici

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.clone()
        next_state.step(action)
        child_node = Node(state=next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def best_child(self, c_param=1.4):
        total_visits = self.visits
        choices_weights = [
            (child.value / child.visits) +
            c_param * np.sqrt(2 * np.log(total_visits) / child.visits)
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def update(self, reward):
        self.visits += 1
        self.value += reward

class MCTS:
    def __init__(self, env, n_iterations=1000, c_param=1.4):
        self.env = env
        self.n_iterations = n_iterations
        self.c_param = c_param

    def choose_action(self):
        root = Node(state=self.env.clone())

        for _ in range(self.n_iterations):
            node = root
            state = self.env.clone()

            # Sélection
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.c_param)
                state.step(node.action)

            # Expansion
            if not state.is_game_over():
                node = node.expand()
                state.step(node.action)

            # Simulation
            sim_state = state.clone()
            while not sim_state.is_game_over():
                actions = sim_state.available_actions_ids()
                action = random.choice(actions)
                sim_state.step(action)

            # Rétropropagation
            reward = sim_state.score()
            while node is not None:
                node.update(reward)
                reward = -reward  # Inversion du résultat pour l'adversaire
                node = node.parent

        # Retourne l'action avec le plus grand nombre de visites
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action
