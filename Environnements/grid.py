import numpy as np
from numba import njit

class GridWorld:
    # Définition des actions
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    TOTAL_ACTIONS = 4  # Total number of possible actions

    def __init__(self):
        self.rows = 5
        self.cols = 5
        self.agent_pos = (0, 0)
        self.done = False
        self.reward = 0.0
        self.state_size = self.rows * self.cols  # Size of the state representation
        self.reset()

    def reset(self):
        self.agent_pos = (0, 0)
        self.done = False
        self.reward = 0.0
        return self.state_description()

    def step(self, action: int):
        action = int(action)  # Assurer que l'action est un entier natif
        if self.done:
            return self.state_description(), self.reward, self.done, {}

        if self.action_mask()[action] == 0:
            # Action invalide, la partie se termine
            self.done = True
            self.reward = -10.0  # Pénalité pour action invalide
            return self.state_description(), self.reward, self.done, {}

        self.agent_pos = _update_position(self.agent_pos, action, self.rows, self.cols)
        self.reward = _compute_reward(self.agent_pos)
        self.done = _check_done(self.agent_pos)

        return self.state_description(), self.reward, self.done, {}

    def state_description(self) -> np.ndarray:
        state = np.zeros(self.rows * self.cols, dtype=np.float32)
        index = self.agent_pos[0] * self.cols + self.agent_pos[1]
        state[index] = 1.0
        return state

    def available_actions_ids(self) -> np.ndarray:
        mask = self.action_mask()
        available_actions = np.where(mask == 1.0)[0]
        return available_actions.astype(np.int32)

    def action_mask(self) -> np.ndarray:
        mask = _compute_action_mask(self.agent_pos, self.rows, self.cols)
        return mask

    def is_game_over(self) -> bool:
        return self.done

    def score(self) -> float:
        return self.reward

    def clone(self):
        new_env = GridWorld()
        new_env.agent_pos = self.agent_pos
        new_env.done = self.done
        new_env.reward = self.reward
        return new_env

    def __str__(self):
        grid = np.full((self.rows, self.cols), '_', dtype=str)
        row, col = self.agent_pos
        grid[row, col] = 'A'  # A pour l'agent
        grid_str = '\n'.join([' '.join(row) for row in grid])
        return f"GridWorld State:\n{grid_str}\nReward: {self.reward}\nDone: {self.done}"

    def hash(self):
        state_tuple = (self.agent_pos, self.done)
        return hash(state_tuple)

# JIT-compiled helper functions

@njit
def _update_position(agent_pos, action, rows, cols):
    row, col = agent_pos
    # Deltas pour les actions
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT
    delta_row, delta_col = deltas[action]
    new_row = row + delta_row
    new_col = col + delta_col
    if 0 <= new_row < rows and 0 <= new_col < cols:
        return (new_row, new_col)
    else:
        return (row, col)

@njit
def _compute_reward(agent_pos):
    if agent_pos[0] == 4 and agent_pos[1] == 4:
        return 1.0  # Objectif atteint
    elif agent_pos[0] == 0 and agent_pos[1] == 4:
        return -1.0  # État négatif
    else:
        return -0.01  # Coût de déplacement

@njit
def _check_done(agent_pos):
    return (agent_pos[0] == 4 and agent_pos[1] == 4) or (agent_pos[0] == 0 and agent_pos[1] == 4)

@njit
def _compute_action_mask(agent_pos, rows, cols):
    mask = np.ones(4, dtype=np.float32)
    row, col = agent_pos
    if row == 0:
        mask[0] = 0.0  # Impossible d'aller vers le haut
    if row == rows - 1:
        mask[1] = 0.0  # Impossible d'aller vers le bas
    if col == 0:
        mask[2] = 0.0  # Impossible d'aller à gauche
    if col == cols - 1:
        mask[3] = 0.0  # Impossible d'aller à droite
    return mask
