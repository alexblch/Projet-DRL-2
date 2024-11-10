import numpy as np
from numba import njit

class GridWorld:
    TOTAL_ACTIONS = 4  # Total number of possible actions

    def __init__(self):
        self.rows = 5
        self.cols = 5
        self.action_space = 4
        self.agent_pos = (0, 0)
        self.done = False
        self.reward = 0.0
        self.state_size = self.rows * self.cols  # Size of the state representation

    def reset(self):
        self.agent_pos = (0, 0)
        self.done = False
        self.reward = 0.0
        return self.state_description()

    def step(self, action: int):
        if self.done:
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

    def action_mask(self) -> np.ndarray:
        mask = _compute_action_mask(self.agent_pos, self.rows, self.cols)
        return mask

    # Additional methods (if needed) ...

# JIT-compiled helper functions

@njit
def _update_position(agent_pos, action, rows, cols):
    row, col = agent_pos
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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
        return 1.0
    elif agent_pos[0] == 0 and agent_pos[1] == 4:
        return -3.0
    else:
        return 0.0

@njit
def _check_done(agent_pos):
    return (agent_pos[0] == 4 and agent_pos[1] == 4) or (agent_pos[0] == 0 and agent_pos[1] == 4)

@njit
def _compute_action_mask(agent_pos, rows, cols):
    mask = np.ones(4, dtype=np.float32)
    row, col = agent_pos
    if col == 0:
        mask[0] = 0.0
    if col == cols - 1:
        mask[1] = 0.0
    if row == 0:
        mask[2] = 0.0
    if row == rows - 1:
        mask[3] = 0.0
    return mask
