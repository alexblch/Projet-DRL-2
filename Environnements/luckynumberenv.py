import numpy as np
import random

class DeepDiscreteActionsEnv:
    def reset(self):
        raise NotImplementedError

    def state_description(self) -> np.ndarray:
        raise NotImplementedError

    def available_actions_ids(self) -> np.ndarray:
        raise NotImplementedError

    def action_mask(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: int):
        raise NotImplementedError

    def is_game_over(self) -> bool:
        raise NotImplementedError

    def score(self) -> float:
        raise NotImplementedError

    def __str__(self):
        return "DeepDiscreteActionsEnv"

class LuckyNumbersEnv(DeepDiscreteActionsEnv):
    def __init__(self, size=4):
        self.size = size
        self.total_tiles = self.size * self.size
        self.num_numbers = 20
        # Définition des constantes pour les actions
        self.MAX_CACHE_SIZE = 2 * self.num_numbers  # Taille maximale du cache
        self.ACTION_DRAW_FROM_DECK = 0
        self.ACTION_TAKE_FROM_CACHE_START = 1
        self.ACTION_TAKE_FROM_CACHE_END = self.ACTION_TAKE_FROM_CACHE_START + self.num_numbers - 1
        self.ACTION_ADD_TO_CACHE = self.ACTION_TAKE_FROM_CACHE_END + 1
        self.ACTION_PLACE_TILE_START = self.ACTION_ADD_TO_CACHE + 1
        self.ACTION_PLACE_TILE_END = self.ACTION_PLACE_TILE_START + self.size * self.size - 1
        self.TOTAL_ACTIONS = self.ACTION_PLACE_TILE_END + 1
        self.reset()

    def reset(self):
        self.numbers = [num for num in range(1, self.num_numbers + 1)] * 2
        random.shuffle(self.numbers)
        self.agent_grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.opponent_grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.place_initial_tiles(self.agent_grid)
        self.place_initial_tiles(self.opponent_grid)
        self.shared_cache = []
        self._is_game_over = False
        self._score = 0.0
        self.agent_turn = True
        self.current_tile = None
        return self.state_description()

    def place_initial_tiles(self, grid):
        diagonal_positions = [(i, i) for i in range(self.size)]
        initial_numbers = sorted([self.numbers.pop() for _ in range(self.size)])
        for pos, num in zip(diagonal_positions, initial_numbers):
            row, col = pos
            grid[row][col] = num

    def state_description(self) -> np.ndarray:
        agent_grid_flat = [cell if cell is not None else 0 for row in self.agent_grid for cell in row]
        opponent_grid_flat = [cell if cell is not None else 0 for row in self.opponent_grid for cell in row]
        # Encodage du cache partagé en un histogramme de taille fixe
        cache_encoding = np.zeros(self.num_numbers + 1, dtype=np.int32)  # +1 pour l'indice 0 non utilisé
        for tile in self.shared_cache:
            cache_encoding[tile] += 1  # Compte le nombre de chaque tuile dans le cache
        current_tile_value = [self.current_tile if self.current_tile is not None else 0]
        state = np.array(agent_grid_flat + opponent_grid_flat + list(cache_encoding[1:]) + current_tile_value, dtype=np.float32)
        return state

    def available_actions_ids(self) -> np.ndarray:
        actions = []
        if self.current_tile is None:
            actions.append(self.ACTION_DRAW_FROM_DECK)
            # Actions pour prendre des tuiles spécifiques du cache
            for tile in set(self.shared_cache):
                action_id = self.ACTION_TAKE_FROM_CACHE_START + tile - 1
                actions.append(action_id)
        else:
            actions.append(self.ACTION_ADD_TO_CACHE)
            valid_positions = self.get_valid_positions(self.agent_grid, self.current_tile)
            for pos in valid_positions:
                i, j = pos
                action_id = self.ACTION_PLACE_TILE_START + i * self.size + j
                actions.append(action_id)
        return np.array(actions, dtype=np.int32)

    def action_mask(self) -> np.ndarray:
        mask = np.zeros((self.TOTAL_ACTIONS,), dtype=np.float32)
        if self.current_tile is None:
            mask[self.ACTION_DRAW_FROM_DECK] = 1.0
            for tile in set(self.shared_cache):
                action_id = self.ACTION_TAKE_FROM_CACHE_START + tile - 1
                mask[action_id] = 1.0
        else:
            mask[self.ACTION_ADD_TO_CACHE] = 1.0
            valid_positions = self.get_valid_positions(self.agent_grid, self.current_tile)
            for pos in valid_positions:
                i, j = pos
                action_id = self.ACTION_PLACE_TILE_START + i * self.size + j
                mask[action_id] = 1.0
        return mask

    def step(self, action: int):
        if self._is_game_over:
            raise ValueError("La partie est terminée, veuillez réinitialiser l'environnement.")
        if not self.agent_turn:
            raise ValueError("Ce n'est pas le tour de l'agent.")
        if self.action_mask()[action] == 0:
            raise ValueError("Action invalide.")

        reward = 0.0

        if self.current_tile is None:
            if action == self.ACTION_DRAW_FROM_DECK:
                self.current_tile = self.draw_tile_from_deck()
            elif self.ACTION_TAKE_FROM_CACHE_START <= action <= self.ACTION_TAKE_FROM_CACHE_END:
                tile_value = action - self.ACTION_TAKE_FROM_CACHE_START + 1
                if tile_value in self.shared_cache:
                    self.shared_cache.remove(tile_value)
                    self.current_tile = tile_value
                else:
                    raise ValueError("Tuile non disponible dans le cache.")
            else:
                raise ValueError("Action invalide lors de la sélection d'une tuile.")
            reward = 0.0  # Pas de récompense pour prendre une tuile
        else:
            if action == self.ACTION_ADD_TO_CACHE:
                self.shared_cache.append(self.current_tile)
                self.current_tile = None
                self.agent_turn = False
                self.opponent_play()
                reward = 0.0  # Pas de récompense pour ajouter au cache
            elif self.ACTION_PLACE_TILE_START <= action <= self.ACTION_PLACE_TILE_END:
                action_index = action - self.ACTION_PLACE_TILE_START
                i = action_index // self.size
                j = action_index % self.size
                if not self.is_valid_placement_with_replacement(self.agent_grid, i, j, self.current_tile):
                    raise ValueError("Action invalide.")
                if self.agent_grid[i][j] is not None:
                    self.shared_cache.append(self.agent_grid[i][j])
                self.agent_grid[i][j] = self.current_tile
                self.current_tile = None
                if self.is_grid_complete_and_valid(self.agent_grid):
                    self._is_game_over = True
                    reward = 1.0  # Récompense pour avoir complété la grille
                else:
                    reward = 0.0  # Pas de récompense pour un placement valide
                self.agent_turn = False
                self.opponent_play()
            else:
                raise ValueError("Action invalide.")

        next_state = self.state_description()
        done = self._is_game_over
        return next_state, reward, done, {}

    def draw_tile_from_deck(self):
        if self.numbers:
            return self.numbers.pop()
        else:
            return None

    def opponent_play(self):
        while not self.agent_turn and not self._is_game_over:
            if self.current_tile is None:
                # Décider aléatoirement de prendre du deck ou du cache
                choices = []
                if self.numbers:
                    choices.append('deck')
                if self.shared_cache:
                    choices.append('cache')
                if not choices:
                    self._is_game_over = True
                    self._score = self.calculate_score()
                    return
                choice = random.choice(choices)
                if choice == 'deck':
                    self.current_tile = self.draw_tile_from_deck()
                else:
                    self.current_tile = random.choice(self.shared_cache)
                    self.shared_cache.remove(self.current_tile)
            else:
                # Décider aléatoirement d'ajouter au cache ou de placer sur la grille
                actions = ['cache']
                valid_positions = self.get_valid_positions(self.opponent_grid, self.current_tile)
                if valid_positions:
                    actions.append('place')
                action = random.choice(actions)
                if action == 'cache':
                    self.shared_cache.append(self.current_tile)
                    self.current_tile = None
                    self.agent_turn = True
                else:
                    i, j = random.choice(valid_positions)
                    if self.opponent_grid[i][j] is not None:
                        self.shared_cache.append(self.opponent_grid[i][j])
                    self.opponent_grid[i][j] = self.current_tile
                    self.current_tile = None
                    if self.is_grid_complete_and_valid(self.opponent_grid):
                        self._is_game_over = True
                        self._score = -1.0
                        return
                    self.agent_turn = True

    def is_game_over(self) -> bool:
        return self._is_game_over

    def score(self) -> float:
        return self._score

    def __str__(self):
        grid_str = "Grille de l'Agent:\n"
        for row in self.agent_grid:
            grid_str += ' '.join(['_' if cell is None else str(cell) for cell in row]) + '\n'
        grid_str += "\nGrille de l'Adversaire:\n"
        for row in self.opponent_grid:
            grid_str += ' '.join(['_' if cell is None else str(cell) for cell in row]) + '\n'
        grid_str += f"\nCache Partagé: {self.shared_cache}\n"
        grid_str += f"Tuile Courante: {self.current_tile}\n"
        grid_str += f"Score: {self._score}\n"
        grid_str += f"Partie Terminée: {self._is_game_over}\n"
        return grid_str

    def is_valid_placement(self, grid, row, col, number):
        for j in range(self.size):
            num = grid[row][j]
            if num is not None:
                if j < col and num > number:
                    return False
                if j > col and num < number:
                    return False
        for i in range(self.size):
            num = grid[i][col]
            if num is not None:
                if i < row and num > number:
                    return False
                if i > row and num < number:
                    return False
        return True

    def is_valid_placement_with_replacement(self, grid, row, col, number):
        original_value = grid[row][col]
        grid[row][col] = number
        valid = self.is_valid_placement(grid, row, col, number)
        grid[row][col] = original_value
        return valid

    def get_valid_positions(self, grid, number):
        valid_positions = []
        for i in range(self.size):
            for j in range(self.size):
                if grid[i][j] is None:
                    if self.is_valid_placement(grid, i, j, number):
                        valid_positions.append((i, j))
                else:
                    if self.is_valid_placement_with_replacement(grid, i, j, number):
                        valid_positions.append((i, j))
        return valid_positions

    def is_grid_complete_and_valid(self, grid):
        for i in range(self.size):
            for j in range(self.size):
                if grid[i][j] is None:
                    return False
                if not self.is_valid_placement(grid, i, j, grid[i][j]):
                    return False
        return True

    def calculate_score(self):
        # Calcul du score en fonction de la somme des tuiles
        agent_score = sum(cell for row in self.agent_grid for cell in row if cell is not None)
        opponent_score = sum(cell for row in self.opponent_grid for cell in row if cell is not None)
        if agent_score < opponent_score:
            return 1.0  # L'agent a gagné
        elif agent_score > opponent_score:
            return -1.0  # L'agent a perdu
        else:
            return 0.0  # Match nul

    def get_state(self):
        return {
            'agent_grid': self.agent_grid,
            'opponent_grid': self.opponent_grid,
            'shared_cache': self.shared_cache,
            'numbers_remaining': len(self.numbers),
            'current_tile': self.current_tile,
            'agent_turn': self.agent_turn,
            'is_game_over': self._is_game_over,
            'score': self._score
        }
