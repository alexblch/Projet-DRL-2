import numpy as np
import random
from numba import njit

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
        self.num_numbers = 20  # Nombres de 1 à 20
        self.total_tiles = self.num_numbers * 2  # Chaque nombre apparaît deux fois
        # Définition des constantes pour les actions
        self.ACTION_DRAW_FROM_DECK = 0
        self.ACTION_TAKE_FROM_CACHE_START = 1
        self.ACTION_TAKE_FROM_CACHE_END = self.ACTION_TAKE_FROM_CACHE_START + self.num_numbers - 1
        self.ACTION_ADD_TO_CACHE = self.ACTION_TAKE_FROM_CACHE_END + 1
        self.ACTION_PLACE_TILE_START = self.ACTION_ADD_TO_CACHE + 1
        self.ACTION_PLACE_TILE_END = self.ACTION_PLACE_TILE_START + self.size * self.size - 1
        self.TOTAL_ACTIONS = self.ACTION_PLACE_TILE_END + 1
        self.reset()

    def reset(self):
        # Créer une liste avec les nombres de 1 à 20, chacun apparaissant deux fois
        self.numbers = [num for num in range(1, self.num_numbers + 1)] * 2
        np.random.shuffle(self.numbers)
        self.agent_grid = np.full((self.size, self.size), -1, dtype=np.int32)
        self.opponent_grid = np.full((self.size, self.size), -1, dtype=np.int32)
        # Les grilles sont initialisées vides, suppression des appels à place_initial_tiles
        self.shared_cache = []
        self._is_game_over = False
        self._score = 0.0
        self.agent_turn = True
        self.current_tile = -1
        return self.state_description()

    # Le reste du code reste inchangé

    def state_description(self) -> np.ndarray:
        # Encodage de la grille de l'agent
        agent_grid_flat = self.agent_grid.flatten()
        # Encodage de la grille de l'adversaire
        opponent_grid_flat = self.opponent_grid.flatten()
        # Encodage du cache partagé
        max_tile_value = self.num_numbers
        cache_encoding = np.zeros(max_tile_value + 1, dtype=np.int32)  # +1 car l'indice 0 n'est pas utilisé
        for tile in self.shared_cache:
            cache_encoding[tile] += 1
        # Tuile courante
        current_tile_value = np.array([self.current_tile], dtype=np.int32)
        # Construction de l'état
        state = np.concatenate((agent_grid_flat, opponent_grid_flat, cache_encoding[1:], current_tile_value)).astype(np.float32)
        return state

    def available_actions_ids(self) -> np.ndarray:
        actions = []
        if self.current_tile == -1:
            actions.append(int(self.ACTION_DRAW_FROM_DECK))
            # Actions pour prendre des tuiles spécifiques du cache
            unique_tiles = np.unique(self.shared_cache)
            for tile in unique_tiles:
                action_id = int(self.ACTION_TAKE_FROM_CACHE_START + tile - 1)
                actions.append(action_id)
        else:
            actions.append(int(self.ACTION_ADD_TO_CACHE))
            valid_positions = self.get_valid_positions(self.agent_grid, self.current_tile)
            for idx in valid_positions:
                action_id = int(self.ACTION_PLACE_TILE_START + idx)
                actions.append(action_id)
        return np.array(actions, dtype=np.int32)

    def action_mask(self) -> np.ndarray:
        mask = np.zeros((self.TOTAL_ACTIONS,), dtype=np.float32)
        available_actions = self.available_actions_ids()
        mask[available_actions] = 1.0
        return mask

    def step(self, action: int):
        action = int(action)  # Conversion en entier Python natif
        if self._is_game_over:
            # La partie est déjà terminée
            next_state = self.state_description()
            reward = self._score
            done = True
            return next_state, reward, done, {}

        # Si ce n'est pas le tour de l'agent, l'adversaire doit jouer
        if not self.agent_turn:
            # Laisser l'adversaire jouer
            self.opponent_play()
            # Après le tour de l'adversaire, vérifier si la partie est terminée
            if self._is_game_over:
                next_state = self.state_description()
                reward = self._score
                done = True
                return next_state, reward, done, {}
            else:
                # Le tour revient à l'agent
                self.agent_turn = True
                # Continuer pour traiter l'action de l'agent

        # Vérifier à nouveau si la partie est terminée
        if self._is_game_over:
            next_state = self.state_description()
            reward = self._score
            done = True
            return next_state, reward, done, {}

        # Maintenant, c'est le tour de l'agent
        if self.action_mask()[action] == 0:
            # Action invalide, la partie se termine avec récompense 0.0
            self._is_game_over = True
            self._score = 0.0  # Récompense neutre pour action invalide
            reward = self._score
            next_state = self.state_description()
            done = True
            return next_state, reward, done, {}

        reward = 0.0

        # Traitement de l'action de l'agent
        if self.current_tile == -1:
            if action == self.ACTION_DRAW_FROM_DECK:
                self.current_tile = self.draw_tile_from_deck()
                if self.current_tile == -1:
                    # Plus de tuiles à piocher, fin du jeu
                    self._is_game_over = True
                    self._score = self.calculate_score()
            elif self.ACTION_TAKE_FROM_CACHE_START <= action <= self.ACTION_TAKE_FROM_CACHE_END:
                tile_value = action - self.ACTION_TAKE_FROM_CACHE_START + 1
                if tile_value in self.shared_cache:
                    self.shared_cache.remove(tile_value)
                    self.current_tile = tile_value
                else:
                    # Tuile non disponible dans le cache
                    self._is_game_over = True
                    self._score = 0.0
                    reward = self._score
                    next_state = self.state_description()
                    done = True
                    return next_state, reward, done, {}
            else:
                # Action invalide lors de la sélection d'une tuile
                self._is_game_over = True
                self._score = 0.0
                reward = self._score
                next_state = self.state_description()
                done = True
                return next_state, reward, done, {}
        else:
            # Vérifier si la tuile est déjà dans la grille de l'agent
            if self.current_tile in self.agent_grid:
                if action != self.ACTION_ADD_TO_CACHE:
                    # Impossible de placer un doublon
                    self._is_game_over = True
                    self._score = 0.0
                    reward = self._score
                    next_state = self.state_description()
                    done = True
                    return next_state, reward, done, {}
            if action == self.ACTION_ADD_TO_CACHE:
                self.shared_cache.append(self.current_tile)
                self.current_tile = -1
                self.agent_turn = False
                # L'adversaire va jouer au prochain appel de step
            elif self.ACTION_PLACE_TILE_START <= action <= self.ACTION_PLACE_TILE_END:
                action_index = action - self.ACTION_PLACE_TILE_START
                i = action_index // self.size
                j = action_index % self.size
                if not is_valid_placement_with_replacement_numba(self.agent_grid, i, j, self.current_tile):
                    # Placement invalide, la partie se termine avec récompense 0.0
                    self._is_game_over = True
                    self._score = 0.0
                    reward = self._score
                    next_state = self.state_description()
                    done = True
                    return next_state, reward, done, {}
                if self.agent_grid[i, j] != -1:
                    old_tile = self.agent_grid[i, j]
                    self.shared_cache.append(old_tile)
                self.agent_grid[i, j] = self.current_tile
                self.current_tile = -1

                # Vérifier si la grille de l'agent est entièrement remplie
                if self.is_grid_full(self.agent_grid):
                    self._is_game_over = True
                    self._score = self.calculate_score()
                    reward = self._score
                else:
                    self.agent_turn = False
                    # L'adversaire va jouer au prochain appel de step
            else:
                # Action invalide
                self._is_game_over = True
                self._score = 0.0
                reward = self._score
                next_state = self.state_description()
                done = True
                return next_state, reward, done, {}

        # Après le tour de l'agent, vérifier si la partie est terminée
        if self._is_game_over:
            reward = self._score
            next_state = self.state_description()
            done = True
            return next_state, reward, done, {}

        # Si ce n'est plus le tour de l'agent, l'adversaire doit jouer
        if not self.agent_turn:
            self.opponent_play()
            # Vérifier si la partie est terminée après le tour de l'adversaire
            if self._is_game_over:
                reward = self._score
                next_state = self.state_description()
                done = True
                return next_state, reward, done, {}

        # Retourner l'état suivant
        next_state = self.state_description()
        done = self._is_game_over
        return next_state, reward, done, {}

    def draw_tile_from_deck(self):
        if self.numbers:
            tile = self.numbers.pop()
            return tile
        else:
            return -1

    def opponent_play(self):
        while not self.agent_turn and not self._is_game_over:
            if self.current_tile == -1:
                # Décider de prendre du deck ou du cache
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
                    if self.current_tile == -1:
                        self._is_game_over = True
                        self._score = self.calculate_score()
                        return
                else:
                    available_tiles = [tile for tile in self.shared_cache]
                    if available_tiles:
                        self.current_tile = random.choice(available_tiles)
                        self.shared_cache.remove(self.current_tile)
                    else:
                        # Si aucune tuile disponible, passer le tour
                        self.agent_turn = True
                        self.current_tile = -1
                        continue
            else:
                # Vérifier si la tuile est déjà dans la grille de l'adversaire
                if self.current_tile in self.opponent_grid:
                    # L'adversaire ne peut pas placer un doublon
                    self.shared_cache.append(self.current_tile)
                    self.current_tile = -1
                    self.agent_turn = True
                    continue
                # Décider d'ajouter au cache ou de placer sur la grille
                actions = ['cache']
                valid_positions = self.get_valid_positions(self.opponent_grid, self.current_tile)
                if valid_positions:
                    actions.append('place')
                action = random.choice(actions)
                if action == 'cache':
                    self.shared_cache.append(self.current_tile)
                    self.current_tile = -1
                    self.agent_turn = True
                else:
                    idx = random.choice(valid_positions)
                    i = idx // self.size
                    j = idx % self.size
                    if self.opponent_grid[i, j] != -1:
                        old_tile = self.opponent_grid[i, j]
                        self.shared_cache.append(old_tile)
                    self.opponent_grid[i, j] = self.current_tile
                    self.current_tile = -1

                    # Vérifier si la grille de l'adversaire est entièrement remplie
                    if self.is_grid_full(self.opponent_grid):
                        self._is_game_over = True
                        self._score = self.calculate_score()
                        return
                    self.agent_turn = True

    def is_game_over(self) -> bool:
        return self._is_game_over

    def score(self) -> float:
        return self._score

    def __str__(self):
        grid_str = "Grille de l'Agent:\n"
        for row in self.agent_grid:
            grid_str += ' '.join(['_' if cell == -1 else str(cell) for cell in row]) + '\n'
        grid_str += "\nGrille de l'Adversaire:\n"
        for row in self.opponent_grid:
            grid_str += ' '.join(['_' if cell == -1 else str(cell) for cell in row]) + '\n'
        grid_str += f"\nCache Partagé: {self.shared_cache}\n"
        grid_str += f"Tuile Courante: {self.current_tile if self.current_tile != -1 else 'Aucune'}\n"
        grid_str += f"Score: {self._score}\n"
        grid_str += f"Partie Terminée: {self._is_game_over}\n"
        return grid_str

    def get_valid_positions(self, grid, number):
        if number in grid:
            return []
        valid_positions = get_valid_positions_numba(grid, number)
        valid_positions = [int(idx) for idx in valid_positions]  # Conversion en entiers Python natifs
        return valid_positions

    def is_grid_full(self, grid):
        return np.all(grid != -1)

    def calculate_score(self):
        # Calcul du score en fonction de qui a rempli la grille en premier
        agent_full = self.is_grid_full(self.agent_grid)
        opponent_full = self.is_grid_full(self.opponent_grid)
        if agent_full and not opponent_full:
            return 1.0  # L'agent a gagné
        elif opponent_full and not agent_full:
            return -1.0  # L'agent a perdu
        elif agent_full and opponent_full:
            return 0.0  # Match nul
        else:
            agent_empty = np.sum(self.agent_grid == -1)
            opponent_empty = np.sum(self.opponent_grid == -1)
            if agent_empty < opponent_empty:
                return 1.0  # L'agent a moins de cases vides
            elif agent_empty > opponent_empty:
                return -1.0  # L'adversaire a moins de cases vides
            else:
                return 0.0  # Match nul

    def get_state(self):
        return {
            'agent_grid': self.agent_grid.copy(),
            'opponent_grid': self.opponent_grid.copy(),
            'shared_cache': self.shared_cache.copy(),
            'numbers_remaining': len(self.numbers),
            'current_tile': self.current_tile,
            'agent_turn': self.agent_turn,
            'is_game_over': self._is_game_over,
            'score': self._score
        }

    def clone(self):
        new_env = LuckyNumbersEnv(self.size)
        new_env.numbers = self.numbers.copy()
        new_env.agent_grid = self.agent_grid.copy()
        new_env.opponent_grid = self.opponent_grid.copy()
        new_env.shared_cache = self.shared_cache.copy()
        new_env._is_game_over = self._is_game_over
        new_env._score = self._score
        new_env.agent_turn = self.agent_turn
        new_env.current_tile = self.current_tile
        return new_env

    def hash(self):
        agent_grid_bytes = self.agent_grid.tobytes()
        opponent_grid_bytes = self.opponent_grid.tobytes()
        shared_cache_tuple = tuple(sorted(self.shared_cache))
        state_tuple = (
            agent_grid_bytes,
            opponent_grid_bytes,
            shared_cache_tuple,
            self.current_tile,
            self.agent_turn,
        )
        return hash(state_tuple)

# Fonctions Numba
@njit
def is_valid_placement_numba(grid, row, col, number):
    size = grid.shape[0]
    # Vérification sur la ligne
    for j in range(size):
        num = grid[row, j]
        if num != -1:
            if j < col and num >= number:
                return False
            if j > col and num <= number:
                return False
    # Vérification sur la colonne
    for i in range(size):
        num = grid[i, col]
        if num != -1:
            if i < row and num >= number:
                return False
            if i > row and num <= number:
                return False
    return True

@njit
def is_valid_placement_with_replacement_numba(grid, row, col, number):
    original_value = grid[row, col]
    grid[row, col] = number
    valid = is_valid_placement_numba(grid, row, col, number)
    grid[row, col] = original_value
    return valid

@njit
def get_valid_positions_numba(grid, number):
    size = grid.shape[0]
    valid_positions = []
    for i in range(size):
        for j in range(size):
            if grid[i, j] == -1:
                if is_valid_placement_numba(grid, i, j, number):
                    idx = i * size + j
                    valid_positions.append(idx)
            else:
                if is_valid_placement_with_replacement_numba(grid, i, j, number):
                    idx = i * size + j
                    valid_positions.append(idx)
    return valid_positions

@njit
def is_grid_complete_and_valid_numba(grid):
    size = grid.shape[0]
    for i in range(size):
        for j in range(size):
            number = grid[i, j]
            if number == -1:
                return False
            if not is_valid_placement_numba(grid, i, j, number):
                return False
    return True
