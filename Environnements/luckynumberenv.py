import numpy as np
import random

class DeepDiscreteActionsEnv:
    """
    Classe de base pour les environnements avec actions discrètes, adaptée pour le deep reinforcement learning.
    """
    def reset(self):
        """
        Réinitialise l'environnement à l'état initial et retourne l'observation initiale.
        """
        raise NotImplementedError

    def state_description(self) -> np.ndarray:
        """
        Retourne la description de l'état actuel sous forme de vecteur numpy.
        """
        raise NotImplementedError

    def available_actions_ids(self) -> np.ndarray:
        """
        Retourne un tableau numpy des IDs des actions disponibles dans l'état actuel.
        """
        raise NotImplementedError

    def action_mask(self) -> np.ndarray:
        """
        Retourne un masque numpy où chaque action disponible est marquée par 1 et les autres par 0.
        """
        raise NotImplementedError

    def step(self, action: int):
        """
        Exécute l'action donnée et met à jour l'état de l'environnement.

        Paramètres:
            action (int): L'ID de l'action à exécuter.
        """
        raise NotImplementedError

    def is_game_over(self) -> bool:
        """
        Indique si la partie est terminée.

        Retourne:
            bool: True si la partie est terminée, False sinon.
        """
        raise NotImplementedError

    def score(self) -> float:
        """
        Retourne le score actuel de la partie.

        Retourne:
            float: Le score de la partie.
        """
        raise NotImplementedError

    def __str__(self):
        """
        Retourne une représentation textuelle de l'état actuel de l'environnement.
        """
        return "DeepDiscreteActionsEnv"

class LuckyNumbersEnv(DeepDiscreteActionsEnv):
    def __init__(self, size=4):
        self.size = size
        self.total_tiles = self.size * self.size
        self.num_numbers = 20  # Nombres de 1 à 20
        self.reset()

    def reset(self):
        # Créer une liste de nombres de 1 à 20, deux fois chacun
        self.numbers = [num for num in range(1, self.num_numbers + 1)] * 2
        random.shuffle(self.numbers)

        # Initialiser la grille de l'agent et de l'adversaire (aléatoire)
        self.agent_grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.opponent_grid = [[None for _ in range(self.size)] for _ in range(self.size)]

        # Placer les tuiles initiales sur la diagonale principale
        self.place_initial_tiles(self.agent_grid)
        self.place_initial_tiles(self.opponent_grid)

        # Cache partagé
        self.shared_cache = []

        # L'état du jeu
        self._is_game_over = False
        self._score = 0.0  # 1.0 si l'agent gagne, -1.0 s'il perd, 0.0 pour un match nul

        # Indique si c'est au tour de l'agent de jouer
        self.agent_turn = True

        # La tuile actuelle que l'agent doit placer
        self.current_tile = None

    def place_initial_tiles(self, grid):
        diagonal_positions = [(i, i) for i in range(self.size)]
        initial_numbers = sorted([self.numbers.pop() for _ in range(self.size)])
        for pos, num in zip(diagonal_positions, initial_numbers):
            row, col = pos
            grid[row][col] = num

    def state_description(self) -> np.ndarray:
        """
        Retourne une description de l'état sous forme de vecteur numpy.
        """
        # Convertir les grilles en vecteurs
        agent_grid_flat = [cell if cell is not None else 0 for row in self.agent_grid for cell in row]
        opponent_grid_flat = [cell if cell is not None else 0 for row in self.opponent_grid for cell in row]
        shared_cache_flat = self.shared_cache + [0] * (self.num_numbers * 2 - len(self.shared_cache))

        # Combiner les vecteurs
        state = np.array(agent_grid_flat + opponent_grid_flat + shared_cache_flat, dtype=np.float32)
        return state

    def available_actions_ids(self) -> np.ndarray:
        actions = []
        if self.current_tile is None:
            # Seule l'action de piocher est disponible
            actions.append(0)  # ID pour "piocher une tuile"
        else:
            # Action pour mettre la tuile au cache
            actions.append(1)  # ID pour "mettre la tuile au cache"
            # Actions pour placer la tuile sur la grille
            valid_positions = self.get_valid_positions(self.agent_grid, self.current_tile)
            for pos in valid_positions:
                i, j = pos
                action_id = 2 + i * self.size + j  # Décalage de 2 pour les actions spéciales
                actions.append(action_id)
        return np.array(actions, dtype=np.int32)

    def action_mask(self) -> np.ndarray:
        mask_size = 2 + self.size * self.size  # 2 actions spéciales + actions de placement
        mask = np.zeros((mask_size,), dtype=np.float32)
        if self.current_tile is None:
            mask[0] = 1.0  # Action "piocher une tuile"
        else:
            mask[1] = 1.0  # Action "mettre la tuile au cache"
            valid_positions = self.get_valid_positions(self.agent_grid, self.current_tile)
            for pos in valid_positions:
                i, j = pos
                action_id = 2 + i * self.size + j
                mask[action_id] = 1.0
        return mask

    def step(self, action: int):
        if self._is_game_over:
            raise ValueError("La partie est terminée, veuillez réinitialiser l'environnement.")

        if not self.agent_turn:
            raise ValueError("Ce n'est pas le tour de l'agent.")

        if action == 0:
            # Action "piocher une tuile"
            self.current_tile = self.draw_tile()
            if self.current_tile is None:
                self._is_game_over = True
                self._score = 0.0  # Match nul
            return
        elif action == 1:
            # Action "mettre la tuile au cache"
            if self.current_tile is None:
                raise ValueError("Aucune tuile à mettre au cache.")
            self.shared_cache.append(self.current_tile)
            self.current_tile = None
            self.agent_turn = False
            self.opponent_play()
            return
        else:
            # Actions de placement
            if self.current_tile is None:
                raise ValueError("Pas de tuile à placer. Vous devez piocher une tuile d'abord.")
            action_index = action - 2  # Ajuster l'index pour les actions de placement
            i = action_index // self.size
            j = action_index % self.size
            if not self.is_valid_placement_with_replacement(self.agent_grid, i, j, self.current_tile):
                raise ValueError("Action invalide.")
            if self.agent_grid[i][j] is not None:
                self.shared_cache.append(self.agent_grid[i][j])
            self.agent_grid[i][j] = self.current_tile
            self.current_tile = None

            # Vérifier si l'agent a gagné
            if self.is_grid_complete_and_valid(self.agent_grid):
                self._is_game_over = True
                self._score = 1.0  # L'agent a gagné
                return

            # Tour de l'adversaire
            self.agent_turn = False
            self.opponent_play()
            return

    def draw_tile(self):
        if self.numbers:
            return self.numbers.pop()
        elif self.shared_cache:
            return self.shared_cache.pop(0)
        else:
            return None

    def opponent_play(self):
        # L'adversaire joue de manière aléatoire
        tile = self.draw_tile()
        if tile is None:
            # Plus de tuiles à jouer
            self._is_game_over = True
            self._score = 0.0  # Match nul
            return

        valid_positions = self.get_valid_positions(self.opponent_grid, tile)
        if valid_positions:
            i, j = random.choice(valid_positions)
            if self.opponent_grid[i][j] is not None:
                self.shared_cache.append(self.opponent_grid[i][j])
            self.opponent_grid[i][j] = tile
        else:
            self.shared_cache.append(tile)

        # Vérifier si l'adversaire a gagné
        if self.is_grid_complete_and_valid(self.opponent_grid):
            self._is_game_over = True
            self._score = -1.0  # L'adversaire a gagné
            return

        # Repasser le tour à l'agent
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
        grid_str += f"Score: {self._score}\n"
        grid_str += f"Partie Terminée: {self._is_game_over}\n"
        return grid_str

    # Méthodes utilitaires

    def is_valid_placement(self, grid, row, col, number):
        """Vérifie si le placement respecte l'ordre croissant sur la ligne et la colonne sans remplacement."""
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
        """Vérifie si le remplacement d'une tuile respecte l'ordre croissant sur la ligne et la colonne."""
        original_value = grid[row][col]
        grid[row][col] = number
        valid = self.is_valid_placement(grid, row, col, number)
        grid[row][col] = original_value
        return valid

    def get_valid_positions(self, grid, number):
        """Retourne une liste des positions valides pour placer le nombre donné, y compris les remplacements."""
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
        """Vérifie si la grille est entièrement remplie et respecte les règles de tri."""
        for i in range(self.size):
            for j in range(self.size):
                if grid[i][j] is None:
                    return False  # Grille incomplète
                if not self.is_valid_placement(grid, i, j, grid[i][j]):
                    return False  # Placement invalide
        return True  # Grille complète et valide

    # Fonction pour obtenir l'état actuel du jeu
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
