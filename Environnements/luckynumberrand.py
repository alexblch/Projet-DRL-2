import random
from tqdm import tqdm

class LuckyNumbersGameRandConsole:
    def __init__(self, size=4):
        self.size = size  # Taille de la grille
        self.total_tiles = self.size * self.size
        # Créer une liste de nombres de 1 à 20, deux fois chacun
        self.numbers = [num for num in range(1, 21)] * 2  # Nombres entre 1 et 20, deux fois
        random.shuffle(self.numbers)
        self.ai1_grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.ai2_grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.shared_cache = []  # Cache partagé sans limite de capacité
        self.turn = 'ai1'  # Pour gérer le tour de chaque IA
        self.current_tile = None

        # Placer les tuiles initiales
        self.place_initial_tiles()

    def place_initial_tiles(self):
        """Place les nombres initiaux sur la diagonale principale de chaque grille."""
        diagonal_positions = [(i, i) for i in range(self.size)]

        # Pour l'IA 1
        initial_numbers_ai1 = sorted([self.numbers.pop() for _ in range(self.size)])
        for pos, num in zip(diagonal_positions, initial_numbers_ai1):
            row, col = pos
            self.ai1_grid[row][col] = num

        # Pour l'IA 2
        initial_numbers_ai2 = sorted([self.numbers.pop() for _ in range(self.size)])
        for pos, num in zip(diagonal_positions, initial_numbers_ai2):
            row, col = pos
            self.ai2_grid[row][col] = num

    def run_game(self):
        while not self.check_winner():
            if self.turn == 'ai1':
                self.ai_turn(self.ai1_grid, 'IA 1')
                self.turn = 'ai2'
            elif self.turn == 'ai2':
                self.ai_turn(self.ai2_grid, 'IA 2')
                self.turn = 'ai1'

    def ai_turn(self, grid, ai_name):
        if not self.numbers and not self.shared_cache:
            return

        # L'IA choisit d'utiliser le cache s'il n'est pas vide
        if self.shared_cache and (not self.numbers or random.choice([True, False])):
            ai_tile = self.shared_cache.pop(0)
        elif self.numbers:
            ai_tile = self.numbers.pop()
        else:
            ai_tile = None

        if ai_tile is not None:
            positions = self.get_valid_positions(grid, ai_tile)
            if positions:
                i, j = random.choice(positions)
                # Si l'IA remplace une tuile, elle ajoute l'ancienne au cache
                if grid[i][j] is not None:
                    self.shared_cache.append(grid[i][j])
                grid[i][j] = ai_tile
            else:
                self.shared_cache.append(ai_tile)

    def is_valid_placement(self, grid, row, col, number):
        """Vérifie si le placement respecte l'ordre croissant sur la ligne et la colonne sans remplacement."""
        # Vérification sur la ligne
        for j in range(self.size):
            num = grid[row][j]
            if num is not None:
                if j < col and num > number:
                    return False
                if j > col and num < number:
                    return False

        # Vérification sur la colonne
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

    def check_winner(self):
        if self.is_grid_complete_and_valid(self.ai1_grid):
            return 'IA 1'
        if self.is_grid_complete_and_valid(self.ai2_grid):
            return 'IA 2'
        if not self.numbers and not self.shared_cache:
            # Vérifier si une des grilles est valide même si elle n'est pas complète
            ai1_valid = all(
                self.is_valid_placement(self.ai1_grid, i, j, self.ai1_grid[i][j])
                for i in range(self.size)
                for j in range(self.size)
                if self.ai1_grid[i][j] is not None
            )
            ai2_valid = all(
                self.is_valid_placement(self.ai2_grid, i, j, self.ai2_grid[i][j])
                for i in range(self.size)
                for j in range(self.size)
                if self.ai2_grid[i][j] is not None
            )
            if ai1_valid and not ai2_valid:
                return 'IA 1'
            if ai2_valid and not ai1_valid:
                return 'IA 2'
            return 'Draw'
        return None

def play_n_games(n):
    ia1_wins = 0
    ia2_wins = 0
    draws = 0

    for i in tqdm(range(n)):
        game = LuckyNumbersGameRandConsole()
        game.run_game()
        result = game.check_winner()
        if result == 'IA 1':
            ia1_wins += 1
        elif result == 'IA 2':
            ia2_wins += 1
        else:
            draws += 1

    print(f"Sur {n} parties :")
    print(f"IA 1 a gagné {ia1_wins} fois")
    print(f"IA 2 a gagné {ia2_wins} fois")
    print(f"Il y a eu {draws} matchs nuls")

if __name__ == "__main__":
    n = int (input("Entrez le nombre de parties à jouer : "))
    play_n_games(n)  # Jouer n parties
