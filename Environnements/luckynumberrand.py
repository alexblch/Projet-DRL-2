import tkinter as tk
import random

class LuckyNumbersGameRand:
    def __init__(self, master):
        self.master = master
        self.master.title("Lucky Numbers - AI vs AI")
        self.size = 4  # Taille de la grille
        self.total_tiles = self.size * self.size
        # Créer une liste de nombres de 1 à 20, deux fois chacun
        self.numbers = [num for num in range(1, 21)] * 2  # Nombres entre 1 et 20, deux fois
        random.shuffle(self.numbers)
        self.ai1_grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.ai2_grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.shared_cache = []  # Cache partagé sans limite de capacité
        self.create_widgets()
        self.current_tile = None
        self.turn = 'ai1'  # Pour gérer le tour de chaque IA
        self.place_initial_tiles()
        self.master.after(1000, self.run_game)  # Démarrer le jeu automatiquement après 1 seconde

    def place_initial_tiles(self):
        """Place les nombres initiaux sur la diagonale principale de chaque grille."""
        diagonal_positions = [(i, i) for i in range(self.size)]

        # Pour l'IA 1
        initial_numbers_ai1 = sorted([self.numbers.pop() for _ in range(self.size)])
        for pos, num in zip(diagonal_positions, initial_numbers_ai1):
            row, col = pos
            self.ai1_grid[row][col] = num
            self.ai1_labels[row][col].config(text=str(num), bg='light blue')

        # Pour l'IA 2
        initial_numbers_ai2 = sorted([self.numbers.pop() for _ in range(self.size)])
        for pos, num in zip(diagonal_positions, initial_numbers_ai2):
            row, col = pos
            self.ai2_grid[row][col] = num
            self.ai2_labels[row][col].config(text=str(num), bg='light blue')

    def create_widgets(self):
        self.frame = tk.Frame(self.master)
        self.frame.pack(padx=20, pady=20)

        # Grille de l'IA 1
        tk.Label(self.frame, text="Grille de l'IA 1", font=('Helvetica', 16, 'bold')).grid(row=0, column=0, columnspan=self.size)
        self.ai1_labels = [[tk.Label(self.frame, width=6, height=3, font=('Helvetica', 14), relief='sunken', bg='light grey')
                            for j in range(self.size)] for i in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                self.ai1_labels[i][j].grid(row=i+1, column=j, padx=5, pady=5)

        # Espace entre les deux grilles
        tk.Label(self.frame, text="", width=2).grid(row=1, column=self.size, rowspan=self.size)

        # Grille de l'IA 2
        tk.Label(self.frame, text="Grille de l'IA 2", font=('Helvetica', 16, 'bold')).grid(row=0, column=self.size+1, columnspan=self.size)
        self.ai2_labels = [[tk.Label(self.frame, width=6, height=3, font=('Helvetica', 14), relief='sunken', bg='light grey')
                            for j in range(self.size)] for i in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                self.ai2_labels[i][j].grid(row=i+1, column=self.size+1 + j, padx=5, pady=5)

        # Affichage du cache partagé
        tk.Label(self.frame, text="Cache :", font=('Helvetica', 14, 'bold')).grid(row=self.size+1, column=0, columnspan=self.size*2)
        self.cache_frame = tk.Frame(self.frame)
        self.cache_frame.grid(row=self.size+2, column=0, columnspan=self.size*2, pady=5)
        self.cache_buttons = []

        self.info_label = tk.Label(self.frame, text="Le jeu commence...", font=('Helvetica', 14))
        self.info_label.grid(row=self.size+3, column=0, columnspan=self.size*2, pady=10)

        # Bouton pour redémarrer le jeu
        self.restart_button = tk.Button(self.frame, text="Rejouer", font=('Helvetica', 14), command=self.restart_game)
        self.restart_button.grid(row=self.size+4, column=0, columnspan=self.size*2, pady=10)

    def update_cache_display(self):
        """Met à jour l'affichage du cache partagé."""
        # Supprimer les anciens boutons du cache
        for button in self.cache_buttons:
            button.destroy()
        self.cache_buttons = []

        # Créer un label pour chaque tuile du cache
        for idx, tile in enumerate(self.shared_cache):
            lbl = tk.Label(self.cache_frame, text=str(tile), font=('Helvetica', 14), width=4, relief='raised')
            lbl.grid(row=0, column=idx, padx=2)
            self.cache_buttons.append(lbl)

        # Si le cache est vide, afficher un label "Vide"
        if not self.shared_cache:
            tk.Label(self.cache_frame, text="Vide", font=('Helvetica', 14)).grid(row=0, column=0)

    def run_game(self):
        if self.turn == 'ai1':
            self.ai_turn(self.ai1_grid, self.ai1_labels, 'IA 1')
            self.turn = 'ai2'
        elif self.turn == 'ai2':
            self.ai_turn(self.ai2_grid, self.ai2_labels, 'IA 2')
            self.turn = 'ai1'

        if not self.check_winner():
            self.master.after(1000, self.run_game)  # Continuer le jeu après 1 seconde

    def ai_turn(self, grid, labels, ai_name):
        if not self.numbers and not self.shared_cache:
            return

        # L'IA choisit d'utiliser le cache s'il n'est pas vide
        if self.shared_cache and (not self.numbers or random.choice([True, False])):
            ai_tile = self.shared_cache.pop(0)
            self.update_cache_display()
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
                    self.update_cache_display()
                grid[i][j] = ai_tile
                labels[i][j].config(text=str(ai_tile), bg='white')
                self.info_label.config(text=f"{ai_name} a placé {ai_tile} en position ({i+1},{j+1})")
            else:
                self.shared_cache.append(ai_tile)
                self.update_cache_display()
                self.info_label.config(text=f"{ai_name} ne peut pas placer {ai_tile} et l'ajoute au cache")
        else:
            self.info_label.config(text=f"{ai_name} ne peut plus jouer")
        self.master.update()

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

    def check_winner(self):
        if all(all(cell is not None for cell in row) for row in self.ai1_grid):
            self.info_label.config(text="IA 1 a gagné!")
            self.end_game()
            return True

        if all(all(cell is not None for cell in row) for row in self.ai2_grid):
            self.info_label.config(text="IA 2 a gagné!")
            self.end_game()
            return True

        if not self.numbers and not self.shared_cache:
            self.info_label.config(text="Match nul!")
            self.end_game()
            return True

        return False

    def end_game(self):
        # Désactiver les mises à jour supplémentaires
        pass

    def restart_game(self):
        """Redémarre le jeu en réinitialisant toutes les variables et l'interface."""
        # Réinitialiser les variables du jeu
        self.numbers = [num for num in range(1, 21)] * 2  # Nombres de 1 à 20, deux fois
        random.shuffle(self.numbers)
        self.shared_cache = []
        self.turn = 'ai1'
        self.ai1_grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.ai2_grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.info_label.config(text="Le jeu recommence...")
        self.update_cache_display()

        # Réinitialiser les labels de la grille de l'IA 1
        for i in range(self.size):
            for j in range(self.size):
                self.ai1_labels[i][j].config(text="", bg='light grey')

        # Réinitialiser les labels de la grille de l'IA 2
        for i in range(self.size):
            for j in range(self.size):
                self.ai2_labels[i][j].config(text="", bg='light grey')

        # Placer les tuiles initiales
        self.place_initial_tiles()
        self.master.after(1000, self.run_game)  # Recommencer le jeu après 1 seconde
        
if __name__ == "__main__":
    root = tk.Tk()
    game = LuckyNumbersGameRand(root)
    root.mainloop()