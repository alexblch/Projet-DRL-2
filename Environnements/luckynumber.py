from numba import njit
import numpy as np
import random

class LuckyNumberEnv:
    def __init__(self):
        self.RED = "\033[31m"
        self.GREEN = "\033[32m"
        self.YELLOW = "\033[33m"
        self.BLUE = "\033[34m"
        self.RESET = "\033[0m"
        self.rows = 4
        self.cols = 4
        self.state = np.zeros((self.rows, self.cols))  # Grille 4x4
        self.done = False
        self.action_space = self.rows * self.cols  # 16 actions possibles (une pour chaque case)
        self.reward = 0
        self.available_numbers = []  # Liste des nombres disponibles pour tirage
        self.list_scores = []  # Liste des scores pour chaque épisode

    def reset(self):
        """Réinitialise l'état du jeu pour un nouvel épisode"""
        self.state = np.zeros((self.rows, self.cols))
        self.done = False
        self.reward = 0
        self.available_numbers = list(range(1, 21))  # Réinitialise la liste des nombres uniques
        random.shuffle(self.available_numbers)  # Mélange les nombres disponibles
        return self.state.flatten()  # Retourne un vecteur 1D pour l'état

    def get_random_trefle(self):
        """Tire un nombre aléatoire unique de la liste des nombres disponibles."""
        if self.available_numbers:
            return self.available_numbers.pop()  # Retire et retourne un nombre unique
        else:
            self.done = True
            return -1  # Retourne une valeur spéciale pour indiquer qu'il n'y a plus de tuiles

    def is_valid_action(self, row, col, number):
        """Vérifie si une action est valide"""
        if number == -1:
            return False  # Si aucun nombre n'est disponible, l'action est invalide

        if self.state[row][col] != 0:
            return False  # Case déjà occupée

        # Vérification du tri croissant dans la colonne
        for i in range(self.rows):
            if self.state[i][col] != 0:  # Ignorer les zéros
                if i < row and self.state[i][col] >= number:
                    return False  # Le nombre doit être plus grand que ceux au-dessus
                if i > row and self.state[i][col] <= number:
                    return False  # Le nombre doit être plus petit que ceux en dessous

        # Vérification du tri croissant dans la ligne
        for j in range(self.cols):
            if self.state[row][j] != 0:  # Ignorer les zéros
                if j < col and self.state[row][j] >= number:
                    return False  # Le nombre doit être plus grand que ceux à gauche
                if j > col and self.state[row][j] <= number:
                    return False  # Le nombre doit être plus petit que ceux à droite

        # Vérification que le max de la ligne est inférieur au min de la ligne suivante
        if row < self.rows - 1:
            max_current_row = max([num for num in self.state[row] if num != 0], default=-float('inf'))
            min_next_row = min([num for num in self.state[row + 1] if num != 0], default=float('inf'))

            if number >= min_next_row or max_current_row >= number:
                return False

        return True

    def step(self, action):
        """Exécute l'action de l'agent"""
        if self.done:
            return self.state.flatten(), self.reward, self.done, {}

        row, col = divmod(action, self.cols)
        number = self.get_random_trefle()

        if number == -1:
            self.done = True
            return self.state.flatten(), self.reward, self.done, {}

        if self.is_valid_action(row, col, number):
            self.state[row][col] = number
            self.reward = 1
            if self.check_completion(row, col):
                self.reward += 10
        else:
            self.reward = -1

        if not self._can_play():
            self.done = True

        return self.state.flatten(), self.reward, self.done, {}

    def check_completion(self, row, col):
        """Vérifie si une ligne ou une colonne est complétée après une action"""
        row_complete = np.all(self.state[row] > 0) and np.all(np.diff(self.state[row]) > 0)
        col_complete = np.all(self.state[:, col] > 0) and np.all(np.diff(self.state[:, col]) > 0)

        return row_complete or col_complete

    def _can_play(self):
        """Vérifie s'il reste des actions valides possibles avec les nombres disponibles."""
        for i in range(self.rows):
            for j in range(self.cols):
                if self.state[i][j] == 0:  # Case vide
                    for number in self.available_numbers:  # Vérifie seulement les nombres disponibles
                        if self.is_valid_action(i, j, number):
                            return True  # Si au moins une action est valide
        return False  # Aucun placement valide n'est possible

    def render_agent(self):
        """Affiche la grille après l'action de l'agent"""
        print("État de la grille après l'action de l'agent :")
        print(self.state)
        print(f"Récompense actuelle de l'agent : {self.reward}\n")

    def render(self):
        """Affiche l'état général de la grille"""
        print(self.state)
        print(f"Récompense actuelle : {self.reward}")
        if self.done:
            print("Partie terminée")

    def graph_scores(self):
        """Affiche un graphique des scores cumulés par épisode"""
        import matplotlib.pyplot as plt
        plt.plot(self.list_scores)
        plt.title("Scores cumulés par épisode")
        plt.xlabel("Épisodes")
        plt.ylabel("Scores")
        plt.show()


import tkinter as tk
from tkinter import messagebox
import random

class LuckyNumbersGame:
    def __init__(self, master):
        self.master = master
        self.master.title("Lucky Numbers")
        self.size = 4  # Taille de la grille
        self.total_tiles = self.size * self.size
        self.numbers = list(range(1, self.total_tiles * 2 + 1))
        random.shuffle(self.numbers)
        self.cache_capacity = 3  # Capacité maximale du cache
        self.player_grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.ai_grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.player_cache = []  # Réserve du joueur
        self.ai_cache = []      # Réserve de l'IA
        self.create_widgets()
        self.current_tile = None
        self.turn = 'player'  # Pour gérer le tour du joueur et de l'IA
        self.place_initial_tiles()

    def place_initial_tiles(self):
        """Place les nombres initiaux sur la diagonale principale de chaque grille."""
        diagonal_positions = [(i, i) for i in range(self.size)]

        # Pour le joueur
        initial_numbers_player = sorted([self.numbers.pop() for _ in range(self.size)])
        for pos, num in zip(diagonal_positions, initial_numbers_player):
            row, col = pos
            self.player_grid[row][col] = num
            self.player_buttons[row][col].config(text=str(num), state='disabled', bg='light blue')

        # Pour l'IA
        initial_numbers_ai = sorted([self.numbers.pop() for _ in range(self.size)])
        for pos, num in zip(diagonal_positions, initial_numbers_ai):
            row, col = pos
            self.ai_grid[row][col] = num
            self.ai_labels[row][col].config(text=str(num), bg='light blue')

    def create_widgets(self):
        self.frame = tk.Frame(self.master)
        self.frame.pack()

        # Grille du joueur
        tk.Label(self.frame, text="Votre Grille", font=('Helvetica', 14, 'bold')).grid(row=0, column=0, columnspan=self.size)
        self.player_buttons = [[tk.Button(self.frame, width=4, height=2, font=('Helvetica', 12),
                                          command=lambda r=i, c=j: self.place_tile(r, c))
                                for j in range(self.size)] for i in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                self.player_buttons[i][j].grid(row=i+1, column=j, padx=2, pady=2)

        # Grille de l'IA
        tk.Label(self.frame, text="Grille de l'adversaire", font=('Helvetica', 14, 'bold')).grid(row=0, column=self.size+1, columnspan=self.size)
        self.ai_labels = [[tk.Label(self.frame, width=4, height=2, font=('Helvetica', 12), relief='sunken', bg='light grey')
                           for j in range(self.size)] for i in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                self.ai_labels[i][j].grid(row=i+1, column=self.size+1 + j, padx=2, pady=2)

        # Zone de pioche et cache
        self.draw_button = tk.Button(self.frame, text="Piocher une tuile", font=('Helvetica', 12), command=self.draw_tile)
        self.draw_button.grid(row=self.size+1, column=0, columnspan=self.size)

        self.cache_button = tk.Button(self.frame, text="Utiliser une tuile du cache", font=('Helvetica', 12), command=self.use_cache_tile)
        self.cache_button.grid(row=self.size+2, column=0, columnspan=self.size)

        self.add_to_cache_button = tk.Button(self.frame, text="Ajouter dans le cache", font=('Helvetica', 12), command=self.add_to_cache)
        self.add_to_cache_button.grid(row=self.size+3, column=0, columnspan=self.size)

        self.info_label = tk.Label(self.frame, text="À vous de jouer!", font=('Helvetica', 12))
        self.info_label.grid(row=self.size+4, column=0, columnspan=self.size*2)

        # Affichage du cache du joueur
        self.cache_label = tk.Label(self.frame, text="Cache : Vide", font=('Helvetica', 12))
        self.cache_label.grid(row=self.size+1, column=self.size+1, columnspan=self.size)

    def update_cache_label(self):
        if self.player_cache:
            cache_text = "Cache : " + ", ".join(map(str, self.player_cache))
        else:
            cache_text = "Cache : Vide"
        self.cache_label.config(text=cache_text)

    def draw_tile(self):
        if self.turn != 'player':
            self.info_label.config(text="Ce n'est pas votre tour.")
            return
        if not self.numbers:
            self.check_winner()
            return
        self.current_tile = self.numbers.pop()
        self.info_label.config(text=f"Tuile piochée : {self.current_tile}")

    def use_cache_tile(self):
        if self.turn != 'player':
            self.info_label.config(text="Ce n'est pas votre tour.")
            return
        if not self.player_cache:
            self.info_label.config(text="Votre cache est vide.")
            return
        # Sélectionner la première tuile du cache
        self.current_tile = self.player_cache.pop(0)
        self.info_label.config(text=f"Tuile du cache : {self.current_tile}")
        self.update_cache_label()

    def add_to_cache(self):
        if self.turn != 'player':
            self.info_label.config(text="Ce n'est pas votre tour.")
            return
        if self.current_tile is None:
            self.info_label.config(text="Vous n'avez pas de tuile à ajouter au cache.")
            return
        if len(self.player_cache) >= self.cache_capacity:
            messagebox.showinfo("Cache plein", "Votre cache est plein. Vous devez défausser une tuile du cache pour faire de la place.")
            self.discard_from_cache()
        self.player_cache.append(self.current_tile)
        self.update_cache_label()
        self.current_tile = None
        self.turn = 'ai'
        self.check_winner()
        self.master.after(500, self.ai_turn)

    def place_tile(self, row, col):
        if self.turn != 'player':
            self.info_label.config(text="Ce n'est pas votre tour.")
            return
        if self.current_tile is None:
            self.info_label.config(text="Vous devez piocher une tuile ou utiliser le cache d'abord.")
            return

        # Vérifier si le placement ou le remplacement est valide
        if self.player_grid[row][col] is None:
            valid = self.is_valid_placement(self.player_grid, row, col, self.current_tile)
        else:
            valid = self.is_valid_placement_with_replacement(self.player_grid, row, col, self.current_tile)

        if not valid:
            self.info_label.config(text="Placement invalide selon les règles. Essayez à nouveau.")
            return  # Si placement invalide, ne pas passer le tour ni changer la tuile.

        # Si on remplace une tuile, ajouter l'ancienne au cache
        if self.player_grid[row][col] is not None:
            if len(self.player_cache) >= self.cache_capacity:
                messagebox.showinfo("Cache plein", "Votre cache est plein. Vous devez défausser une tuile du cache pour faire de la place.")
                self.discard_from_cache()
            self.player_cache.append(self.player_grid[row][col])
            self.update_cache_label()

        # Placement de la tuile
        self.player_grid[row][col] = self.current_tile
        self.player_buttons[row][col].config(text=str(self.current_tile), state='disabled', disabledforeground='black')
        self.current_tile = None  # Réinitialiser la tuile courante après un placement réussi
        self.check_winner()
        self.turn = 'ai'  # Changer de tour
        self.master.after(500, self.ai_turn)  # Laisser un court délai avant le tour de l'IA


    def is_valid_placement(self, grid, row, col, number):
        """Vérifie si le placement respecte l'ordre croissant sur la ligne et la colonne."""
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
        old_value = grid[row][col]
        grid[row][col] = number
        valid = self.is_valid_placement(grid, row, col, number)
        grid[row][col] = old_value
        return valid

    def cannot_place_tile(self):
        """Si le joueur ne peut pas placer la tuile, il la met dans son cache ou doit défausser."""
        if len(self.player_cache) >= self.cache_capacity:
            messagebox.showinfo("Cache plein", "Votre cache est plein. Vous devez défausser une tuile du cache pour faire de la place.")
            self.discard_from_cache()
        self.player_cache.append(self.current_tile)
        self.update_cache_label()
        self.current_tile = None
        self.check_winner()
        self.turn = 'ai'
        self.master.after(500, self.ai_turn)

    def discard_from_cache(self):
        """Permet au joueur de défausser une tuile de son cache."""
        discarded_tile = random.choice(self.player_cache)
        self.player_cache.remove(discarded_tile)
        messagebox.showinfo("Défausse", f"La tuile {discarded_tile} a été défaussée de votre cache.")
        self.update_cache_label()

    def ai_turn(self):
        if self.turn != 'ai':
            return
        if not self.numbers and not self.ai_cache:
            self.check_winner()
            return
        if self.ai_cache:
            ai_tile = self.ai_cache.pop(0)
        elif self.numbers:
            ai_tile = self.numbers.pop()
        else:
            ai_tile = None

        if ai_tile is not None:
            positions = [(i, j) for i in range(self.size) for j in range(self.size)]
            random.shuffle(positions)
            placed = False
            for i, j in positions:
                if self.ai_grid[i][j] is None:
                    if self.is_valid_placement(self.ai_grid, i, j, ai_tile):
                        self.ai_grid[i][j] = ai_tile
                        self.ai_labels[i][j].config(text=str(ai_tile), bg='white')
                        placed = True
                        break
                else:
                    if self.is_valid_placement_with_replacement(self.ai_grid, i, j, ai_tile):
                        if len(self.ai_cache) >= self.cache_capacity:
                            self.ai_discard_from_cache()
                        self.ai_cache.append(self.ai_grid[i][j])
                        self.ai_grid[i][j] = ai_tile
                        self.ai_labels[i][j].config(text=str(ai_tile), bg='white')
                        placed = True
                        break
            if not placed:
                if len(self.ai_cache) >= self.cache_capacity:
                    self.ai_discard_from_cache()
                self.ai_cache.append(ai_tile)

        self.check_winner()
        self.turn = 'player'
        self.info_label.config(text="À vous de jouer!")

    def ai_discard_from_cache(self):
        discarded_tile = random.choice(self.ai_cache)
        self.ai_cache.remove(discarded_tile)

    def check_winner(self):
        if all(all(cell is not None for cell in row) for row in self.player_grid):
            self.info_label.config(text="Vous avez gagné!")
            self.end_game()
            return

        if all(all(cell is not None for cell in row) for row in self.ai_grid):
            self.info_label.config(text="L'adversaire a gagné!")
            self.end_game()
            return

        if not self.numbers and not self.player_cache and not self.ai_cache:
            self.info_label.config(text="Match nul!")
            self.end_game()

    def end_game(self):
        for row in self.player_buttons:
            for button in row:
                button.config(state='disabled')
        self.draw_button.config(state="disabled")
        self.cache_button.config(state="disabled")
        self.add_to_cache_button.config(state="disabled")

    def pass_turn(self):
        if self.turn != 'player':
            self.info_label.config(text="Ce n'est pas votre tour.")
            return
        if self.current_tile is None:
            self.info_label.config(text="Vous n'avez pas de tuile à mettre dans le cache.")
            return
        self.cannot_place_tile()

