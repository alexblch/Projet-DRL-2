from numba import njit
import numpy as np
import random
import tkinter as tk
from tkinter import messagebox


class LuckyNumbersGame:
    def __init__(self, master):
        self.master = master
        self.master.title("Lucky Numbers")
        self.size = 4  # Taille de la grille
        self.total_tiles = self.size * self.size
        # Créer une liste de nombres de 1 à 20, deux fois chacun
        self.numbers = [num for num in range(1, 21)] * 2  # Nombres entre 1 et 20, deux fois
        random.shuffle(self.numbers)
        self.player_grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.ai_grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.shared_cache = []  # Cache partagé sans limite de capacité
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
            self.player_buttons[row][col].config(text=str(num), state='normal', bg='light blue')
            # Les tuiles de la diagonale sont placées mais restent modifiables

        # Pour l'IA
        initial_numbers_ai = sorted([self.numbers.pop() for _ in range(self.size)])
        for pos, num in zip(diagonal_positions, initial_numbers_ai):
            row, col = pos
            self.ai_grid[row][col] = num
            self.ai_labels[row][col].config(text=str(num), bg='light blue')

    def create_widgets(self):
        self.frame = tk.Frame(self.master)
        self.frame.pack(padx=20, pady=20)

        # Grille du joueur
        tk.Label(self.frame, text="Votre Grille", font=('Helvetica', 16, 'bold')).grid(row=0, column=0, columnspan=self.size)
        self.player_buttons = [[tk.Button(self.frame, width=6, height=3, font=('Helvetica', 14),
                                          command=lambda r=i, c=j: self.place_tile(r, c))
                                for j in range(self.size)] for i in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                self.player_buttons[i][j].grid(row=i+1, column=j, padx=5, pady=5)

        # Espace entre les deux grilles
        tk.Label(self.frame, text="", width=2).grid(row=1, column=self.size, rowspan=self.size)

        # Grille de l'IA
        tk.Label(self.frame, text="Grille de l'adversaire", font=('Helvetica', 16, 'bold')).grid(row=0, column=self.size+1, columnspan=self.size)
        self.ai_labels = [[tk.Label(self.frame, width=6, height=3, font=('Helvetica', 14), relief='sunken', bg='light grey')
                           for j in range(self.size)] for i in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                self.ai_labels[i][j].grid(row=i+1, column=self.size+1 + j, padx=5, pady=5)

        # Zone de pioche
        self.draw_button = tk.Button(self.frame, text="Piocher une tuile", font=('Helvetica', 14), command=self.draw_tile)
        self.draw_button.grid(row=self.size+1, column=0, columnspan=self.size*2, pady=10)

        # Bouton ajouter dans le cache
        self.add_to_cache_button = tk.Button(self.frame, text="Ajouter dans le cache", font=('Helvetica', 14), command=self.add_to_cache)
        self.add_to_cache_button.grid(row=self.size+2, column=0, columnspan=self.size*2, pady=5)

        # Affichage du cache partagé
        tk.Label(self.frame, text="Cache :", font=('Helvetica', 14, 'bold')).grid(row=self.size+3, column=0, columnspan=self.size*2)
        self.cache_frame = tk.Frame(self.frame)
        self.cache_frame.grid(row=self.size+4, column=0, columnspan=self.size*2, pady=5)

        # Message d'information
        self.info_label = tk.Label(self.frame, text="À vous de jouer!", font=('Helvetica', 14))
        self.info_label.grid(row=self.size+5, column=0, columnspan=self.size*2, pady=10)

        # Bouton pour redémarrer le jeu
        self.restart_button = tk.Button(self.frame, text="Rejouer", font=('Helvetica', 14), command=self.restart_game)
        self.restart_button.grid(row=self.size+6, column=self.size*2-1, sticky='e', pady=10)

    def update_cache_display(self):
        """Met à jour l'affichage du cache partagé."""
        # Supprimer les anciens boutons du cache
        for widget in self.cache_frame.winfo_children():
            widget.destroy()

        self.cache_buttons = []

        # Créer un bouton pour chaque tuile du cache
        for idx, tile in enumerate(self.shared_cache):
            btn = tk.Button(self.cache_frame, text=str(tile), font=('Helvetica', 14), width=4,
                            command=lambda t=tile: self.select_cache_tile(t))
            btn.pack(side='left', padx=2)
            self.cache_buttons.append(btn)

        # Si le cache est vide, afficher un label "Vide"
        if not self.shared_cache:
            tk.Label(self.cache_frame, text="Vide", font=('Helvetica', 14)).pack()

    def select_cache_tile(self, tile):
        """Sélectionne une tuile du cache."""
        if self.turn != 'player':
            self.info_label.config(text="Ce n'est pas votre tour.")
            return
        if tile not in self.shared_cache:
            self.info_label.config(text="Tuile non disponible.")
            return

        # Si le joueur a déjà une tuile en main, lui proposer de l'échanger avec celle du cache
        if self.current_tile is not None:
            response = messagebox.askyesno("Échanger la tuile", f"Vous avez déjà la tuile {self.current_tile} en main. Voulez-vous l'échanger avec la tuile {tile} du cache ?")
            if response:
                self.shared_cache.remove(tile)
                self.shared_cache.append(self.current_tile)
                self.current_tile = tile
                self.info_label.config(text=f"Tuile du cache sélectionnée : {self.current_tile}")
                self.update_cache_display()
            else:
                self.info_label.config(text="Action annulée.")
        else:
            self.shared_cache.remove(tile)
            self.current_tile = tile
            self.info_label.config(text=f"Tuile du cache sélectionnée : {self.current_tile}")
            self.update_cache_display()

    def draw_tile(self):
        if self.turn != 'player':
            self.info_label.config(text="Ce n'est pas votre tour.")
            return
        if self.current_tile is not None:
            self.info_label.config(text="Vous avez déjà une tuile en main.")
            return
        if not self.numbers:
            self.check_winner()
            return
        self.current_tile = self.numbers.pop()
        self.info_label.config(text=f"Tuile piochée : {self.current_tile}")

    def add_to_cache(self):
        if self.turn != 'player':
            self.info_label.config(text="Ce n'est pas votre tour.")
            return
        if self.current_tile is None:
            self.info_label.config(text="Vous n'avez pas de tuile à ajouter au cache.")
            return
        self.shared_cache.append(self.current_tile)
        self.update_cache_display()
        self.current_tile = None
        self.turn = 'ai'
        self.check_winner()
        self.master.after(500, self.ai_turn)

    def place_tile(self, row, col):
        if self.turn != 'player':
            self.info_label.config(text="Ce n'est pas votre tour.")
            return
        if self.current_tile is None:
            self.info_label.config(text="Vous devez piocher une tuile ou sélectionner une tuile du cache.")
            return

        # Vérifier si le nombre est déjà dans la grille (excluant la case actuelle si on remplace)
        if self.number_in_grid(self.player_grid, self.current_tile, exclude_position=(row, col)):
            self.info_label.config(text="Vous ne pouvez pas avoir des nombres identiques dans votre grille.")
            return

        # Vérifier si le remplacement est possible
        can_replace = self.player_grid[row][col] is not None

        # Vérifier si le placement est valide
        if can_replace:
            valid = self.is_valid_placement_with_replacement(self.player_grid, row, col, self.current_tile)
        else:
            valid = self.is_valid_placement(self.player_grid, row, col, self.current_tile)

        if not valid:
            self.info_label.config(text="Placement invalide selon les règles. Essayez à nouveau.")
            return

        # Si on remplace une tuile, l'ancienne va dans le cache
        if can_replace:
            old_number = self.player_grid[row][col]
            self.shared_cache.append(old_number)
            self.update_cache_display()

        # Placement de la tuile
        self.player_grid[row][col] = self.current_tile
        self.player_buttons[row][col].config(text=str(self.current_tile), state='normal', bg='SystemButtonFace')
        self.current_tile = None  # Réinitialiser la tuile courante après un placement réussi
        self.check_winner()
        self.turn = 'ai'  # Changer de tour
        self.master.after(500, self.ai_turn)

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

    def number_in_grid(self, grid, number, exclude_position=None):
        """Vérifie si un nombre est déjà présent dans la grille."""
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) != exclude_position and grid[i][j] == number:
                    return True
        return False

    def ai_turn(self):
        if self.turn != 'ai':
            return
        if not self.numbers and not self.shared_cache:
            self.check_winner()
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
            positions = self.get_valid_positions(self.ai_grid, ai_tile)
            if positions:
                i, j = random.choice(positions)
                # Si l'IA remplace une tuile, elle ajoute l'ancienne au cache
                if self.ai_grid[i][j] is not None:
                    old_number = self.ai_grid[i][j]
                    self.shared_cache.append(old_number)
                    self.update_cache_display()
                self.ai_grid[i][j] = ai_tile
                self.ai_labels[i][j].config(text=str(ai_tile), bg='white')
            else:
                self.shared_cache.append(ai_tile)
                self.update_cache_display()

        self.check_winner()
        self.turn = 'player'
        self.info_label.config(text="À vous de jouer!")

    def get_valid_positions(self, grid, number):
        """Retourne une liste des positions valides pour placer le nombre donné, sans créer de doublons."""
        # Vérifier si le nombre est déjà présent dans la grille
        if self.number_in_grid(grid, number):
            return []

        valid_positions = []
        for i in range(self.size):
            for j in range(self.size):
                if grid[i][j] is None:
                    if self.is_valid_placement(grid, i, j, number):
                        valid_positions.append((i, j))
                else:
                    if self.is_valid_placement_with_replacement(grid, i, j, number):
                        # S'assurer que le remplacement n'introduit pas de doublon
                        if not self.number_in_grid(grid, number, exclude_position=(i, j)):
                            valid_positions.append((i, j))
        return valid_positions

    def check_winner(self):
        player_full = all(all(cell is not None for cell in row) for row in self.player_grid)
        ai_full = all(all(cell is not None for cell in row) for row in self.ai_grid)

        if player_full and ai_full:
            self.info_label.config(text="Match nul!")
            self.end_game("Match nul")
            return
        elif player_full:
            self.info_label.config(text="Vous avez gagné!")
            self.end_game("Victoire")
            return
        elif ai_full:
            self.info_label.config(text="L'adversaire a gagné!")
            self.end_game("Défaite")
            return

        if not self.numbers and not self.shared_cache:
            player_score = sum(sum(cell for cell in row if cell is not None) for row in self.player_grid)
            ai_score = sum(sum(cell for cell in row if cell is not None) for row in self.ai_grid)
            if player_score < ai_score:
                self.info_label.config(text="Vous avez gagné!")
                self.end_game("Victoire")
            elif player_score > ai_score:
                self.info_label.config(text="L'adversaire a gagné!")
                self.end_game("Défaite")
            else:
                self.info_label.config(text="Match nul!")
                self.end_game("Match nul")

    def end_game(self, result):
        for row in self.player_buttons:
            for button in row:
                button.config(state='disabled')
        self.draw_button.config(state="disabled")
        self.add_to_cache_button.config(state="disabled")
        # Afficher un message de fin de partie
        messagebox.showinfo("Fin de la partie", f"{result}!")
        # Laisser le bouton de redémarrage actif
        self.restart_button.config(state="normal")

    def restart_game(self):
        """Redémarre le jeu en réinitialisant toutes les variables et l'interface."""
        # Réinitialiser les variables du jeu
        self.numbers = [num for num in range(1, 21)] * 2  # Nombres de 1 à 20, deux fois
        random.shuffle(self.numbers)
        self.shared_cache = []
        self.current_tile = None
        self.turn = 'player'
        self.player_grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.ai_grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.info_label.config(text="À vous de jouer!")
        self.update_cache_display()

        # Réinitialiser les boutons de la grille du joueur
        for i in range(self.size):
            for j in range(self.size):
                self.player_buttons[i][j].config(text="", state='normal', bg='SystemButtonFace')

        # Réinitialiser les labels de la grille de l'IA
        for i in range(self.size):
            for j in range(self.size):
                self.ai_labels[i][j].config(text="", bg='light grey')

        # Réactiver les boutons
        self.draw_button.config(state="normal")
        self.add_to_cache_button.config(state="normal")

        # Placer les tuiles initiales
        self.place_initial_tiles()

if __name__ == "__main__":
    root = tk.Tk()
    game = LuckyNumbersGame(root)
    root.mainloop()
