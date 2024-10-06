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
        """Vérifie si une action est valide en respectant le tri croissant dans les lignes et colonnes, et globalement entre les lignes."""
        if number == -1:
            return False  # Si aucun nombre n'est disponible, l'action est invalide

        if self.state[row][col] != 0:
            return False  # Case déjà occupée

        # Vérification du tri croissant dans la colonne
        for i in range(self.rows):
            if self.state[i][col] != 0:  # Ignorer les zéros
                if i < row and self.state[i][col] >= number:
                    # print(f"Nombre {number} à la position ({row}, {col}) est inférieur ou égal à {self.state[i][col]} au-dessus dans la colonne")
                    return False  # Le nombre doit être plus grand que ceux au-dessus
                if i > row and self.state[i][col] <= number:
                    # print(f"Nombre {number} à la position ({row}, {col}) est supérieur ou égal à {self.state[i][col]} en dessous dans la colonne")
                    return False  # Le nombre doit être plus petit que ceux en dessous

        # Vérification du tri croissant dans la ligne
        for j in range(self.cols):
            if self.state[row][j] != 0:  # Ignorer les zéros
                if j < col and self.state[row][j] >= number:
                    # print(f"Nombre {number} à la position ({row}, {col}) est inférieur ou égal à {self.state[row][j]} à gauche dans la ligne")
                    return False  # Le nombre doit être plus grand que ceux à gauche
                if j > col and self.state[row][j] <= number:
                    # print(f"Nombre {number} à la position ({row}, {col}) est supérieur ou égal à {self.state[row][j]} à droite dans la ligne")
                    return False  # Le nombre doit être plus petit que ceux à droite

        # Vérification que le max de la ligne est inférieur au min de la ligne suivante
        if row < self.rows - 1:  # S'assurer qu'on ne dépasse pas les limites de la grille
            max_current_row = max([num for num in self.state[row] if num != 0], default=-float('inf'))  # Max de la ligne courante (ignorer les zéros)
            min_next_row = min([num for num in self.state[row + 1] if num != 0], default=float('inf'))  # Min de la ligne suivante (ignorer les zéros)

            if number >= min_next_row:
                # print(f"Le nombre {number} à la position ({row}, {col}) est supérieur ou égal au minimum de la ligne suivante ({min_next_row})")
                return False
            if max_current_row >= number:
                # print(f"Le max de la ligne actuelle ({max_current_row}) est supérieur ou égal au nombre {number}")
                return False 

        return True


    def step(self, action):
        """Exécute l'action de l'agent"""
        if self.done:
            return self.state.flatten(), self.reward, self.done, {}

        # L'agent joue en premier
        row, col = divmod(action, self.cols)
        number = self.get_random_trefle()

        if number == -1:
            self.done = True
            return self.state.flatten(), self.reward, self.done, {}

        # Si l'action est valide
        if self.is_valid_action(row, col, number):
            self.state[row][col] = number
            self.reward = 1  # Récompense pour un coup valide
            # print(f"\nAgent a placé {number} à la position ({row}, {col})")
            # self.render_agent()  # Afficher l'état après l'action de l'agent
            if self.check_completion(row, col):  # Vérifie si une ligne ou colonne est complétée
                self.reward += 10  # Bonus pour complétion
                print("Agent a complété une ligne ou colonne!")
        else:
            self.reward = -1  # Pénalité pour un coup invalide
            print(f"\nAgent a tenté un coup invalide à la position ({row}, {col})")

        # Si la grille est pleine ou aucune action possible
        if not self._can_play():
            self.done = True
            return self.state.flatten(), self.reward, self.done, {}

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
