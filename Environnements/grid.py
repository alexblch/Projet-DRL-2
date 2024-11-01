import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self):
        self.rows = 5
        self.cols = 5
        self.state = np.zeros((self.rows, self.cols))  # Grille 5x5
        self.agent_pos_col = 0
        self.agent_pos_row = 0
        self.done = False
        self.action_space = 4  # 4 actions possibles (gauche, droite, haut, bas)
        self.reward = 0
        self.list_scores = []  # Liste des scores pour chaque épisode

    def reset(self):
        """Réinitialise l'état du jeu pour un nouvel épisode"""
        self.state = np.zeros((self.rows, self.cols))
        self.agent_pos_col = 0
        self.agent_pos_row = 0
        self.done = False
        self.reward = 0
        return self.state_description()  # Retourne l'état sous forme de vecteur 1D

    def step(self, action: int):
        """Exécute l'action de l'agent"""
        if self.done:
            return self.state_description(), self.reward, self.done, {}

        # Mouvement de l'agent
        if action == 0 and self.agent_pos_col > 0:  # Gauche
            self.agent_pos_col -= 1
        elif action == 1 and self.agent_pos_col < self.cols - 1:  # Droite
            self.agent_pos_col += 1
        elif action == 2 and self.agent_pos_row > 0:  # Haut
            self.agent_pos_row -= 1
        elif action == 3 and self.agent_pos_row < self.rows - 1:  # Bas
            self.agent_pos_row += 1

        # Mettre à jour la récompense
        self.reward = self.score()

        # Si l'agent atteint une position de fin, la partie est terminée
        if self.is_game_over():
            self.done = True
            self.list_scores.append(self.reward)  # Enregistrer le score pour cet épisode

        return self.state_description(), self.reward, self.done, {}

    def is_game_over(self) -> bool:
        """Vérifie si le jeu est terminé (si l'agent atteint une position finale)"""
        return (self.agent_pos_row == 4 and self.agent_pos_col == 4) or (self.agent_pos_row == 0 and self.agent_pos_col == 4)

    def score(self) -> float:
        """Calcule la récompense en fonction de la position de l'agent"""
        if self.agent_pos_row == 0 and self.agent_pos_col == 4:
            return -3.0
        elif self.agent_pos_row == 4 and self.agent_pos_col == 4:
            return 1.0
        elif self.agent_pos_row > 4 or self.agent_pos_row < 0 or self.agent_pos_col > 4 or self.agent_pos_col < 0:
            return -1.0
        return 0.0

    def state_description(self) -> np.ndarray:
        """Retourne la description de l'état sous forme de vecteur 1D"""
        # Encode la position de l'agent sous forme de vecteur 1D
        state = np.zeros((self.rows, self.cols))
        state[self.agent_pos_row, self.agent_pos_col] = 1  # Position de l'agent marquée par un 1
        return state.flatten()

    def render(self):
        """Affiche la grille actuelle"""
        for num_row in range(self.rows):
            for num_col in range(self.cols):
                print('X' if self.agent_pos_col == num_col and self.agent_pos_row == num_row else '_', end=' ')
            print()
        print(f"Récompense actuelle : {self.reward}")
        if self.done:
            print("Partie terminée")

    def graph_scores(self):
        """Affiche un graphique des scores cumulés par épisode"""
        plt.plot(self.list_scores)
        plt.title("Scores cumulés par épisode")
        plt.xlabel("Épisodes")
        plt.ylabel("Scores")
        plt.show()

    def available_actions_ids(self):
        """Retourne les actions possibles sous forme d'ID en fonction de la position actuelle de l'agent"""
        actions = []
        if self.agent_pos_col > 0:  # Peut aller à gauche
            actions.append(0)
        if self.agent_pos_col < 4:  # Peut aller à droite
            actions.append(1)
        if self.agent_pos_row > 0:  # Peut aller en haut
            actions.append(2)
        if self.agent_pos_row < 4:  # Peut aller en bas
            actions.append(3)
        return np.array(actions)

    def action_mask(self) -> np.ndarray:
        """Retourne un masque des actions valides (1 pour valide, 0 pour invalide)"""
        mask = np.zeros(self.action_space, dtype=np.float32)
        if self.agent_pos_col > 0:  # Gauche
            mask[0] = 1
        if self.agent_pos_col < self.cols - 1:  # Droite
            mask[1] = 1
        if self.agent_pos_row > 0:  # Haut
            mask[2] = 1
        if self.agent_pos_row < self.rows - 1:  # Bas
            mask[3] = 1
        return mask
