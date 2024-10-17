import numpy as np
from Environnements.luckynumbergame import LuckyNumbersGame
from DQN.dqn import DQNAgent
from Environnements.luckynumberenv import LuckyNumbersEnv
# from reinforce import REINFORCEAgent  # Ajoutez votre implémentation REINFORCE ici
import os
import tkinter as tk  # Importer Tkinter

def clear_screen():
    """Efface l'écran (compatible Windows et Unix)."""
    if os.name == 'nt':
        os.system('cls')  # Windows
    else:
        os.system('clear')  # Unix/Linux/Mac

def choose_algorithm(state_size, action_size):
    """Menu pour choisir l'algorithme de renforcement."""
    print("Veuillez choisir l'algorithme de renforcement :")
    print("1 - DQN")
    print("2 - REINFORCE")
    choice = input("Entrez votre choix (1 ou 2) : ")

    if choice == '1':
        algorithm = 'DQN'
        return DQNAgent(state_size, action_size), algorithm
    elif choice == '2':
        print("REINFORCE n'est pas encore implémenté. Utilisation de DQN par défaut.")
        algorithm = 'DQN'
        return DQNAgent(state_size, action_size), algorithm
    else:
        print("Choix invalide. Utilisation de DQN par défaut.")
        algorithm = 'DQN'
        return DQNAgent(state_size, action_size), algorithm

def choose_game():
    """Menu pour choisir le jeu et l'action à effectuer."""
    print("Veuillez choisir le jeu :")
    print("1 - Lucky Number classique vs agent random")
    print("2 - Lucky Number - Entraîner un agent")
    print("3 - GridWorld")
    game_choice = input("Entrez votre choix (1, 2 ou 3) : ")

    if game_choice == '1':
        return 'play_classic', 'LuckyNumber'
    elif game_choice == '2':
        return 'train', 'LuckyNumber'
    elif game_choice == '3':
        print("GridWorld sélectionné.")
        # Vous pouvez ajouter des options pour GridWorld ici
        print("Fonctionnalité non implémentée pour GridWorld.")
        return None, None
    else:
        print("Choix invalide. Retour au menu principal.")
        return None, None

def main():
    action, game = choose_game()
    if action == 'play_classic' and game == 'LuckyNumber':
        # Lancer le jeu Lucky Number classique contre un agent aléatoire
        print("Lancement du jeu Lucky Number classique contre un agent aléatoire.")
        root = tk.Tk()  # Créer la fenêtre principale Tkinter
        game_instance = LuckyNumbersGame(root)  # Passer 'root' à LuckyNumbersGame
        root.mainloop()  # Lancer la boucle principale Tkinter
    elif action == 'train' and game == 'LuckyNumber':
        # Entraîner l'agent pour Lucky Number
        print("Entraînement de l'agent pour Lucky Number.")
        env = LuckyNumbersEnv()
        state_size = env.state_description().shape[0]
        action_size = env.action_mask().shape[0]

        agent, algo = choose_algorithm(state_size, action_size)  # Sélectionner l'algorithme
        batch_size = 32
        EPISODES = int(input("Entrez le nombre d'épisodes pour l'entraînement (par ex. 1000) : "))

        # Vérifie si le répertoire 'models' existe, sinon le crée
        if not os.path.exists(f'models/saved_models'):
            os.makedirs(f'models/saved_models')

        for e in range(EPISODES):
            env.reset()
            state = env.state_description()
            done = env.is_game_over()

            total_reward = 0

            while not done:
                available_actions = env.available_actions_ids()
                action_mask = env.action_mask()

                # L'agent choisit une action
                action = agent.choose_action(state, available_actions, action_mask)

                # Exécute l'action
                try:
                    env.step(action)
                except ValueError as e:
                    print(f"Action invalide: {e}")
                    done = True
                    reward = -1  # Pénalité pour action invalide
                    agent.learn(state, action, reward, state, done, action_mask)
                    break

                # Obtenir le nouvel état et la récompense
                next_state = env.state_description()
                reward = env.score()
                done = env.is_game_over()
                action_mask_next = env.action_mask()

                # L'agent apprend de l'expérience
                agent.learn(state, action, reward, next_state, done, action_mask_next)

                state = next_state
                total_reward += reward

            print(f"Episode {e + 1}/{EPISODES}, Récompense Totale: {total_reward}, Epsilon: {agent.epsilon}")

        print("Entraînement terminé.")
    else:
        print("Option non reconnue ou fonctionnalité non implémentée.")

if __name__ == "__main__":
    main()
