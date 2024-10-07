import numpy as np
from Environnements.luckynumber import LuckyNumberEnv, LuckyNumbersGame
from DQN.dqn import DQNAgent
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
    print("1 - Lucky Number")
    print("2 - GridWorld")
    game_choice = input("Entrez votre choix (1 ou 2) : ")

    if game_choice == '1':
        print("Lucky Number sélectionné.")
        print("Voulez-vous jouer contre un agent aléatoire ou entraîner un agent ?")
        print("1 - Jouer contre un agent aléatoire")
        print("2 - Entraîner un agent")
        action_choice = input("Entrez votre choix (1 ou 2) : ")
        if action_choice == '1':
            return 'play', 'LuckyNumber'
        elif action_choice == '2':
            return 'train', 'LuckyNumber'
        else:
            print("Choix invalide. Retour au menu principal.")
            return None, None
    elif game_choice == '2':
        print("GridWorld sélectionné.")
        # Vous pouvez ajouter des options pour GridWorld ici
        print("Fonctionnalité non implémentée pour GridWorld.")
        return None, None
    else:
        print("Choix invalide. Retour au menu principal.")
        return None, None

def main():
    action, game = choose_game()
    if action == 'play' and game == 'LuckyNumber':
        # Lancer le jeu contre un agent aléatoire
        print("Lancement du jeu Lucky Number contre un agent aléatoire.")
        root = tk.Tk()  # Créer la fenêtre principale Tkinter
        game_instance = LuckyNumbersGame(root)  # Passer 'root' à LuckyNumbersGame
        root.mainloop()  # Lancer la boucle principale Tkinter
    elif action == 'train' and game == 'LuckyNumber':
        # Entraîner l'agent pour Lucky Number
        print("Entraînement de l'agent pour Lucky Number.")
        env = LuckyNumberEnv()
        state_size = env.rows * env.cols
        action_size = env.action_space

        agent, algo = choose_algorithm(state_size, action_size)  # Sélectionner l'algorithme
        batch_size = 32
        EPISODES = int(input("Entrez le nombre d'épisodes pour l'entraînement (par ex. 1000) : "))

        # Vérifie si le répertoire 'models' existe, sinon le crée
        if not os.path.exists(f'{algo}/models'):
            os.makedirs(f'{algo}/models')

        for e in range(EPISODES):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            total_reward = 0

            for time in range(500):
                action = agent.act(state, env)  # L'agent agit selon l'algorithme choisi
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, state_size])

                agent.remember(state, action, reward, next_state, done)
                state = next_state

                if done:
                    print(f"Partie terminée après {time+1} tours.")
                    agent.update_target_model()
                    print(f"Episode: {e+1}/{EPISODES}, Score: {total_reward}, Epsilon: {agent.epsilon}")
                    env.list_scores.append(time+1)
                    break

                if len(agent.memory) > batch_size:
                    agent.replay()

            if e % 50 == 0:
                # Sauvegarder dans un dossier spécifique à l'algorithme choisi
                agent.save(f"{algo}/models/model_weights_episode_{e}.h5")

        env.graph_scores()
        print("Entraînement terminé.")
    else:
        print("Option non reconnue ou fonctionnalité non implémentée.")

if __name__ == "__main__":
    main()
