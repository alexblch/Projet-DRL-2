import numpy as np
from Environnements.luckynumbergame import LuckyNumbersGame
from Environnements.luckynumberenv import LuckyNumbersEnv
from Environnements.luckynumberrand import LuckyNumbersGameRandConsole  
from Agent.dqn import DQNAgent
from Agent.dqn_with_replay import DQNAgentWithReplay
from Agent.dqn_with_p_replay import DQNAgentWithPrioritizedReplay
from Agent.reinforce import REINFORCEAgent  
import os
import tkinter as tk

def clear_screen():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def choose_algorithm(state_size, action_size):
    print("Veuillez choisir l'algorithme de renforcement :")
    print("1 - DQN")
    print("2 - DQN avec Experience Replay")
    print("3 - DQN avec Prioritized Experience Replay")
    print("4 - REINFORCE")
    print("5 - Agent aléatoire (Random)")

    choice = input("Entrez votre choix (1 à 5) : ")

    if choice == '1':
        return DQNAgent(state_size, action_size), 'DQN'
    elif choice == '2':
        return DQNAgentWithReplay(state_size, action_size), 'DQN avec Experience Replay'
    elif choice == '3':
        return DQNAgentWithPrioritizedReplay(state_size, action_size), 'DQN avec Prioritized Experience Replay'
    elif choice == '4':
        return REINFORCEAgent(state_size, action_size), 'REINFORCE'  # Si REINFORCE est implémenté
    elif choice == '5':
        return LuckyNumbersGameRandConsole(), 'Agent aléatoire'  # Retourne l'agent random
    else:
        print("Choix invalide. Utilisation de DQN par défaut.")
        return DQNAgent(state_size, action_size), 'DQN'

def choose_game():
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
        print("Fonctionnalité non implémentée pour GridWorld.")
        return None, None
    else:
        print("Choix invalide. Retour au menu principal.")
        return None, None

def main():
    action, game = choose_game()
    victory = []

    if action == 'play_classic' and game == 'LuckyNumber':
        print("Lancement du jeu Lucky Number classique contre un agent aléatoire.")
        root = tk.Tk()
        game_instance = LuckyNumbersGame(root)
        root.mainloop()

    elif action == 'train' and game == 'LuckyNumber':
        print("Entraînement de l'agent pour Lucky Number.")
        env = LuckyNumbersEnv()
        state_size = env.state_description().shape[0]
        action_size = env.action_mask().shape[0]

        agent, algo = choose_algorithm(state_size, action_size)
        batch_size = 32
        EPISODES = int(input("Entrez le nombre d'épisodes pour l'entraînement (par ex. 1000) : "))

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
                action = agent.choose_action(state, available_actions, action_mask)
                try:
                    env.step(action)
                except ValueError as e:
                    print(f"Action invalide: {e}")
                    done = True
                    reward = -1
                    agent.learn(state, action, reward, state, done, action_mask)
                    break

                next_state = env.state_description()
                reward = env.score()
                done = env.is_game_over()
                action_mask_next = env.action_mask()
                agent.learn(state, action, reward, next_state, done, action_mask_next)
                state = next_state
                total_reward += reward

            if total_reward > 0:
                victory.append(1)
            elif total_reward == 0:
                victory.append(0)
            else:
                victory.append(-1)

            print(f"Episode {e + 1}/{EPISODES}, Récompense Totale: {total_reward}, Epsilon: {agent.epsilon}")

        print("Entraînement terminé.")
        print("Nombre de victoires : ", victory.count(1))
        print("Nombre de défaites : ", victory.count(-1))
        print("Nombre de match nul : ", victory.count(0))

    else:
        print("Option non reconnue ou fonctionnalité non implémentée.")

if __name__ == "__main__":
    main()
