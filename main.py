import numpy as np
import os
import matplotlib.pyplot as plt
import tkinter as tk
from Environnements.luckynumbergame import LuckyNumbersGame
from Environnements.luckynumberenv import LuckyNumbersEnv
from Environnements.grid import GridWorld
from Environnements.luckynumberrand import LuckyNumbersGameRandConsole, play_n_games
from Agent.dqn import DQNAgent
from Agent.dqn_with_replay import DQNAgentWithReplay
from Agent.dqn_with_p_replay import DQNAgentWithPrioritizedReplay
from Agent.ppo import PPOAgent
from Agent.reinforce import REINFORCEAgent

def clear_screen():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def choose_algorithm(state_size, action_size, env):
    print("Veuillez choisir l'algorithme de renforcement :")
    print("1 - DQN")
    print("2 - DQN avec Experience Replay")
    print("3 - DQN avec Prioritized Experience Replay")
    print("4 - PPO")
    print("5 - REINFORCE")
    print("6 - Agent aléatoire (Random)")

    choice = input("Entrez votre choix (1 à 6) : ")

    if choice == '1':
        return DQNAgent(state_size, action_size, env), 'DQN'
    elif choice == '2':
        return DQNAgentWithReplay(env), 'DQN avec Experience Replay'  # En passant `env`
    elif choice == '3':
        return DQNAgentWithPrioritizedReplay(env), 'DQN avec Prioritized Experience Replay'  # En passant `env`
    elif choice == '4':
        return PPOAgent(env), 'PPO'  # En passant `env`
    elif choice == '5':
        return REINFORCEAgent(env), 'REINFORCE'  # En passant `env`
    elif choice == '6':
        return LuckyNumbersGameRandConsole(), 'Agent aléatoire'
    else:
        print("Choix invalide. Utilisation de DQN par défaut.")
        return DQNAgent(state_size, action_size, env), 'DQN'  # En passant `env`


def choose_game():
    print("Veuillez choisir le jeu :")
    print("1 - Lucky Number classique contre un agent aléatoire")
    print("2 - Lucky Number - Entraîner un agent")
    print("3 - GridWorld")
    print("4 - Lucky Number IA vs IA (Aléatoire)")

    game_choice = input("Entrez votre choix (1 à 4) : ")

    if game_choice == '1':
        return 'play_classic', 'LuckyNumber'
    elif game_choice == '2':
        return 'train', 'LuckyNumber'
    elif game_choice == '3':
        return 'train', 'GridWorld'
    elif game_choice == '4':
        return 'random_vs_random', 'LuckyNumber'
    else:
        print("Choix invalide. Retour au menu principal.")
        return None, None

def plot_training_rewards(episode_rewards, window=10):
    """Affiche la récompense cumulée par épisode avec une moyenne mobile pour visualiser l'entraînement."""
    # Calculer la moyenne mobile
    moving_avg_rewards = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Récompense cumulée par épisode', color='skyblue')
    plt.plot(range(window - 1, len(episode_rewards)), moving_avg_rewards, label=f'Moyenne mobile ({window} épisodes)', color='blue')
    plt.xlabel('Épisodes')
    plt.ylabel('Récompense cumulée (G)')
    plt.title('Évolution de la récompense cumulée par épisode')
    plt.legend()
    plt.grid()
    plt.show()
    
def plot_epsilon(epsilon):
    plt.figure(figsize=(10, 5))
    plt.plot(epsilon, label='Epsilon')
    plt.xlabel('Épisodes')
    plt.ylabel('Epsilon')
    plt.title('Évolution de Epsilon')
    plt.legend()
    plt.grid()
    plt.show()


def main():
    action, game = choose_game()
    victory = []
    episode_rewards = []
    epsilon = np.array([])

    if action == 'play_classic' and game == 'LuckyNumber':
        print("Lancement du jeu Lucky Number classique contre un agent aléatoire.")
        root = tk.Tk()
        game_instance = LuckyNumbersGame(root)
        root.mainloop()

    elif action == 'random_vs_random' and game == 'LuckyNumber':
        print("Lancement du mode Lucky Number IA vs IA (Aléatoire)")
        n = int(input("Entrez le nombre de parties à jouer : "))
        play_n_games(n)

    elif action == 'train':
        if game == 'LuckyNumber':
            print("Entraînement de l'agent pour Lucky Number.")
            env = LuckyNumbersEnv()
        elif game == 'GridWorld':
            print("Entraînement de l'agent pour GridWorld.")
            env = GridWorld()
        else:
            print("Option non reconnue ou fonctionnalité non implémentée.")
            return

        state_size = env.state_description().shape[0]
        action_size = env.action_mask().shape[0]
        agent, algo = choose_algorithm(state_size, action_size, env)
        EPISODES = int(input("Entrez le nombre d'épisodes pour l'entraînement (par ex. 1000) : "))

        if not os.path.exists(f'models/saved_models'):
            os.makedirs(f'models/saved_models')

        for e in range(EPISODES):
            state = env.reset()
            if state is None:
                raise ValueError("L'état initial de l'environnement est None. Vérifiez la méthode reset de l'environnement.")
            done = False
            total_reward = 0

            while not done:
                if algo == 'PPO':
                    # `choose_action` de PPO retourne action, log_prob, et value
                    action, log_prob, value = agent.choose_action(state)
                    agent.store_transition(state, action, total_reward, done, log_prob, value)
                else:
                    # Les autres agents retournent uniquement `action`
                    action = agent.choose_action(state)

                try:
                    next_state, reward, done, _ = env.step(action)
                    if next_state is None:
                        raise ValueError("L'état suivant est None. Vérifiez la méthode step de l'environnement.")
                except ValueError as e:
                    print(f"Action invalide ou erreur d'état: {e}")
                    done = True
                    reward = -1
                    if algo == 'PPO':
                        agent.store_transition(state, action, reward, done, log_prob, value)
                    else:
                        agent.remember(state, action, reward, next_state, done)
                    break

                # Stockage des transitions pour PPO ou REINFORCE
                if algo == 'PPO':
                    agent.store_transition(state, action, reward, done, log_prob, value)
                elif algo == 'REINFORCE':
                    agent.store_transition(state, action, reward)
                else:
                    agent.remember(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

            # Entraînement après chaque épisode
            if algo == 'PPO':
                next_value = agent.value_model.predict(np.expand_dims(state, axis=0), verbose=0)[0][0]
                agent.train(next_value)
            elif algo == 'REINFORCE':
                agent.train()
            else:
                agent.replay()

            # Enregistrement de la récompense cumulée pour l'épisode
            episode_rewards.append(total_reward)

            # Ajout du résultat dans la liste `victory`
            if total_reward > 0:
                victory.append(1)  # Victoire
            elif total_reward < 0:
                victory.append(-1)  # Défaite
            else:
                victory.append(0)  # Match nul

            # Affichage des résultats par épisode
            print(f"Épisode {e + 1}/{EPISODES}, Récompense Totale: {total_reward}, Epsilon: {getattr(agent, 'epsilon', 'N/A')}")
            epsilon = np.append(epsilon, getattr(agent, 'epsilon', 0))
            
        agent.save()

        print("Entraînement terminé.")
        print("Nombre de victoires : ", victory.count(1))
        print("Nombre de défaites : ", victory.count(-1))
        print("Nombre de matchs nuls : ", victory.count(0))

        # Affichage du graphique des récompenses cumulées
        plot_training_rewards(episode_rewards)
        plot_epsilon(epsilon)

    else:
        print("Option non reconnue ou fonctionnalité non implémentée.")


if __name__ == "__main__":
    main()
