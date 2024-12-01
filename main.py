# main.py

import numpy as np
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox

# Import des environnements
from Environnements.luckynumberenv import LuckyNumbersEnv
from Environnements.luckynumberrand import LuckyNumbersGameRandConsole, play_n_games
from Environnements.grid import GridWorld  # Assurez-vous que le fichier s'appelle gridworldenv.py
from Environnements.luckynumbergame import LuckyNumbersGame

# Import des agents
from Agent.dqn import DQNAgent
from Agent.double_dqn_replay import DoubleDQNAgentWithReplay
from Agent.double_dqn_with_p_replay import DoubleDQNAgentWithPrioritizedReplay
from Agent.ppo import A2CAgent as PPOAgent
from Agent.reinforce import REINFORCEAgent
from Agent.reinforce_baseline import REINFORCEWithBaselineAgent
from Agent.reinforcebaselinelearnedcritic import REINFORCEWithBaselineLearnedCriticAgent
from Agent.mcts_with_NN import MCTSWithNN
from Agent.alphazero import AlphaZeroAgent
from Agent.muzero import MuZeroAgent
from Agent.doubledqn import DoubleDQNAgent as DoubleDQNAgentNoReplay
from Agent.mcts import MCTS
from Agent.mcts_random_rollouts import MCTSWithRandomRollouts

def clear_screen():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def choose_game():
    game_choice = []
    action_choice = []

    def on_select():
        game = game_var.get()
        action = action_var.get()
        game_choice.append(game)
        action_choice.append(action)
        game_window.destroy()

    def on_close():
        game_window.destroy()

    game_window = tk.Tk()
    game_window.title("Choix du Jeu et de l'Action")
    game_window.protocol("WM_DELETE_WINDOW", on_close)

    tk.Label(game_window, text="Veuillez choisir un jeu :").pack(pady=10)
    game_var = tk.StringVar(value="LuckyNumber")

    games = [
        ("Lucky Number", "LuckyNumber"),
        ("GridWorld", "GridWorld"),
    ]

    for text, value in games:
        tk.Radiobutton(game_window, text=text, variable=game_var, value=value).pack(anchor=tk.W)

    tk.Label(game_window, text="Veuillez choisir une action :").pack(pady=10)
    action_var = tk.StringVar(value="train")

    actions = [
        ("Jouer en mode classique", "play_classic"),
        ("IA vs IA (aléatoire)", "random_vs_random"),
        ("Entraîner un agent", "train"),
    ]

    for text, value in actions:
        tk.Radiobutton(game_window, text=text, variable=action_var, value=value).pack(anchor=tk.W)

    tk.Button(game_window, text="Valider", command=on_select).pack(pady=10)

    game_window.mainloop()

    return action_choice[0] if action_choice else None, game_choice[0] if game_choice else None

def get_number_of_games():
    n_games = []

    def on_submit():
        try:
            n = int(entry.get())
            if n <= 0:
                raise ValueError
            n_games.append(n)
            num_games_window.destroy()
        except ValueError:
            messagebox.showerror("Entrée invalide", "Veuillez entrer un nombre entier positif.")

    num_games_window = tk.Tk()
    num_games_window.title("Nombre de parties")
    tk.Label(num_games_window, text="Entrez le nombre de parties à jouer :").pack(pady=10)
    entry = tk.Entry(num_games_window)
    entry.pack(pady=5)
    tk.Button(num_games_window, text="Valider", command=on_submit).pack(pady=10)
    num_games_window.mainloop()

    return n_games[0] if n_games else None

def get_number_of_episodes():
    n_episodes = []

    def on_submit():
        try:
            n = int(entry.get())
            if n <= 0:
                raise ValueError
            n_episodes.append(n)
            num_episodes_window.destroy()
        except ValueError:
            messagebox.showerror("Entrée invalide", "Veuillez entrer un nombre entier positif.")

    num_episodes_window = tk.Tk()
    num_episodes_window.title("Nombre d'épisodes")
    tk.Label(num_episodes_window, text="Entrez le nombre d'épisodes d'entraînement :").pack(pady=10)
    entry = tk.Entry(num_episodes_window)
    entry.pack(pady=5)
    tk.Button(num_episodes_window, text="Valider", command=on_submit).pack(pady=10)
    num_episodes_window.mainloop()

    return n_episodes[0] if n_episodes else None

def moving_average(values, window):
    """Calcule la moyenne mobile d'une liste de valeurs."""
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma

# Modification des fonctions de traçage pour inclure le nom du jeu
def plot_training_rewards(episode_rewards, algo, game, window=1):
    plt.figure()
    if window > 1:
        rewards_sma = moving_average(episode_rewards, window)
        plt.plot(range(window - 1, len(episode_rewards)), rewards_sma, label='Moyenne Mobile')
    else:
        plt.plot(episode_rewards, label='Récompense')
    plt.xlabel('Épisode')
    plt.ylabel('Récompense Totale')
    plt.title(f"{game} - Récompenses Totales par Épisode - {algo}")
    plt.legend()
    # Créer le dossier plots/game s'il n'existe pas
    if not os.path.exists(f'plots/{game}'):
        os.makedirs(f'plots/{game}')
    plt.savefig(f"plots/{game}/{algo}_training_rewards.png")
    plt.show()

def plot_epsilon(epsilon_values, algo, game):
    plt.figure()
    plt.plot(epsilon_values)
    plt.xlabel('Épisode')
    plt.ylabel('Valeur de epsilon')
    plt.title(f"{game} - Évolution de epsilon par Épisode - {algo}")
    # Créer le dossier plots/game s'il n'existe pas
    if not os.path.exists(f'plots/{game}'):
        os.makedirs(f'plots/{game}')
    plt.savefig(f"plots/{game}/{algo}_epsilon.png")
    plt.show()

def plot_losses(losses, algo, game):
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Épisode')
    plt.ylabel('Perte')
    plt.title(f"{game} - Perte par Épisode - {algo}")
    # Créer le dossier plots/game s'il n'existe pas
    if not os.path.exists(f'plots/{game}'):
        os.makedirs(f'plots/{game}')
    plt.savefig(f"plots/{game}/{algo}_losses.png")
    plt.show()

def choose_algorithm_gui():
    algo_choice = []

    def on_select():
        choice = algo_var.get()
        algo_choice.append(choice)
        algo_window.destroy()

    def on_close():
        algo_window.destroy()

    algo_window = tk.Tk()
    algo_window.title("Choix de l'Algorithme")
    algo_window.protocol("WM_DELETE_WINDOW", on_close)

    tk.Label(algo_window, text="Veuillez choisir l'algorithme de renforcement :").pack(pady=10)

    algo_var = tk.StringVar(value="1")

    algorithms = [
        ("1 - DQN", "1"),
        ("2 - Double DQN avec Experience Replay", "2"),
        ("3 - Double DQN avec Prioritized Experience Replay", "3"),
        ("4 - Double DQN (sans Experience Replay)", "4"),
        ("5 - REINFORCE", "5"),
        ("6 - REINFORCE with Baseline", "6"),
        ("7 - REINFORCE with Learned Critic", "7"),
        ("8 - PPO", "8"),
        ("9 - Agent aléatoire (Random)", "9"),
        ("10 - MCTS", "10"),
        ("11 - MCTS with Neural Networks", "11"),
        ("12 - MCTS with Random Rollouts", "12"),
        ("13 - AlphaZero", "13"),
        ("14 - MuZero (Simplifié)", "14"),
    ]

    for text, value in algorithms:
        tk.Radiobutton(algo_window, text=text, variable=algo_var, value=value).pack(anchor=tk.W)

    tk.Button(algo_window, text="Valider", command=on_select).pack(pady=10)

    algo_window.mainloop()

    return algo_choice[0] if algo_choice else None

def choose_algorithm(env):
    choice = choose_algorithm_gui()

    if choice == '1':
        return DQNAgent(env), 'DQN'
    elif choice == '2':
        return DoubleDQNAgentWithReplay(env), 'Double DQN avec Experience Replay'
    elif choice == '3':
        return DoubleDQNAgentWithPrioritizedReplay(env), 'Double DQN avec Prioritized Experience Replay'
    elif choice == '4':
        return DoubleDQNAgentNoReplay(env), 'Double DQN (sans Experience Replay)'
    elif choice == '5':
        return REINFORCEAgent(env), 'REINFORCE'
    elif choice == '6':
        return REINFORCEWithBaselineAgent(env), 'REINFORCE with Baseline'
    elif choice == '7':
        return REINFORCEWithBaselineLearnedCriticAgent(env), 'REINFORCE with Learned Critic'
    elif choice == '8':
        return PPOAgent(env), 'PPO'
    elif choice == '9':
        return LuckyNumbersGameRandConsole(), 'Agent aléatoire'
    elif choice == '10':
        return MCTS(env=env, n_iterations=1000), 'MCTS'
    elif choice == '11':
        return MCTSWithNN(env=env, n_iterations=1000), 'MCTS with Neural Networks'
    elif choice == '12':
        return MCTSWithRandomRollouts(env=env, n_simulations=1000), 'MCTS with Random Rollouts'
    elif choice == '13':
        return AlphaZeroAgent(env), 'AlphaZero'
    elif choice == '14':
        return MuZeroAgent(env), 'MuZero (Simplifié)'
    else:
        messagebox.showwarning("Choix invalide", "Choix invalide. Utilisation de DQN par défaut.")
        return DQNAgent(env), 'DQN'

def get_moving_average_window():
    window_size = []

    def on_submit():
        try:
            n = int(entry.get())
            if n <= 0:
                raise ValueError
            window_size.append(n)
            window_size_window.destroy()
        except ValueError:
            messagebox.showerror("Entrée invalide", "Veuillez entrer un nombre entier positif.")

    window_size_window = tk.Tk()
    window_size_window.title("Taille de la Fenêtre de Moyenne Mobile")
    tk.Label(window_size_window, text="Entrez la taille de la fenêtre pour la moyenne mobile :").pack(pady=10)
    entry = tk.Entry(window_size_window)
    entry.pack(pady=5)
    tk.Button(window_size_window, text="Valider", command=on_submit).pack(pady=10)
    window_size_window.mainloop()

    return window_size[0] if window_size else None

def main():
    action, game = choose_game()
    victory = []
    episode_rewards = []
    epsilon = []
    losses = []

    if action is None or game is None:
        print("Aucune action ou jeu valide sélectionné. Fin du programme.")
        return

    if action == 'play_classic' and game == 'LuckyNumber':
        print("Lancement du jeu Lucky Number classique contre un agent aléatoire.")
        root = tk.Tk()
        game_instance = LuckyNumbersGame(root)
        root.mainloop()

    elif action == 'random_vs_random' and game == 'LuckyNumber':
        print("Lancement du mode Lucky Number IA vs IA (Aléatoire)")
        n = get_number_of_games()
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

        agent, algo = choose_algorithm(env)
        if agent is None:
            print("Aucun algorithme sélectionné. Fin du programme.")
            return

        print(f"Entraînement de l'agent avec l'algorithme {algo}.")
        EPISODES = get_number_of_episodes()

        if not os.path.exists('models/saved_models'):
            os.makedirs('models/saved_models')

        for e in range(EPISODES):
            state = env.reset()
            if state is None:
                raise ValueError("L'état initial de l'environnement est None. Vérifiez la méthode reset de l'environnement.")
            done = False
            total_reward = 0

            while not done:
                if algo in ['MCTS', 'MCTS with Neural Networks', 'MCTS with Random Rollouts']:
                    action = agent.choose_action()
                elif algo in ['AlphaZero', 'MuZero (Simplifié)']:
                    action = agent.choose_action()
                elif algo == 'Double DQN (sans Experience Replay)':
                    action = agent.choose_action(state)
                elif algo in ['A2C', 'PPO']:
                    action, log_prob, value = agent.choose_action(state)
                else:
                    action = agent.choose_action(state)

                try:
                    next_state, reward, done, _ = env.step(action)
                    if next_state is None:
                        raise ValueError("L'état suivant est None. Vérifiez la méthode step de l'environnement.")
                except ValueError as ex:
                    print(f"Action invalide ou erreur d'état: {ex}")
                    done = True
                    reward = 0.0
                    next_state = state
                    if algo not in ['MCTS', 'MCTS with Neural Networks', 'MCTS with Random Rollouts', 'AlphaZero', 'MuZero (Simplifié)'] and hasattr(agent, 'remember') and hasattr(agent, 'replay'):
                        agent.remember(state, action, reward, next_state, done)
                        agent.replay()
                    break

                if algo in ['DQN', 'Double DQN avec Experience Replay', 'Double DQN avec Prioritized Experience Replay']:
                    if hasattr(agent, 'remember') and hasattr(agent, 'replay'):
                        agent.remember(state, action, reward, next_state, done)
                        agent.replay()
                elif algo == 'Double DQN (sans Experience Replay)':
                    if hasattr(agent, 'learn'):
                        agent.learn(state, action, reward, next_state, done)
                elif algo in ['REINFORCE with Baseline', 'REINFORCE with Learned Critic']:
                    agent.store_transition(state, action, reward)
                elif algo in ['A2C', 'PPO']:
                    agent.store_transition(state, action, reward, done, log_prob, value)
                    if done or len(agent.states) >= agent.n_steps:
                        loss = agent.train(next_state, done)
                        losses.append(loss)
                elif hasattr(agent, 'remember') and hasattr(agent, 'replay'):
                    agent.remember(state, action, reward, next_state, done)
                    agent.replay()

                state = next_state
                total_reward += reward

                if env.is_game_over():
                    done = True
                    if reward == 0.0:
                        final_reward = env.score()
                        total_reward += final_reward
                        reward = final_reward
                    print(env)
                    break

            if algo in ['REINFORCE with Baseline', 'REINFORCE with Learned Critic']:
                loss = agent.train()
                losses.append(loss)
            elif algo in ['AlphaZero']:
                winner = env.score()
                agent.train(winner)
            elif algo == 'MuZero (Simplifié)':
                winner = env.score()
                agent.train(winner)
            elif algo in ['A2C', 'PPO']:
                if len(agent.states) > 0:
                    loss = agent.train(next_state, done)
                    losses.append(loss)

            if hasattr(agent, 'save'):
                agent.save()

            episode_rewards.append(total_reward)
            if hasattr(agent, 'epsilon'):
                epsilon.append(agent.epsilon)

            if total_reward > 0:
                victory.append(1)
            elif total_reward < 0:
                victory.append(-1)
            else:
                victory.append(0)

            print(f"Épisode {e + 1}/{EPISODES}, Récompense Totale: {total_reward}")

        print("Entraînement terminé.")
        print("Nombre de victoires : ", victory.count(1))
        print("Nombre de défaites : ", victory.count(-1))
        print("Nombre de matchs nuls ou d'interruptions de partie : ", victory.count(0))

        # Demander la taille de la fenêtre pour la moyenne mobile des récompenses
        window_size = get_moving_average_window()
        if window_size is None:
            window_size = 1000  # Par défaut, moyenne sur 1000 épisodes

        # Tracer la courbe des récompenses avec la moyenne mobile
        plot_training_rewards(episode_rewards, algo, game, window=window_size)
        if len(epsilon) > 0:
            plot_epsilon(epsilon, algo, game)
        if len(losses) > 0:
            plot_losses(losses, algo, game)

    else:
        print("Option non reconnue ou fonctionnalité non implémentée.")

if __name__ == "__main__":
    main()
