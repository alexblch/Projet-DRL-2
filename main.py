import numpy as np
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
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

def choose_algorithm_gui():
    algo_choice = []

    def on_select():
        choice = algo_var.get()
        algo_choice.append(choice)
        algo_window.destroy()

    algo_window = tk.Tk()
    algo_window.title("Choix de l'Algorithme")

    tk.Label(algo_window, text="Veuillez choisir l'algorithme de renforcement :").pack(pady=10)

    algo_var = tk.StringVar(value="1")

    algorithms = [
        ("1 - DQN", "1"),
        ("2 - DQN avec Experience Replay", "2"),
        ("3 - DQN avec Prioritized Experience Replay", "3"),
        ("4 - PPO", "4"),
        ("5 - REINFORCE", "5"),
        ("6 - Agent aléatoire (Random)", "6")
    ]

    for text, value in algorithms:
        tk.Radiobutton(algo_window, text=text, variable=algo_var, value=value).pack(anchor=tk.W)

    tk.Button(algo_window, text="Valider", command=on_select).pack(pady=10)

    algo_window.mainloop()

    return algo_choice[0]

def choose_algorithm(env):
    choice = choose_algorithm_gui()

    if choice == '1':
        return DQNAgent(env), 'DQN'
    elif choice == '2':
        return DQNAgentWithReplay(env), 'DQN avec Experience Replay'
    elif choice == '3':
        return DQNAgentWithPrioritizedReplay(env), 'DQN avec Prioritized Experience Replay'
    elif choice == '4':
        return PPOAgent(env), 'PPO'
    elif choice == '5':
        return REINFORCEAgent(env), 'REINFORCE'
    elif choice == '6':
        return LuckyNumbersGameRandConsole(), 'Agent aléatoire'
    else:
        messagebox.showwarning("Choix invalide", "Choix invalide. Utilisation de DQN par défaut.")
        return DQNAgent(env), 'DQN'

def choose_game_gui():
    game_choice = []

    def on_select():
        choice = game_var.get()
        game_choice.append(choice)
        game_window.destroy()

    game_window = tk.Tk()
    game_window.title("Choix du Jeu")

    tk.Label(game_window, text="Veuillez choisir le jeu :").pack(pady=10)

    game_var = tk.StringVar(value="1")

    games = [
        ("1 - Lucky Number classique contre un agent aléatoire", "1"),
        ("2 - Lucky Number - Entraîner un agent", "2"),
        ("3 - GridWorld", "3"),
        ("4 - Lucky Number IA vs IA (Aléatoire)", "4")
    ]

    for text, value in games:
        tk.Radiobutton(game_window, text=text, variable=game_var, value=value).pack(anchor=tk.W)

    tk.Button(game_window, text="Valider", command=on_select).pack(pady=10)

    game_window.mainloop()

    return game_choice[0]

def choose_game():
    choice = choose_game_gui()

    if choice == '1':
        return 'play_classic', 'LuckyNumber'
    elif choice == '2':
        return 'train', 'LuckyNumber'
    elif choice == '3':
        return 'train', 'GridWorld'
    elif choice == '4':
        return 'random_vs_random', 'LuckyNumber'
    else:
        messagebox.showwarning("Choix invalide", "Choix invalide. Retour au menu principal.")
        return None, None

def plot_training_rewards(episode_rewards, name, window=10):
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

def plot_epsilon(epsilon, name):
    plt.figure(figsize=(10, 5))
    plt.plot(epsilon, label='Epsilon')
    plt.xlabel('Épisodes')
    plt.ylabel('Epsilon')
    plt.title('Évolution de Epsilon')
    plt.legend()
    plt.grid()
    plt.show()

def get_number_of_episodes():
    episodes = []

    def on_submit():
        try:
            n = int(entry.get())
            episodes.append(n)
            episodes_window.destroy()
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer un nombre entier.")

    episodes_window = tk.Tk()
    episodes_window.title("Nombre d'Épisodes")

    tk.Label(episodes_window, text="Entrez le nombre d'épisodes pour l'entraînement (par ex. 1000) :").pack(pady=10)
    entry = tk.Entry(episodes_window)
    entry.pack()
    tk.Button(episodes_window, text="Valider", command=on_submit).pack(pady=10)

    episodes_window.mainloop()

    return episodes[0] if episodes else 1000  # Valeur par défaut

def get_number_of_games():
    games = []

    def on_submit():
        try:
            n = int(entry.get())
            games.append(n)
            games_window.destroy()
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer un nombre entier.")

    games_window = tk.Tk()
    games_window.title("Nombre de Parties")

    tk.Label(games_window, text="Entrez le nombre de parties à jouer :").pack(pady=10)
    entry = tk.Entry(games_window)
    entry.pack()
    tk.Button(games_window, text="Valider", command=on_submit).pack(pady=10)

    games_window.mainloop()

    return games[0] if games else 1  # Valeur par défaut

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
        EPISODES = get_number_of_episodes()

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
                    next_state = state  # Garder le même état
                    # Apprentissage immédiat en cas d'erreur
                    if hasattr(agent, 'learn'):
                        agent.learn(state, action, reward, next_state, done)
                    elif hasattr(agent, 'remember'):
                        agent.remember(state, action, reward, next_state, done)
                    break  # Sortir de la boucle while

                # Stockage des transitions ou apprentissage immédiat
                if algo == 'PPO':
                    agent.store_transition(state, action, reward, done, log_prob, value)
                elif algo == 'REINFORCE':
                    agent.store_transition(state, action, reward)
                elif hasattr(agent, 'remember'):
                    # Agents avec expérience replay
                    agent.remember(state, action, reward, next_state, done)
                elif hasattr(agent, 'learn'):
                    # Agents sans expérience replay
                    agent.learn(state, action, reward, next_state, done)
                else:
                    # Autres agents
                    pass

                state = next_state
                total_reward += reward

            # Entraînement après chaque épisode pour certains agents
            if algo == 'PPO':
                next_value = agent.value_model.predict(np.expand_dims(state, axis=0), verbose=0)[0][0]
                agent.train(next_value)
            elif algo == 'REINFORCE':
                agent.train()
            elif hasattr(agent, 'replay'):
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
        print("Nombre de matchs nuls ou d'interruptions de partie : ", victory.count(0))

        # Affichage du graphique des récompenses cumulées
        plot_training_rewards(episode_rewards, algo)
        plot_epsilon(epsilon, algo)

    else:
        print("Option non reconnue ou fonctionnalité non implémentée.")

if __name__ == "__main__":
    main()
