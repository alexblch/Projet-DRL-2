import numpy as np
from DQN.luckynumber import LuckyNumberEnv
from DQN.dqn import DQNAgent
# from reinforce import REINFORCEAgent  # Ajoutez votre implémentation REINFORCE ici
import os

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

def main():
    env = LuckyNumberEnv()  # Vous pouvez également ajouter un menu pour l'environnement si besoin
    state_size = env.rows * env.cols
    action_size = env.action_space
    
    agent, algo = choose_algorithm(state_size, action_size)  # Sélectionner l'algorithme
    batch_size = 32
    EPISODES = 10000

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
                print(f"Episode: {e}/{EPISODES}, Score: {total_reward}, Epsilon: {agent.epsilon}")
                env.list_scores.append(time+1)
                print("scores:", env.list_scores)
                break

            if len(agent.memory) > batch_size:
                agent.replay()

        if e % 50 == 0:
            # Sauvegarder dans un dossier spécifique à l'algorithme choisi
            agent.save(f"{algo}/models/model_weights_episode_{e}.h5")
            
    env.graph_scores()

if __name__ == "__main__":
    main()