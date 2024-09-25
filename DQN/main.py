import numpy as np
from luckynumber import LuckyNumberEnv
from dqn import DQNAgent
import os

def clear_screen():
    """Efface l'écran (compatible Windows et Unix)."""
    if os.name == 'nt':
        os.system('cls')  # Windows
    else:
        os.system('clear')  # Unix/Linux/Mac

def main():
    env = LuckyNumberEnv()
    state_size = env.rows * env.cols  # 16 états
    action_size = env.action_space  # 16 actions possibles
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    EPISODES = 1000

    # Vérifie si le répertoire 'models' existe, sinon le crée
    if not os.path.exists('models'):
        os.makedirs('models')

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0  # Récompense cumulée pour l'épisode

        for time in range(500):  # Limite le nombre de tours par épisode

            # L'agent choisit une action, on passe l'environnement pour la vérification
            action = agent.act(state, env)
            next_state, reward, done, _ = env.step(action)  # Exécuter l'action dans l'environnement
            total_reward += reward  # Cumul des récompenses

            reward = reward if not done else -10  # Pénalité si l'épisode est terminé
            next_state = np.reshape(next_state, [1, state_size])  # Mise en forme du prochain état

            # L'agent mémorise la transition (state, action, reward, next_state, done)
            agent.remember(state, action, reward, next_state, done)
            state = next_state  # Passe au nouvel état

            if done:
                print(f"Partie terminée après {time+1} tours.")
                agent.update_target_model()  # Mise à jour du modèle cible
                print(f"Episode: {e}/{EPISODES}, Score: {total_reward}, Epsilon: {agent.epsilon}")
                break

            # Entraîner l'agent si suffisamment d'expériences sont disponibles
            if len(agent.memory) > batch_size:
                agent.replay()  # Entraînement sur les expériences stockées

        # Sauvegarder le modèle tous les 50 épisodes
        if e % 50 == 0:
            agent.save(f"models/model_weights_episode_{e}.h5")


if __name__ == "__main__":
    main()
