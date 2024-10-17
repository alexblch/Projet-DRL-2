import random
from Environnements.luckynumberenv import LuckyNumbersEnv
from DQN.dqn import DQNAgent

# Importer l'environnement LuckyNumbersEnv
# from Environnements.luckynumberenv import LuckyNumbersEnv  # Assurez-vous que le chemin est correct

# Initialiser l'environnement
env = LuckyNumbersEnv()

# Obtenir les dimensions de l'état et des actions
state_size = env.state_description().shape[0]
action_size = 2 + env.size * env.size  # 2 actions spéciales + actions de placement

agent = DQNAgent(state_size=state_size, action_size=action_size)


num_episodes = 1000  # Nombre d'épisodes pour l'entraînement
for episode in range(num_episodes):
    env.reset()
    state = env.state_description()
    done = env.is_game_over()

    total_reward = 0

    while not done:
        available_actions = env.available_actions_ids()
        action_mask = env.action_mask()

        # L'agent choisit une action
        action = agent.choose_action(state, available_actions, action_mask)

        # L'agent exécute l'action
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

    print(f"Épisode {episode + 1}/{num_episodes}, Récompense Totale: {total_reward}, Epsilon: {agent.epsilon}")

print("Entraînement terminé.")
