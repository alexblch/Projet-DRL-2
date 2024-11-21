from Environnements.luckynumberenv import LuckyNumbersEnv
import numpy as np


def main():
    env = LuckyNumbersEnv()
    state = env.reset()
    env.render()
    done = False

    while not done:
        # Exemple d'action aléatoire
        mask = env.action_mask()
        valid_actions = np.where(mask == 1)[0]
        action = np.random.choice(valid_actions)
        print(f"Action choisie: {action}")
        next_state, reward, done, info = env.step(action)
        env.render()
        print(f"Récompense: {reward}\n")

    print("Partie terminée.")
    print(f"Score final: {env.score}")

if __name__ == "__main__":
    main()

