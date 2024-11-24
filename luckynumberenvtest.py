from Environnements.luckynumberenv import LuckyNumbersEnv
import numpy as np
import random


def main():
    env = LuckyNumbersEnv()
    state = env.reset()
    done = False

    while not done:
        available_actions = env.available_actions_ids()
        action = random.choice(available_actions)
        state, reward, done, _ = env.step(action)
        print(env)


if __name__ == "__main__":
    main()

