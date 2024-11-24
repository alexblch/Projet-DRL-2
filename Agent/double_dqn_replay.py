# Agent/double_dqn.py

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque

class DoubleDQNAgentWithReplay:
    def __init__(self, env, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=10000,
                 batch_size=32, memory_size=10000, target_update_freq=100):
        self.env = env
        self.state_size = env.state_description().shape[0]
        self.action_size = env.TOTAL_ACTIONS
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.target_update_freq = target_update_freq
        self.step_count = 0

        # Optimiseur
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)

        # Construire les modèles
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(self.state_size,)))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    @tf.function
    def train_step(self, states, targets):
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            loss = tf.reduce_mean(tf.square(predictions - targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([transition[0] for transition in minibatch], dtype=np.float32)
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch], dtype=np.float32)
        next_states = np.array([transition[3] for transition in minibatch], dtype=np.float32)
        dones = np.array([transition[4] for transition in minibatch], dtype=bool)

        states = tf.convert_to_tensor(states)
        next_states = tf.convert_to_tensor(next_states)

        # Prédire les valeurs Q pour les états suivants
        q_next_main = self.model(next_states, training=False)
        q_next_target = self.target_model(next_states, training=False)
        max_actions = tf.argmax(q_next_main, axis=1)
        max_q_next = tf.reduce_sum(q_next_target * tf.one_hot(max_actions, self.action_size), axis=1)

        # Calculer les cibles
        targets = self.model(states, training=False).numpy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * max_q_next[i]

        # Entraîner le modèle
        loss = self.train_step(states, targets)

        # Décroissance de epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        # Mettre à jour le réseau cible périodiquement
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_model()

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            # Action aléatoire parmi les actions valides
            action = np.random.choice(self.env.available_actions_ids())
        else:
            # Prédire les valeurs Q
            state = np.array([state], dtype=np.float32)
            q_values = self.model.predict(state, verbose=0)[0]
            # Masquer les actions invalides
            valid_actions = self.env.available_actions_ids()
            masked_q_values = np.full(self.action_size, -np.inf)
            masked_q_values[valid_actions] = q_values[valid_actions]
            action = np.argmax(masked_q_values)
        return action

    def save(self, path="models/double_dqn_with_replay.h5"):
        self.model.save(path)

    def load(self, path="models/double_dqn_with_replay.h5"):
        self.model = models.load_model(path)
        self.update_target_model()
