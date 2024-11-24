# Agent/double_dqn_with_p_replay.py

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class SumTree:
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # Nombre maximal de données
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data  # Stocker l'expérience
        self.update(tree_idx, priority)  # Mettre à jour l'arbre

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0  # Écraser les données anciennes si plein

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # Propager le changement
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0

        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1

        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]  # Racine de l'arbre

class Memory:
    # Hyperparamètres pour le PER
    PER_e = 0.01  # Petite valeur pour éviter une probabilité zéro
    PER_a = 0.6   # Valeur alpha
    PER_b = 0.4   # Valeur beta
    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # Erreur maximale

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, experience):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)

    def sample(self, n):
        memory_b = []
        b_idx = np.empty((n,), dtype=np.int32)
        b_ISWeights = np.empty((n, 1), dtype=np.float32)

        priority_segment = self.tree.total_priority / n

        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])

        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            a, b = priority_segment * i, priority_segment * (i + 1)
            v = random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)

            sampling_probabilities = p / self.tree.total_priority
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight

            b_idx[i] = idx
            memory_b.append(data)

        return b_idx, memory_b, b_ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # Éviter les priorités zéro
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class DoubleDQNAgentWithPrioritizedReplay:
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
        self.memory = Memory(memory_size)
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
        experience = (state, action, reward, next_state, done)
        self.memory.store(experience)

    @tf.function
    def compute_loss(self, states, targets, ISWeights, actions):
        predictions = self.model(states, training=True)
        batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
        indices = tf.stack([batch_indices, actions], axis=1)
        pred_q_values = tf.gather_nd(predictions, indices)
        target_q_values = tf.gather_nd(targets, indices)
        td_errors = pred_q_values - target_q_values
        loss = tf.reduce_mean(ISWeights * tf.square(td_errors))
        return loss, tf.abs(td_errors)

    def replay(self):
        if self.memory.tree.data_pointer < self.batch_size:
            return

        tree_idx, minibatch, ISWeights = self.memory.sample(self.batch_size)
        states = np.array([transition[0] for transition in minibatch], dtype=np.float32)
        actions = np.array([transition[1] for transition in minibatch], dtype=np.int32)
        rewards = np.array([transition[2] for transition in minibatch], dtype=np.float32)
        next_states = np.array([transition[3] for transition in minibatch], dtype=np.float32)
        dones = np.array([transition[4] for transition in minibatch], dtype=bool)

        states = tf.convert_to_tensor(states)
        next_states = tf.convert_to_tensor(next_states)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        ISWeights = tf.convert_to_tensor(ISWeights, dtype=tf.float32)

        # Prédire les valeurs Q pour les états actuels et suivants
        q_values = self.model(states, training=False)
        q_next_main = self.model(next_states, training=False)
        q_next_target = self.target_model(next_states, training=False)

        # Sélectionner les actions optimales pour les prochains états avec le modèle principal
        next_actions = tf.argmax(q_next_main, axis=1, output_type=tf.int32)
        batch_indices = tf.range(self.batch_size, dtype=tf.int32)

        # Obtenir les valeurs Q du réseau cible pour les actions sélectionnées
        next_indices = tf.stack([batch_indices, next_actions], axis=1)
        max_q_next = tf.gather_nd(q_next_target, next_indices).numpy()

        # Calculer les cibles
        targets = q_values.numpy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * max_q_next[i]

        # Entraîner le modèle
        with tf.GradientTape() as tape:
            loss, abs_errors = self.compute_loss(states, targets, ISWeights, actions)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Mettre à jour les priorités
        self.memory.batch_update(tree_idx, abs_errors.numpy())

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

    def save(self, path="models/double_dqn_with_p_replay.h5"):
        self.model.save(path)

    def load(self, path="models/double_dqn_with_p_replay.h5"):
        self.model = models.load_model(path)
        self.update_target_model()
