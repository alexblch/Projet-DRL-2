import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers

class DQNAgentWithReplay:
    def __init__(self, env, memory_size=2000, batch_size=64, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.env = env
        self.state_size = env.state_description().shape[0]
        self.action_size = env.TOTAL_ACTIONS  # Utiliser la constante de l'environnement
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.path = "models/dqn_with_replay.h5"

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Créer le modèle avec une architecture simplifiée pour une exécution plus rapide
        self.model = self.create_model(input_shape=(self.state_size,), action_space=self.action_size)

        # Configurer l'utilisation du GPU si disponible
        self.configure_gpu()

    def configure_gpu(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Utilisation du GPU : {physical_devices}")
            except RuntimeError as e:
                print(e)
        else:
            print("Aucun GPU trouvé. Utilisation du CPU.")

    def create_model(self, input_shape, action_space, layer_sizes=[64, 64]):
        inputs = layers.Input(shape=input_shape)
        x = inputs
        for size in layer_sizes:
            x = layers.Dense(size, activation='relu')(x)
        outputs = layers.Dense(action_space, activation='linear')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mean_squared_error')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        action_mask = self.env.action_mask()
        valid_actions = np.where(action_mask == 1)[0]

        if np.random.rand() < self.epsilon:
            # Action aléatoire parmi les actions valides
            return np.random.choice(valid_actions)
        else:
            state = np.expand_dims(state, axis=0)
            q_values = self.model(state, training=False).numpy()[0]
            # Appliquer le masque d'action
            masked_q_values = np.full_like(q_values, -np.inf)
            masked_q_values[valid_actions] = q_values[valid_actions]
            return np.argmax(masked_q_values)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([sample[0] for sample in minibatch], dtype=np.float32)
        actions = np.array([sample[1] for sample in minibatch], dtype=np.int32)
        rewards = np.array([sample[2] for sample in minibatch], dtype=np.float32)
        next_states = np.array([sample[3] for sample in minibatch], dtype=np.float32)
        dones = np.array([sample[4] for sample in minibatch], dtype=np.float32)

        # Prédiction en batch pour les états et les états suivants
        q_values = self.model(states, training=False).numpy()
        q_values_next = self.model(next_states, training=False).numpy()

        # Calcul des Q-valeurs cibles
        for i in range(self.batch_size):
            action_mask_next = self.env.action_mask()
            valid_actions_next = np.where(action_mask_next == 1)[0]
            if dones[i]:
                q_values[i, actions[i]] = rewards[i]
            else:
                max_q_next = np.max(q_values_next[i][valid_actions_next])
                q_values[i, actions[i]] = rewards[i] + self.gamma * max_q_next

        # Entraîner le modèle
        self.model.fit(states, q_values, epochs=1, verbose=0)

        # Mettre à jour epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)  # S'assurer que epsilon ne descend pas en dessous de epsilon_min

    def save(self):
        self.model.save(self.path)

    def load(self):
        self.model = tf.keras.models.load_model(self.path)
