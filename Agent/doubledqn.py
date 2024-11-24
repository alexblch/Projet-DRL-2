import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class DoubleDQNAgent:
    def __init__(self, env, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=1000,
                 target_update_freq=100):
        self.env = env
        self.state_size = env.state_description().shape[0]
        self.action_size = env.TOTAL_ACTIONS
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
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

    @tf.function
    def compute_loss(self, q_values, target_q):
        return tf.reduce_mean(tf.square(q_values - target_q))

    def learn(self, state, action, reward, next_state, done):
        # Préparer les entrées
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)
        action = tf.convert_to_tensor([action], dtype=tf.int32)
        reward = tf.convert_to_tensor([reward], dtype=tf.float32)
        done = tf.convert_to_tensor([done], dtype=tf.bool)

        # Entraînement du modèle dans une portée GradientTape
        with tf.GradientTape() as tape:
            # Prédire les valeurs Q pour l'état actuel
            q_values = self.model(state, training=True)  # Shape: (1, action_size)

            # Prédire les valeurs Q pour le prochain état avec le réseau principal
            q_next_main = self.model(next_state, training=False)  # Shape: (1, action_size)
            # Prédire les valeurs Q pour le prochain état avec le réseau cible
            q_next_target = self.target_model(next_state, training=False)  # Shape: (1, action_size)

            # Sélectionner l'action avec la valeur Q maximale dans le réseau principal
            next_action = tf.argmax(q_next_main[0])  # Shape: ()

            # Obtenir la valeur Q du prochain état pour l'action sélectionnée
            next_q_value = q_next_target[0][next_action]

            # Calculer la cible
            target = reward + (1.0 - tf.cast(done, tf.float32)) * self.gamma * next_q_value  # Shape: (1,)

            # Créer un masque pour sélectionner l'action
            mask = tf.one_hot(action, self.action_size)  # Shape: (1, action_size)

            # Calculer la cible Q
            target_q = q_values * (1 - mask) + mask * target

            # Calculer la perte
            loss = self.compute_loss(q_values, target_q)

        # Calculer les gradients et mettre à jour les poids
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Décroissance de epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        # Mettre à jour le réseau cible périodiquement
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_model()

    def save(self, path="models/double_dqn.h5"):
        self.model.save(path)

    def load(self, path="models/double_dqn.h5"):
        self.model = models.load_model(path)
        self.update_target_model()
