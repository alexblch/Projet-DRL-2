import tensorflow as tf
from tensorflow.keras import layers

def create_model(input_shape, action_space, layer_sizes=[128, 128], learning_rate=0.001):
    """
    Crée un modèle de réseau neuronal pour DQN classique avec MSE comme fonction de perte.
    
    :param input_shape: La forme de l'état (ex: (16,) pour une grille 4x4).
    :param action_space: Nombre d'actions possibles (taille de l'espace d'actions).
    :param layer_sizes: Liste définissant le nombre de neurones dans chaque couche cachée.
    :param learning_rate: Taux d'apprentissage pour l'optimiseur Adam.
    :return: Modèle Keras compilé.
    """
    model = tf.keras.Sequential()
    
    # Couche d'entrée
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Dense(layer_sizes[0], activation='relu'))
    model.add(layers.Dense(layer_sizes[1], activation='relu'))
    model.add(layers.Dense(layer_sizes[0], activation='relu'))
    # Couche de sortie pour les valeurs Q
    model.add(layers.Dense(action_space, activation='linear'))

    # Utilisation de MSE comme fonction de perte
    loss = tf.keras.losses.Huber()
    
    # Compilation du modèle avec Adam
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss)
    
    return model
