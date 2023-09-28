import tensorflow as tf
from tensorflow import keras

def build_model(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_dim=input_dim),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model