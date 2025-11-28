import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocessing
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# ERROR 1: Forgot to expand dimensions for channel (28, 28) -> (28, 28, 1)
# CNN expects 4D tensor (batch, height, width, channels)

# ERROR 2: Incorrect one-hot encoding depth (should be 10)
y_train = keras.utils.to_categorical(y_train, 5) 
y_test = keras.utils.to_categorical(y_test, 5)

model = keras.Sequential(
    [
        # ERROR 3: Input shape mismatch. MNIST is 28x28, not 32x32
        keras.Input(shape=(32, 32, 1)), 
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        # ERROR 4: Output layer has 1 neuron for multi-class classification
        layers.Dense(1, activation="softmax"), 
    ]
)

# ERROR 5: Using binary_crossentropy for multi-class problem
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=128, epochs=1, validation_split=0.1)
