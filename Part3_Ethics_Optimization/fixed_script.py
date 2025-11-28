import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocessing
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# FIX 1: Expand dimensions to include channel info (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# FIX 2: Correct number of classes (MNIST has 10 digits)
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        # FIX 3: Correct input shape (28, 28, 1)
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        # FIX 4: Output layer must have 'num_classes' neurons (10)
        layers.Dense(num_classes, activation="softmax"),
    ]
)

# FIX 5: Use categorical_crossentropy for multi-class classification
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print("Training fixed model...")
model.fit(x_train, y_train, batch_size=128, epochs=1, validation_split=0.1)
print("Fixed script ran successfully!")
