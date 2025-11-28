# Task 2: Deep Learning with TensorFlow/Keras
# Dataset: MNIST Handwritten Digits
# Goal: Build CNN, achieve >95% accuracy, visualize predictions.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def main():
    print("-------------------------------------------------")
    print("Task 2: MNIST Classification with TensorFlow CNN")
    print("-------------------------------------------------")

    # 1. Load and Preprocess Data
    print("\n[1] Loading and Preprocessing MNIST Data...")
    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    print(f"x_train shape: {x_train.shape}")
    print(f"train samples: {x_train.shape[0]}")
    print(f"test samples: {x_test.shape[0]}")

    # Convert class vectors to binary class matrices (one-hot encoding)
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # 2. Build CNN Model
    print("\n[2] Building CNN Architecture...")
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.summary()

    # 3. Compile and Train
    print("\n[3] Training Model...")
    batch_size = 128
    epochs = 5 # 5 epochs is usually enough for >98% on MNIST

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(
        x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1
    )

    # 4. Evaluate
    print("\n[4] Evaluating on Test Set...")
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {score[0]:.4f}")
    print(f"Test accuracy: {score[1]:.4f}")

    if score[1] > 0.95:
        print("SUCCESS: Accuracy > 95% achieved!")
    else:
        print("WARNING: Accuracy is below 95%. Consider training for more epochs.")

    # 5. Visualize Predictions
    print("\n[5] Visualizing Predictions...")
    # Get 5 random indices
    indices = np.random.choice(range(len(x_test)), 5, replace=False)
    
    # Create a figure to save
    plt.figure(figsize=(15, 5))
    
    for i, idx in enumerate(indices):
        image = x_test[idx].reshape(28, 28)
        true_label = np.argmax(y_test[idx])
        
        # Predict
        prediction = model.predict(np.expand_dims(x_test[idx], axis=0), verbose=0)
        predicted_label = np.argmax(prediction)
        
        ax = plt.subplot(1, 5, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"True: {true_label}\nPred: {predicted_label}")
        plt.axis("off")
        
        color = 'green' if true_label == predicted_label else 'red'
        ax.spines['bottom'].set_color(color)
        ax.spines['top'].set_color(color) 
        ax.spines['right'].set_color(color)
        ax.spines['left'].set_color(color)
        ax.tick_params(axis='x', colors=color)
        ax.tick_params(axis='y', colors=color)

    plt.tight_layout()
    plt.savefig("mnist_predictions.png")
    print("Predictions saved to 'mnist_predictions.png'")

    # Save model for the bonus task
    model.save("mnist_model.h5")
    print("Model saved to 'mnist_model.h5'")

if __name__ == "__main__":
    main()
