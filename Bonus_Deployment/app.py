import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

def main():
    st.title("MNIST Digit Classifier 🧠")
    st.write("Upload an image of a handwritten digit (0-9) to classify it.")

    # Check if model exists
    model_path = "../Part2_Practical/mnist_model.h5"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please run 'Part2_Practical/task2_mnist_cnn.py' first to generate the model.")
        return

    # Load Model
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model(model_path)
    
    model = load_model()

    # File Uploader
    file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if file is not None:
        image = Image.open(file).convert("L") # Convert to grayscale
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess
        # Resize to 28x28
        image = ImageOps.fit(image, (28, 28), Image.Resampling.LANCZOS)
        
        # Invert colors if necessary (MNIST is white on black, user images might be black on white)
        # Simple heuristic: if mean pixel value > 127, it's likely white background, so invert.
        img_array = np.array(image)
        if img_array.mean() > 127:
            image = ImageOps.invert(image)
            st.write("Inverted image colors to match MNIST format (White digit on Black background).")
            st.image(image, caption="Processed Image", width=150)

        # Normalize and reshape
        img_array = np.array(image).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict
        if st.button("Predict"):
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            
            st.success(f"Prediction: **{predicted_class}**")
            st.info(f"Confidence: {confidence:.2%}")
            
            st.bar_chart(prediction[0])

if __name__ == "__main__":
    main()
