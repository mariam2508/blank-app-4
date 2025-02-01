import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os

# Title of the app
st.title("Histopathology Image Classification")

# Load the trained model
model_path = 'histopathology_model.h5'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    st.success("Model loaded successfully!")
else:
    st.error("Model file not found. Please ensure 'histopathology_model.h5' is available.")

# Function to preprocess the image
def preprocess_image(image):
    img = load_img(image, target_size=(150, 150))  # Resize to match model input size
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions
def predict(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    return "Histopathological" if prediction < 0.5 else "Other"

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            result = predict(uploaded_file)
            st.success(f"Prediction: {result}")
