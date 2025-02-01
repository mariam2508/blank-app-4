import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Title of the app
st.title("Histopathology Image Classification")

# Sidebar for navigation and additional options
st.sidebar.title("Options")
st.sidebar.markdown("### Upload an image to classify it as Histopathological or Other.")

# Load the trained model
model_path = 'histopathology_model.h5'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    st.sidebar.success("Model loaded successfully!")
else:
    st.sidebar.error("Model file not found. Please ensure 'histopathology_model.h5' is available.")

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
    confidence = float(prediction[0][0])
    result = "Histopathological" if confidence < 0.5 else "Other"
    return result, confidence

# Initialize session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file

# Display the uploaded image
if st.session_state.uploaded_file is not None:
    st.image(st.session_state.uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            result, confidence = predict(st.session_state.uploaded_file)
            st.session_state.prediction_result = result
            st.session_state.confidence = confidence

# Display prediction result
if st.session_state.prediction_result is not None:
    st.success(f"Prediction: {st.session_state.prediction_result}")
    st.info(f"Confidence: {st.session_state.confidence:.2f}")

# Reset button
if st.button("Reset"):
    st.session_state.uploaded_file = None
    st.session_state.prediction_result = None
    st.session_state.confidence = None
    st.experimental_rerun()  # Refresh the app

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("This app uses a deep learning model to classify histopathology images.")
st.markdown("**How to use:** Upload an image and click on the 'Predict' button to see the result.")
st.markdown("**Note:** The model is trained to classify images as either 'Histopathological' or 'Other'.")

# Custom CSS for better design
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stImage {
        border: 2px solid #4CAF50;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
