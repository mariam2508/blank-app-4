import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Title of the app
st.title("Histopathology Image Classification")

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
    result = "Histopathological" if prediction < 0.5 else "Other"
    return result

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = "home"

# Home Page
if st.session_state.page == "home":
    st.markdown("### Welcome to the Histopathology Image Classification App!")
    st.markdown("""
    This app uses a deep learning model to classify histopathology images into two categories:
    - **Histopathological**: Images that show signs of histopathological conditions.
    - **Other**: Images that do not show signs of histopathological conditions.

    **How to use:**
    1. Click the "Next" button below to proceed to the prediction page.
    2. Upload an image (JPG, JPEG, or PNG) and click the "Predict" button to see the result.
    """)

    if st.button("Next"):
        st.session_state.page = "predict"
        st.experimental_rerun()

# Prediction Page
elif st.session_state.page == "predict":
    st.markdown("### Predict Histopathology Image")
    st.markdown("Upload an image to classify it as Histopathological or Other.")

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

    # Back button to return to the home page
    if st.button("Back to Home"):
        st.session_state.page = "home"
        st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("This app uses a deep learning model to classify histopathology images.")
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
