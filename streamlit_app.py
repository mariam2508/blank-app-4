import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Title of the app
st.title("Histopathology Image Classification")

# Load the trained model
model_path = 'histopathology_model.h5'
model = None
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    st.sidebar.success("✅ Model loaded successfully!")
else:
    st.sidebar.error("❌ Model file not found. Please ensure 'histopathology_model.h5' is available.")

# Function to preprocess the image
def preprocess_image(image):
    img = load_img(image, target_size=(150, 150))  # Resize to match model input size
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions
def predict(image):
    if model is None:
        return "Error: Model not loaded"
    
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0][0]  # Extract prediction value
    result = "Histopathological" if prediction < 0.5 else "Other"
    return result, prediction  # Return both class and confidence score

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = "home"

# Home Page
if st.session_state.page == "home":
    st.markdown("### Welcome to the Histopathology Image Classification App! 🎯")
    st.markdown("""
    This app classifies histopathology images into two categories:
    - 🏥 **Histopathological**: Images showing signs of histopathological conditions.
    - 📷 **Other**: Images without signs of histopathological conditions.

    **Instructions:**
    1. Click the **"Next"** button to proceed to the prediction page.
    2. Upload an image (JPG, JPEG, or PNG) and click **"Predict"** to see the result.
    """)

    if st.button("Next ➡️"):
        st.session_state.page = "predict"
        st.rerun()

# Prediction Page
elif st.session_state.page == "predict":
    st.markdown("### 🔍 Predict Histopathology Image")
    st.markdown("Upload an image to classify it as **Histopathological** or **Other**.")

    # Upload image
    uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="🖼️ Uploaded Image", use_column_width=True)

        # Make prediction
        if st.button("🚀 Predict"):
            if model is None:
                st.error("⚠️ Model not loaded. Please check the model file.")
            else:
                with st.spinner("⏳ Predicting..."):
                    result, confidence = predict(uploaded_file)
                    st.success(f"✅ Prediction: **{result}**")
                    st.info(f"🔢 Confidence Score: {confidence:.4f}")

    # Back button to return to the home page
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"
        st.rerun()

# Footer
st.markdown("---")
st.markdown("### 📌 About")
st.markdown("This app classifies histopathology images using a deep learning model. 🚀")
st.markdown("**Note:** The model is trained to distinguish images as either **'Histopathological'** or **'Other'**.")

# Custom CSS for better UI
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
        font-weight: bold;
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
