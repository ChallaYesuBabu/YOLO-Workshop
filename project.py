import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load pre-trained model (Replace 'model.h5' with your trained model path)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Class labels (Update according to your dataset)
class_labels = ["Eczema", "Psoriasis", "Melanoma", "Acne", "Healthy Skin"]

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Dermatology AI - Skin Disease Identifier")

uploaded_file = st.file_uploader("Upload an image of the affected skin", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Identify Disease"):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        st.success(f"Predicted Disease: {predicted_class} ({confidence:.2f}% confidence)")
