import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
from models.decision_tree import train_tree
from utils.preprocess import load_images

# ------------------ Add Small Logo in Top-Right ------------------
# Load logo image (logo.png must be in the same folder as app.py)
logo = Image.open("logo.JPG")

# Use Streamlit columns to align logo to the top-right corner
col1, col2, col3 = st.columns([6, 1, 1])  # adjust as needed
with col3:
    st.image(logo, width=200)  # Small size (you can make it 40, 60, etc.)

# ------------------ Load and Train the Model ------------------
@st.cache_data
def load_model():
    images, labels = load_images('data/', size=(64, 64))
    model = train_tree(images, labels)
    return model, labels

model, label_list = load_model()

# ------------------ Streamlit UI ------------------
st.title("🌿 Vegetable Plant Disease Detector")
st.write("Upload a leaf image to detect disease using Decision Tree")

# Upload leaf image
uploaded_file = st.file_uploader("📤 Choose a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to image array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_resized = cv2.resize(img, (64, 64))

    # Display uploaded image
    st.image(img_resized, caption='Uploaded Image', use_container_width=True)

    # Flatten the image and make prediction
    input_image = img_resized.reshape(1, -1)
    prediction = model.predict(input_image)

    # Display result
    st.success(f"🌱 Predicted Disease: **{prediction[0]}**")
