import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from keras.utils import normalize
import os

# ==============================
# Load trained model
# ==============================
model = tf.keras.models.load_model("binary_brain_tumor.h5")

# ==============================
# App UI
# ==============================
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to check if there is a tumor.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

# ==============================
# Prediction Function
# ==============================
def preprocess_image(img):
    img = Image.open(img).convert("RGB")
    img = img.resize((64, 64))   # نفس input_size اللي دربت عليه
    img_array = np.array(img)
    img_array = normalize(img_array, axis=1)
    img_array = np.expand_dims(img_array, axis=0)  # (1,64,64,3)
    return img_array, img

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)

    # Preprocess
    x, display_img = preprocess_image(uploaded_file)

    # Prediction
    prediction = model.predict(x)[0][0]

    if prediction > 0.5:
        st.error("⚠️ Tumor detected")
    else:
        st.success("✅ No Tumor detected")
