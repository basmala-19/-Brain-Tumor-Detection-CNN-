import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ==============================
# Load trained model
# ==============================
model = tf.keras.models.load_model("binary_brain_tumor.h5")

# ==============================
# App UI
# ==============================
st.title("🧠 Brain Tumor Detection (Binary)")
st.write("Upload an MRI image to check if there is a tumor.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

# ==============================
# Prediction Function
# ==============================
def preprocess_image(img):
    img = Image.open(img).convert("RGB")
    img = img.resize((64, 64))   # نفس input_size اللي دربت عليه
    img_array = np.array(img) / 255.0   # تطبيع نفس اللي اتعمل في التدريب
    img_array = np.expand_dims(img_array, axis=0)  # (1,64,64,3)
    return img_array, img

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)

    # Preprocess
    x, display_img = preprocess_image(uploaded_file)

    # Prediction
    prediction = model.predict(x)[0][0]

    # Show result
    if prediction >= 0.5:
        st.error(f"⚠️ Tumor detected (Confidence: {prediction:.2f})")
    else:
        st.success(f"✅ No Tumor detected (Confidence: {1-prediction:.2f})")

