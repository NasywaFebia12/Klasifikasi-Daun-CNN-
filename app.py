import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# === Download model dari Google Drive ===
model_url = 'https://drive.google.com/uc?id=1IyCbwhZracAfofTCwXAApEcGzkK4SNpS'
model_path = 'leaf_cnn_model.h5'

if not os.path.exists(model_path):
    with st.spinner('Mengunduh model...'):
        gdown.download(model_url, model_path, quiet=False)

# === Load Model ===
model = load_model(model_path)
class_labels = ['daun_sehat', 'daun_sakit']  # ganti sesuai dataset kamu

# === UI Streamlit ===
st.title("🌿 Klasifikasi Kesehatan Daun Tanaman")
st.markdown("Upload gambar daun, lalu sistem akan memprediksi apakah daun sehat atau sakit.")

uploaded_file = st.file_uploader("Pilih gambar daun (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Klasifikasikan"):
        pred = model.predict(img_array)
        pred_class = class_labels[np.argmax(pred)]
        st.success(f"🌱 Prediksi: **{pred_class.upper()}**")
