import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Activation, BatchNormalization, Dropout, Concatenate, Conv2DTranspose
from tensorflow.keras.models import Model, load_model
from PIL import Image
import numpy as np
import os
import glob
import io

# ==============================================================================
# ⚠️ KONSTANTA (JANGAN UBAH JIKA TIDAK YAKIN) ⚠️
# ==============================================================================
IMG_SIZE = 256  
INPUT_CHANNELS = 7 
OUTPUT_CHANNELS = 3
MODEL_PATH = "models/pix2pix_tryon_G.h5" 
SAMPLE_SHOES_DIR = "images/sample_shoes"
SAMPLE_FEET_DIR = "images/sample_feet"


# --- Arsitektur dan Logika Model (Sama seperti sebelumnya) ---

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(Conv2D(filters, size, strides=2, padding='same',
                      kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(BatchNormalization())
    result.add(LeakyReLU(0.2))
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(Conv2DTranspose(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))
    result.add(BatchNormalization())
    if apply_dropout:
        result.add(Dropout(0.5))
    result.add(Activation('relu'))
    return result

def GeneratorUNet(input_shape=(IMG_SIZE, IMG_SIZE, INPUT_CHANNELS), output_channels=OUTPUT_CHANNELS):
    # ... (Isi GeneratorUNet Anda yang panjang) ...
    # Ganti dengan arsitektur GeneratorUNet lengkap Anda (seperti di jawaban sebelumnya)

    inputs = Input(shape=input_shape)

    down_stack = [
        downsample(64, 4, apply_batchnorm=False), downsample(128, 4), downsample(256, 4), downsample(512, 4), 
        downsample(512, 4), downsample(512, 4), downsample(512, 4), downsample(512, 4, apply_batchnorm=False),
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True), upsample(512, 4, apply_dropout=True), upsample(512, 4, apply_dropout=True), 
        upsample(512, 4), upsample(256, 4), upsample(128, 4), upsample(64, 4),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = Conv2DTranspose(output_channels, 4, strides=2, padding='same',
                           kernel_initializer=initializer, activation='tanh')

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    x = up_stack[0](skips[-1]) 
    for up, skip in zip(up_stack[1:], reversed(skips[:-1])):
        x = Concatenate()([x, skip])
        x = up(x)
    x = Concatenate()([x, skips[0]])
    x = last(x)
    return Model(inputs=inputs, outputs=x, name='Generator')

# Definisikan Custom Objects
CUSTOM_OBJECTS = {
    'GeneratorUNet': GeneratorUNet, 
    'downsample': downsample, 
    'upsample': upsample
}

# 🚨 FUNGSI PEMUATAN MODEL (DENGAN PENANGANAN ERROR PATH) 🚨
@st.cache_resource(show_spinner=True) 
def load_generator_model(model_path):
    st.info("⏳ Memuat model besar dari disk. Ini adalah titik kritis...")
    
    # 🚨 PERBAIKAN PATH 🚨 Coba ganti path jika Streamlit gagal menemukan file
    try:
        model = load_model(model_path, custom_objects=CUSTOM_OBJECTS)
        st.success("✅ Model Generator berhasil dimuat!")
        return model
    except Exception as e:
        # Peringatan: Kemungkinan Memori RAM habis atau Arsitektur salah
        st.error(f"❌ FATAL ERROR SAAT MEMUAT MODEL (Crash Spot): {e}")
        st.warning("Pastikan nama file dan path model (models/pix2pix_tryon_G.h5) 100% benar.")
        return None

# Fungsi Logistik
def preprocess(shoe_image, feet_image, target_size=(IMG_SIZE, IMG_SIZE)):
    shoe = np.array(shoe_image.resize(target_size)) / 127.5 - 1.0
    feet = np.array(feet_image.resize(target_size)) / 127.5 - 1.0
    feet_grayscale = np.array(feet_image.resize(target_size).convert('L')) / 127.5 - 1.0
    mask_channel = np.expand_dims(feet_grayscale, axis=-1)
    combined_input = np.concatenate([shoe, feet, mask_channel], axis=-1)
    return np.expand_dims(combined_input, axis=0)

def postprocess(prediction):
    prediction = (prediction[0] * 0.5 + 0.5) * 255.0
    prediction = prediction.astype(np.uint8)
    return Image.fromarray(prediction)

# Ambil Gambar Sepatu Sample
def get_sample_shoes(sample_dir):
    paths = glob.glob(os.path.join(sample_dir, '*'))
    # Filter hanya file gambar
    return [p for p in paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

# ==============================================================================
# 3. UI STREAMLIT DENGAN GALERI PRODUK
# ==============================================================================

st.set_page_config(page_title="Virtual Shoe Try-On", layout="wide")
st.title("👟 Senikersku Virtual Try-On")
st.markdown("Pilih sepatu produk, tentukan gambar kaki, dan lakukan *Try-On*.")

generator_model = load_generator_model(MODEL_PATH)

# --- 1. PILIH SEPATU DARI GALERI ---
st.header("1. Pilih Sepatu Produk")
st.markdown("*(Klik gambar sepatu di bawah untuk memilih)*")

sample_shoe_paths = get_sample_shoes(SAMPLE_SHOES_DIR)
selected_shoe_path = None

if sample_shoe_paths:
    cols = st.columns(len(sample_shoe_paths))
    for i, path in enumerate(sample_shoe_paths):
        # Gunakan nama file sebagai ID
        shoe_id = os.path.basename(path)
        
        with cols[i]:
            if st.button(f"Pilih", key=f"shoe_btn_{i}"):
                st.session_state['selected_shoe'] = path
            
            st.image(path, caption=shoe_id, width=150)
            
    if 'selected_shoe' in st.session_state and st.session_state['selected_shoe']:
        selected_shoe_path = st.session_state['selected_shoe']
        st.success(f"Anda memilih: {os.path.basename(selected_shoe_path)}")
else:
    st.warning(f"File sepatu sampel tidak ditemukan di folder {SAMPLE_SHOES_DIR}. Pastikan Anda memiliki gambar di sana.")
    st.session_state['selected_shoe'] = None

st.markdown("---")

# --- 2. PILIH GAMBAR KAKI (UNGGAH ATAU SAMPLE) ---
col_kaki_upload, col_kaki_pilih, col_hasil = st.columns([1, 1, 1])
feet_image = None
feet_option = st.radio(
    "2. Sumber Gambar Kaki Pengguna:",
    ("Unggah Citra Kaki", "Pilih Gambar Kaki Sampel"),
    horizontal=True
)

with col_kaki_upload:
    if feet_option == "Unggah Citra Kaki":
        feet_file = st.file_uploader("Unggah Gambar Kaki (RGB)", type=["png", "jpg", "jpeg"], key="feet_upload")
        if feet_file:
            feet_image = Image.open(feet_file).convert("RGB")
            st.image(feet_image, caption="Kaki Pengguna", width=200)

with col_kaki_pilih:
    if feet_option == "Pilih Gambar Kaki Sampel":
        sample_feet_paths = get_sample_shoes(SAMPLE_FEET_DIR)
        feet_selection = st.selectbox("Pilih Kaki Sampel:", [os.path.basename(p) for p in sample_feet_paths])
        
        if feet_selection:
            path = os.path.join(SAMPLE_FEET_DIR, feet_selection)
            feet_image = Image.open(path).convert("RGB")
            st.image(feet_image, caption=feet_selection, width=200)

st.markdown("---")

# --- 3. TOMBOL GENERATE ---

if st.button("👞 Lakukan Try-On dan Generate", use_container_width=True):
    if not generator_model:
        st.error("Gagal menjalankan Try-On: Model tidak berhasil dimuat.")
    elif not selected_shoe_path:
        st.warning("Mohon pilih gambar sepatu produk di langkah 1.")
    elif not feet_image:
        st.warning("Mohon unggah atau pilih gambar kaki di langkah 2.")
    else:
        with st.spinner('⏳ Sedang memproses dan melakukan Try-On...'):
            try:
                processed_shoe_image = Image.open(selected_shoe_path).convert("RGB")
                
                input_tensor = preprocess(processed_shoe_image, feet_image)
                
                if input_tensor.shape[-1] != INPUT_CHANNELS:
                     st.error(f"Error: Input tensor memiliki {input_tensor.shape[-1]} channel, tetapi model membutuhkan {INPUT_CHANNELS} channel.")
                else:
                    prediction = generator_model.predict(input_tensor)
                    try_on_image = postprocess(prediction)
                    
                    st.success("✅ Try-On Selesai!")
                    
                    # Tampilkan hasil di kolom tengah/hasil
                    with col_hasil:
                        st.header("3. Hasil Try-On")
                        st.image(try_on_image, caption="Hasil Virtual Try-On", use_column_width=True)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")
                st.info("Periksa dimensi gambar dan arsitektur model di app.py.")