import streamlit as st
import tensorflow as tf
# Import semua layer Keras yang digunakan untuk menghindari 'Unknown Layer' saat memuat model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Activation, BatchNormalization, Dropout, Concatenate, Conv2DTranspose
from tensorflow.keras.models import Model, load_model
from PIL import Image
import numpy as np
import os
import glob
import io
import base64 # 🚨 BARU: Digunakan untuk mengatasi masalah path gambar

# ==============================================================================
# ⚠️ 1. KONSTANTA UTAMA ⚠️
# ==============================================================================
IMG_SIZE = 256  
INPUT_CHANNELS = 7 # 3 RGB Sepatu + 3 RGB Kaki + 1 Grayscale Masker = 7
OUTPUT_CHANNELS = 3
MODEL_PATH = "models/pix2pix_tryon_G.h5" 
# PASTIKAN NAMA FOLDER DI REPO ANDA BENAR: 'sampel_shoes' dan 'sampel_feet'
SAMPLE_SHOES_DIR = "images/sampel_shoes"
SAMPLE_FEET_DIR = "images/sampel_feet"

# ==============================================================================
# 2. ARSITEKTUR MODEL (Wajib Ada untuk load_model)
# ==============================================================================

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(Conv2D(filters, size, strides=2, padding='same',
                      kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm: result.add(BatchNormalization())
    result.add(LeakyReLU(0.2))
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(Conv2DTranspose(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))
    result.add(BatchNormalization())
    if apply_dropout: result.add(Dropout(0.5))
    result.add(Activation('relu'))
    return result

def GeneratorUNet(input_shape=(IMG_SIZE, IMG_SIZE, INPUT_CHANNELS), output_channels=OUTPUT_CHANNELS):
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
    for down in down_stack: x = down(x); skips.append(x)
    x = up_stack[0](skips[-1]) 
    for up, skip in zip(up_stack[1:], reversed(skips[:-1])): x = Concatenate()([x, skip]); x = up(x)
    x = Concatenate()([x, skips[0]])
    x = last(x)
    return Model(inputs=inputs, outputs=x, name='Generator')

# Definisikan Custom Objects
CUSTOM_OBJECTS = {
    'GeneratorUNet': GeneratorUNet, 
    'downsample': downsample, 
    'upsample': upsample
}

# ==============================================================================
# 3. FUNGSI LOGISTIK & PEMUATAN FILE
# ==============================================================================

@st.cache_resource(show_spinner=True) 
def load_generator_model(model_path):
    st.info("⏳ Memuat model besar dari disk. Ini adalah titik kritis...")
    try:
        model = load_model(model_path, custom_objects=CUSTOM_OBJECTS)
        st.success("✅ Model Generator berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"❌ FATAL ERROR SAAT MEMUAT MODEL: {e}")
        st.warning("Pastikan path model di GitHub sudah benar (models/pix2pix_tryon_G.h5).")
        return None

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

def get_sample_paths(sample_dir):
    paths = glob.glob(os.path.join(sample_dir, '*'))
    return [p for p in paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

@st.cache_data
def get_image_base64(path):
    """Membaca file sebagai Base64 string untuk ditampilkan melalui HTML."""
    try:
        with open(path, "rb") as f:
            base64_img = base64.b64encode(f.read()).decode('utf-8')
            return base64_img
    except Exception as e:
        # Jika gambar gagal dibuka, kembalikan None
        st.error(f"Gagal memuat gambar dari path: {path}. Pastikan folder 'images/sampel_shoes' sudah di-push.")
        return None

# Muat Model
generator_model = load_generator_model(MODEL_PATH)

# ==============================================================================
# 4. UI STREAMLIT DAN LOGIKA APLIKASI
# ==============================================================================

st.set_page_config(page_title="Virtual Shoe Try-On", layout="wide")

# --- Header Palsu & Styling ---
st.markdown(
    """
    <style>
    /* Mengatasi tema gelap dan font */
    body { color: #fff; background-color: #000; } 
    .senikersku-header { font-size: 36px; font-weight: 900; text-align: center; margin-bottom: 20px;}
    .senikersku-nav { display: flex; justify-content: center; gap: 30px; font-weight: 500; margin-bottom: 30px;}
    .product-name { font-size: 16px; font-weight: 600; text-align: center; margin-top: 10px; height: 40px; color: #fff;}
    .product-price { font-size: 14px; color: #888; text-align: center; margin-bottom: 10px;}
    /* Gaya untuk membuat gambar dapat diklik */
    .clickable-image { cursor: pointer; border: 3px solid transparent; transition: border 0.3s; background-color: #111; padding: 5px;}
    .clickable-image:hover { border: 3px solid #00BFFF;} /* Biru saat hover */
    .selected { border: 3px solid #FF4B4B !important;} /* Merah saat terpilih */
    /* Warna teks Streamlit di tema gelap */
    h1, h2, h3, h4, .stMarkdown { color: #fff !important; }
    </style>
    <div style="text-align: right; font-size: 14px; margin-bottom: 10px; color: #888;">
        <span style="margin-right: 15px;">Payment Confirmation</span> | <span>Store Location</span> | <span>Blog</span>
    </div>
    <div class="senikersku-header">SENIKERSKU VIRTUAL TRY-ON</div>
    <div class="senikersku-nav">
        <span>New Arrival</span><span>Footwear</span><span>Accessories</span>
    </div>
    <hr style="border-top: 1px solid #444;">
    """, unsafe_allow_html=True
)


# Inisialisasi state
if 'selected_shoe_path' not in st.session_state:
    st.session_state['selected_shoe_path'] = None

# --- 1. PILIH SEPATU DARI GALERI (DENGAN KLIK GAMBAR) ---
st.header("1. Pilih Sepatu Produk")
st.markdown("*(Klik gambar sepatu yang Anda inginkan untuk melanjutkan ke Langkah 2)*")

sample_shoe_paths = get_sample_paths(SAMPLE_SHOES_DIR)

if sample_shoe_paths:
    # Menggunakan 4 kolom seperti di gambar terakhir yang Anda unggah
    cols = st.columns(4) 
    
    # Tangani URL Query Parameter untuk Klik Gambar
    query_params = st.query_params
    if 'shoe' in query_params:
        st.session_state['selected_shoe_path'] = query_params['shoe']
        # st.query_params.clear() # Dihapus agar state aman saat Rerunning
        st.experimental_rerun()

    # Loop untuk menampilkan 4 gambar produk
    for i, path in enumerate(sample_shoe_paths[:4]): 
        shoe_name = os.path.basename(path).split('.')[0].upper()
        
        # 🚨 Memuat gambar sebagai Base64 🚨
        base64_img = get_image_base64(path)
        
        with cols[i]:
            if base64_img:
                is_selected = 'selected' if st.session_state['selected_shoe_path'] == path else ''
                
                st.markdown(
                    f"""
                    <a href="?shoe={path}" target="_self">
                        <div class="clickable-image {is_selected}">
                            <img src="data:image/jpeg;base64,{base64_img}" style="width: 100%; height: auto; object-fit: contain;">
                        </div>
                    </a>
                    <div class="product-name">ADIDAS {shoe_name} SHOE</div>
                    <div class="product-price">Rp 2.900.000</div>
                    """, unsafe_allow_html=True
                )
            else:
                st.markdown(f"**Gagal memuat gambar produk.**")
            
selected_shoe_path = st.session_state['selected_shoe_path']

if selected_shoe_path:
    st.success(f"✅ Sepatu dipilih: {os.path.basename(selected_shoe_path)}. Lanjutkan ke Langkah 2.")

st.markdown("<hr style='border-top: 1px solid #444;'>", unsafe_allow_html=True)


# ==============================================================================
# --- 2. PILIH GAMBAR KAKI (HANYA MUNCUL JIKA SEPATU TERPILIH) ---
# ==============================================================================

if selected_shoe_path:
    st.header("2. Pilih Gambar Kaki Pengguna")
    feet_image = None

    # Tampilkan gambar sepatu yang sudah dipilih di sidebar untuk referensi
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Sepatu Dipilih:")
    # Gunakan metode st.image standar untuk sidebar
    try:
        st.sidebar.image(Image.open(selected_shoe_path), caption=os.path.basename(selected_shoe_path), width=150)
    except:
        st.sidebar.warning("Gagal memuat gambar pratinjau.")
    st.sidebar.markdown("---")


    feet_option = st.radio(
        "Pilih Sumber Kaki:",
        ("Unggah Citra Kaki", "Pilih Gambar Kaki Sampel"),
        horizontal=True,
        key="feet_source_radio"
    )

    col_upload, col_select = st.columns(2)

    if feet_option == "Unggah Citra Kaki":
        with col_upload:
            feet_file = st.file_uploader("Unggah Gambar Kaki (RGB)", type=["png", "jpg", "jpeg"], key="feet_upload")
            if feet_file:
                feet_image = Image.open(feet_file).convert("RGB")
                st.image(feet_image, caption="Kaki Pengguna", width=200)

    elif feet_option == "Pilih Gambar Kaki Sampel":
        with col_select:
            sample_feet_paths = get_sample_paths(SAMPLE_FEET_DIR)
            feet_files = {os.path.basename(p): p for p in sample_feet_paths}
            
            feet_selection_name = st.selectbox("Pilih Kaki Sampel:", ["Pilih..."] + list(feet_files.keys()))
            
            if feet_selection_name != "Pilih...":
                path = feet_files[feet_selection_name]
                feet_image = Image.open(path).convert("RGB")
                st.image(feet_image, caption=feet_selection_name, width=200)

    st.markdown("---")

    # --- 3. TOMBOL GENERATE DAN HASIL ---
    
    st.header("3. Hasil Virtual Try-On")
    
    if st.button("👞 Lakukan Try-On dan Generate", key="generate_btn", use_container_width=True):
        if not generator_model:
            st.error("Gagal menjalankan Try-On: Model tidak berhasil dimuat.")
        elif not feet_image:
            st.warning("Mohon unggah atau pilih gambar kaki di Langkah 2.")
        else:
            with st.spinner('⏳ Sedang memproses dan melakukan Virtual Try-On...'):
                try:
                    processed_shoe_image = Image.open(selected_shoe_path).convert("RGB")
                    input_tensor = preprocess(processed_shoe_image, feet_image)
                    
                    if input_tensor.shape[-1] != INPUT_CHANNELS:
                         st.error(f"Error: Input tensor memiliki {input_tensor.shape[-1]} channel, tetapi model membutuhkan {INPUT_CHANNELS} channel.")
                    else:
                        prediction = generator_model.predict(input_tensor)
                        try_on_image = postprocess(prediction)
                        
                        st.success("✅ Try-On Selesai!")
                        st.image(try_on_image, caption="Hasil Virtual Try-On", use_column_width=True)

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {e}")