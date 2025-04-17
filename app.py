import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import requests
import pickle
import joblib
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Configure page
st.set_page_config(page_title="Smart Agriculture AI", page_icon="🌾")

# Add custom CSS to style the sidebar
st.markdown("""
<style>
    .sidebar-button {
        width: 100%;
        margin: 5px 0px;
        text-align: left;
        padding: 10px;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    .sidebar-button:hover {
        background-color: #f0f2f6;
    }
    div[data-testid="stSidebar"] {
        padding-top: 2rem;
    }
    .sidebar-title {
        text-align: left;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        padding-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("🌾 Smart Agriculture AI")

# Initialize session state for navigation
if 'menu_choice' not in st.session_state:
    st.session_state.menu_choice = "Prediksi Cuaca"

# Sidebar with custom navigation
with st.sidebar:
    st.markdown('<p class="sidebar-title">Menu Utama</p>', unsafe_allow_html=True)
    
    # Create navigation buttons
    if st.button("Tanya AI (Gemini)", key="btn_gemini", use_container_width=True):
        st.session_state.menu_choice = "Tanya AI (Gemini)"
        st.rerun()

    if st.button("Prediksi Cuaca", key="btn_cuaca", use_container_width=True):
        st.session_state.menu_choice = "Prediksi Cuaca"
        st.rerun()
        
    if st.button("Deteksi Penyakit Tanaman", key="btn_penyakit", use_container_width=True):
        st.session_state.menu_choice = "Deteksi Penyakit Tanaman"
        st.rerun()
        
    if st.button("Deteksi Jenis Tanah", key="btn_tanah", use_container_width=True):
        st.session_state.menu_choice = "Deteksi Jenis Tanah"
        st.rerun()
        
    if st.button("Prediksi Hasil Panen", key="btn_panen", use_container_width=True):
        st.session_state.menu_choice = "Prediksi Hasil Panen"
        st.rerun()



# Display content based on choice
choice = st.session_state.menu_choice

# Now implement each page based on the choice
if choice == "Prediksi Cuaca":
    st.header("🌦️ Prediksi Cuaca Berdasarkan Kota")

    city = st.text_input("Masukkan Nama Kota", "Jakarta")
    api_key = "0d402044f615b840fb0d0e167bb8b23e"  # Ganti dengan API key WeatherStack

    if st.button("Prediksi"):
        url = f"http://api.weatherstack.com/current?access_key={api_key}&query={city}"
        response = requests.get(url).json()

        if "current" not in response:
            st.error("Kota tidak ditemukan atau API key salah!")
        else:
            temp = response["current"]["temperature"]
            humidity = response["current"]["humidity"]
            weather = response["current"]["weather_descriptions"][0]

            st.success(f"Cuaca di {city}: {weather}")
            st.write(f"🌡️ Suhu: {temp}°C")
            st.write(f"💧 Kelembaban: {humidity}%")

elif choice == "Tanya AI (Gemini)":
    st.header("🤖 Tanya AI Menggunakan Gemini")

    st.markdown("Masukkan pertanyaan atau prompt apapun yang berhubungan dengan pertanian:")

    user_prompt = st.text_area("Prompt", placeholder="Contoh: Bagaimana cara merawat tanaman cabai agar hasil panen maksimal?")
    
    # Dropdown untuk memilih timezone
    import pytz
    timezones = [
        "Asia/Jakarta (WIB)", 
        "Asia/Makassar (WITA)", 
        "Asia/Jayapura (WIT)",
        "Asia/Singapore",
        "Asia/Kuala_Lumpur",
        "Asia/Tokyo",
        "Europe/London",
        "America/New_York"
    ]
    selected_timezone = st.selectbox("Pilih Timezone:", timezones, index=0)
    timezone_code = selected_timezone.split(" ")[0]  # Ambil kode timezone saja
    
    if st.button("Tanya Gemini"):
        if user_prompt.strip() == "":
            st.warning("Silakan masukkan prompt terlebih dahulu.")
        else:
            # Menampilkan indikator loading
            with st.spinner("Sedang memproses pertanyaan..."):
                # Dapatkan informasi waktu saat ini untuk ditambahkan ke konteks
                import datetime
                
                try:
                    # Dapatkan waktu saat ini berdasarkan timezone yang dipilih
                    tz = pytz.timezone(timezone_code)
                    current_time = datetime.datetime.now(tz)
                    
                    # Format waktu dengan berbagai format yang mungkin diperlukan
                    waktu_lengkap = current_time.strftime("%A, %d %B %Y, %H:%M:%S %Z")
                    jam = current_time.strftime("%H:%M")
                    tanggal = current_time.strftime("%d %B %Y")
                    hari = current_time.strftime("%A")
                    
                    # Tambahkan informasi waktu ke dalam prompt
                    context_prompt = f"""
                    INFORMASI WAKTU SAAT INI:
                    - Saat ini adalah: {waktu_lengkap}
                    - Jam: {jam}
                    - Tanggal: {tanggal}
                    - Hari: {hari}
                    - Timezone: {selected_timezone}
                    
                    Gunakan informasi waktu di atas dalam memberikan jawaban. Jika pengguna bertanya tentang waktu atau tanggal saat ini, Anda HARUS menggunakan data waktu yang telah disediakan di atas, bukan menyarankan mencari di Google.
                    
                    PERTANYAAN PENGGUNA:
                    {user_prompt}
                    """
                    
                except Exception as e:
                    st.error(f"Error setting timezone: {str(e)}")
                    context_prompt = user_prompt

                # API key Gemini
                api_key = "AIzaSyAqdG2ufJDIOGEPmd0JhEMEc7RbBwloZVU"  # Ganti jika perlu

                # Gunakan endpoint yang benar untuk Gemini API
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"

                headers = {
                    "Content-Type": "application/json"
                }

                data = {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": context_prompt
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.7,
                        "topK": 40,
                        "topP": 0.95,
                        "maxOutputTokens": 2048
                    }
                }

                try:
                    response = requests.post(url, headers=headers, json=data)
                    
                    if response.status_code == 200:
                        hasil = response.json()
                        try:
                            ai_jawaban = hasil["candidates"][0]["content"]["parts"][0]["text"]
                            st.success("Jawaban dari Gemini:")
                            st.markdown(ai_jawaban)
                            
                            # Tampilkan info waktu yang digunakan
                            with st.expander("Informasi Waktu yang Digunakan"):
                                st.info(f"Waktu saat menjawab: {waktu_lengkap}")
                        except Exception as e:
                            st.error(f"Terjadi kesalahan dalam membaca respons dari Gemini: {str(e)}")
                            st.code(hasil)
                    else:
                        st.error(f"Gagal menghubungi API Gemini. Kode status: {response.status_code}")
                        st.code(response.text)
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {str(e)}")


elif choice == "Deteksi Penyakit Tanaman":
    st.header("🌱 Deteksi Penyakit Tanaman")
    try:
        model = load_model("models/plant_disease_cnn.h5")
    except Exception as e:
        st.error("Model tidak ditemukan! Pastikan model berada dalam folder 'models'.")

    uploaded_file = st.file_uploader("Upload Gambar Daun", type=["jpg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 150, 150, 3)

        prediction = model.predict(img_array)
        classes = ["0","Healthy", "Mold"]
        max_index = np.argmax(prediction)

        if max_index < len(classes):
            result = classes[max_index]
            st.success(f"Hasil Prediksi: {result}")
        else:
            st.error("Prediksi tidak valid, periksa model Anda.")

        if result == "Healthy":
            st.info("Tanaman sehat. Tidak ada tindakan yang diperlukan.")
        elif result == "Mold":
            st.warning("Penyebab: Serangan jamur.")
            st.info("Penanganan: Gunakan fungisida alami.")

elif choice == "Deteksi Jenis Tanah":
    st.header("🪵 Deteksi Jenis Tanah & Rekomendasi Pupuk")
    try:
        model = load_model("models/soil_classifier.h5")
    except Exception as e:
        st.error("Model tidak ditemukan! Pastikan model berada dalam folder 'models'.")

    uploaded_file = st.file_uploader("Upload Gambar Tanah", type=["jpg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar Tanah", use_container_width=True)

        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 150, 150, 3)

        prediction = model.predict(img_array)
        classes = ["Tanah Lempung", "Tanah Pasir", "Tanah Gambut"]
        result = classes[np.argmax(prediction)]

        st.success(f"Jenis Tanah: {result}")

        if result == "Tanah Lempung":
            st.info("Rekomendasi Pupuk: Pupuk Organik & Kompos.")

elif choice == "Prediksi Hasil Panen":
    st.header("🌾 Prediksi Hasil Panen (ton/ha)")
    
    try:
        # Load Model
        model = joblib.load("models/yield_prediction_pipeline.pkl")

        # Input
        region = st.selectbox("Region", ["Central", "East", "West", "South"])
        soil = st.selectbox("Soil Type", ["Clay", "Sandy", "Loam"])
        crop = st.selectbox("Crop", ["Rice", "Corn", "Wheat"])
        rainfall = st.number_input("Rainfall (mm)", min_value=0)
        temperature = st.number_input("Temperature (°C)", min_value=0)
        fertilizer = st.number_input("Fertilizer Used (kg/ha)", min_value=0)
        irrigation = st.selectbox("Irrigation Used", ["Yes", "No"])
        weather = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Rainy"])
        days = st.number_input("Days to Harvest", min_value=90)

        # Buat dataframe
        input_data = pd.DataFrame([{
            "Region": region,
            "Soil_Type": soil,
            "Crop": crop,
            "Rainfall_mm": rainfall,
            "Temperature_Celsius": temperature,
            "Fertilizer_Used": fertilizer,
            "Irrigation_Used": irrigation,
            "Weather_Condition": weather,
            "Days_to_Harvest": days
        }])

        if st.button("Prediksi"):
            hasil = model.predict(input_data)[0]
            st.success(f"Perkiraan Hasil Panen: {hasil:.2f} ton/ha")
    except Exception as e:
        st.error("Model tidak ditemukan! Pastikan model berada dalam folder 'models'.")