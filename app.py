import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import requests
import pickle
import joblib
import json
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
    st.session_state.menu_choice = "Ask AI (Gemini)"

# Sidebar with custom navigation
with st.sidebar:
    st.markdown('<p class="sidebar-title">Main Menu</p>', unsafe_allow_html=True)
    # Create navigation buttons
    if st.button("Ask AI (Gemini)", key="btn_gemini", use_container_width=True):
        st.session_state.menu_choice = "Ask AI (Gemini)"
        st.rerun()
    if st.button("Weather Prediction", key="btn_cuaca", use_container_width=True):
        st.session_state.menu_choice = "Weather Prediction"
        st.rerun()
    if st.button("Plant Disease Detection", key="btn_penyakit", use_container_width=True):
        st.session_state.menu_choice = "Plant Disease Detection"
        st.rerun()
    if st.button("Soil Type Detection", key="btn_tanah", use_container_width=True):
        st.session_state.menu_choice = "Soil Type Detection"
        st.rerun()
    if st.button("Harvest Prediction", key="btn_panen", use_container_width=True):
        st.session_state.menu_choice = "Harvest Prediction"
        st.rerun()

# Display content based on choice
choice = st.session_state.menu_choice

if choice == "Weather Prediction":
    st.header("🌾 Weather Prediction for Agriculture")
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        city = st.text_input("Enter City/Location", "Jakarta")
        crop_type = st.selectbox("Select Crop Type", [
            "Padi (Rice)", "Jagung (Corn)", "Kedelai (Soybean)", 
            "Cabai (Chili)", "Tomat (Tomato)", "Kentang (Potato)",
            "Bawang Merah (Shallot)", "Kangkung (Water Spinach)", 
            "Bayam (Spinach)", "Selada (Lettuce)", "Other"
        ])
        
    with col2:
        growth_stage = st.selectbox("Growth Stage", [
            "Persiapan Lahan (Land Preparation)",
            "Penanaman (Planting)", 
            "Pertumbuhan Awal (Early Growth)",
            "Pertumbuhan Vegetatif (Vegetative Growth)",
            "Pembungaan (Flowering)",
            "Pembuahan (Fruiting)",
            "Panen (Harvesting)"
        ])
        
        prediction_days = st.selectbox("Prediction Period", [
            "3 days", "7 days", "14 days", "30 days"
        ])
    
    api_key = "AIzaSyAqdG2ufJDIOGEPmd0JhEMEc7RbBwloZVU"  # Ganti dengan API key Gemini Anda
    
    if st.button("🌦️ Get Agricultural Weather Prediction"):
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
            
            # Prompt yang disesuaikan untuk pertanian
            prompt = f"""Sebagai ahli agroklimatologi, berikan prediksi cuaca untuk pertanian dalam format JSON dengan struktur berikut:

            Lokasi: {city}
            Jenis Tanaman: {crop_type}
            Tahap Pertumbuhan: {growth_stage}
            Periode Prediksi: {prediction_days}

            Format JSON yang diinginkan:
            {{
                "location": "{city}",
                "crop": "{crop_type}",
                "growth_stage": "{growth_stage}",
                "prediction_period": "{prediction_days}",
                "weather_forecast": {{
                    "temperature_range": "suhu minimum-maksimum dalam celsius",
                    "humidity": "kelembaban rata-rata dalam persen",
                    "rainfall_probability": "probabilitas hujan dalam persen",
                    "rainfall_amount": "perkiraan curah hujan dalam mm",
                    "wind_speed": "kecepatan angin rata-rata",
                    "sun_exposure": "tingkat paparan sinar matahari"
                }},
                "agricultural_impact": {{
                    "crop_suitability": "tingkat kesesuaian cuaca untuk tanaman (sangat baik/baik/cukup/kurang)",
                    "growth_condition": "kondisi pertumbuhan yang diperkirakan",
                    "potential_risks": ["daftar risiko potensial"],
                    "water_requirement": "kebutuhan air/irigasi"
                }},
                "recommendations": {{
                    "farming_activities": ["aktivitas pertanian yang disarankan"],
                    "preventive_measures": ["tindakan pencegahan yang perlu dilakukan"],
                    "optimal_timing": ["waktu optimal untuk aktivitas tertentu"],
                    "irrigation_schedule": "jadwal irigasi yang disarankan"
                }},
                "pest_disease_alert": {{
                    "risk_level": "tingkat risiko hama/penyakit (rendah/sedang/tinggi)",
                    "potential_issues": ["kemungkinan masalah hama/penyakit"],
                    "prevention_tips": ["tips pencegahan"]
                }}
            }}

            Berikan prediksi yang realistis berdasarkan kondisi iklim Indonesia dan karakteristik tanaman yang dipilih."""
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
            
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                if "candidates" in data and len(data["candidates"]) > 0:
                    gemini_response = data["candidates"][0]["content"]["parts"][0]["text"]
                    
                    try:
                        # Ekstrak JSON dari response
                        json_start = gemini_response.find('{')
                        json_end = gemini_response.rfind('}') + 1
                        json_str = gemini_response[json_start:json_end]
                        
                        weather_data = json.loads(json_str)
                        
                        # Display hasil prediksi
                        st.success(f"🌾 Agricultural Weather Prediction for {weather_data['location']}")
                        
                        # Weather Forecast Section
                        st.subheader("🌤️ Weather Forecast")
                        forecast = weather_data.get('weather_forecast', {})
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🌡️ Temperature", forecast.get('temperature_range', 'N/A'))
                            st.metric("💧 Humidity", forecast.get('humidity', 'N/A'))
                        with col2:
                            st.metric("🌧️ Rain Probability", forecast.get('rainfall_probability', 'N/A'))
                            st.metric("🌊 Rainfall Amount", forecast.get('rainfall_amount', 'N/A'))
                        with col3:
                            st.metric("💨 Wind Speed", forecast.get('wind_speed', 'N/A'))
                            st.metric("☀️ Sun Exposure", forecast.get('sun_exposure', 'N/A'))
                        
                        # Agricultural Impact Section
                        st.subheader("🌱 Agricultural Impact")
                        impact = weather_data.get('agricultural_impact', {})
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            suitability = impact.get('crop_suitability', 'Unknown')
                            if "sangat baik" in suitability.lower() or "excellent" in suitability.lower():
                                st.success(f"✅ Crop Suitability: {suitability}")
                            elif "baik" in suitability.lower() or "good" in suitability.lower():
                                st.info(f"ℹ️ Crop Suitability: {suitability}")
                            else:
                                st.warning(f"⚠️ Crop Suitability: {suitability}")
                            
                            st.write(f"🌿 **Growth Condition:** {impact.get('growth_condition', 'N/A')}")
                            st.write(f"💧 **Water Requirement:** {impact.get('water_requirement', 'N/A')}")
                        
                        with col2:
                            st.write("⚠️ **Potential Risks:**")
                            risks = impact.get('potential_risks', [])
                            if isinstance(risks, list):
                                for risk in risks:
                                    st.write(f"• {risk}")
                            else:
                                st.write(f"• {risks}")
                        
                        # Recommendations Section
                        st.subheader("📋 Farming Recommendations")
                        recs = weather_data.get('recommendations', {})
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("🚜 **Recommended Activities:**")
                            activities = recs.get('farming_activities', [])
                            if isinstance(activities, list):
                                for activity in activities:
                                    st.write(f"• {activity}")
                            else:
                                st.write(f"• {activities}")
                            
                            st.write("🕐 **Optimal Timing:**")
                            timings = recs.get('optimal_timing', [])
                            if isinstance(timings, list):
                                for timing in timings:
                                    st.write(f"• {timing}")
                            else:
                                st.write(f"• {timings}")
                        
                        with col2:
                            st.write("🛡️ **Preventive Measures:**")
                            measures = recs.get('preventive_measures', [])
                            if isinstance(measures, list):
                                for measure in measures:
                                    st.write(f"• {measure}")
                            else:
                                st.write(f"• {measures}")
                            
                            st.write(f"💧 **Irrigation Schedule:** {recs.get('irrigation_schedule', 'N/A')}")
                        
                        # Pest & Disease Alert Section
                        st.subheader("🐛 Pest & Disease Alert")
                        pest_alert = weather_data.get('pest_disease_alert', {})
                        
                        risk_level = pest_alert.get('risk_level', 'Unknown').lower()
                        if "tinggi" in risk_level or "high" in risk_level:
                            st.error(f"🚨 Risk Level: {pest_alert.get('risk_level', 'Unknown')}")
                        elif "sedang" in risk_level or "medium" in risk_level:
                            st.warning(f"⚠️ Risk Level: {pest_alert.get('risk_level', 'Unknown')}")
                        else:
                            st.success(f"✅ Risk Level: {pest_alert.get('risk_level', 'Unknown')}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("🦠 **Potential Issues:**")
                            issues = pest_alert.get('potential_issues', [])
                            if isinstance(issues, list):
                                for issue in issues:
                                    st.write(f"• {issue}")
                            else:
                                st.write(f"• {issues}")
                        
                        with col2:
                            st.write("🛡️ **Prevention Tips:**")
                            tips = pest_alert.get('prevention_tips', [])
                            if isinstance(tips, list):
                                for tip in tips:
                                    st.write(f"• {tip}")
                            else:
                                st.write(f"• {tips}")
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        # Jika gagal parse JSON, tampilkan response langsung dengan format yang lebih baik
                        st.success(f"🌾 Agricultural Weather Analysis for {city}")
                        st.write("**Crop:** " + crop_type)
                        st.write("**Growth Stage:** " + growth_stage)
                        st.write("**Prediction Period:** " + prediction_days)
                        st.write("---")
                        st.write(gemini_response)
                        
                else:
                    st.error("No response from Gemini API")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
    
    # Additional Information Section
    with st.expander("ℹ️ About Agricultural Weather Prediction"):
        st.write("""
        **Fitur Prediksi Cuaca Pertanian:**
        
        🌡️ **Parameter Cuaca:**
        - Suhu udara (minimum & maksimum)
        - Kelembaban relatif
        - Probabilitas dan curah hujan
        - Kecepatan angin
        - Intensitas sinar matahari
        
        🌱 **Analisis Dampak Pertanian:**
        - Kesesuaian cuaca untuk jenis tanaman
        - Kondisi pertumbuhan yang diperkirakan
        - Identifikasi risiko potensial
        - Kebutuhan air dan irigasi
        
        📋 **Rekomendasi Praktis:**
        - Aktivitas pertanian yang optimal
        - Tindakan pencegahan
        - Waktu terbaik untuk berbagai kegiatan
        - Jadwal irigasi yang efisien
        
        🐛 **Alert Hama & Penyakit:**
        - Tingkat risiko berdasarkan kondisi cuaca
        - Prediksi masalah potensial
        - Tips pencegahan dini
        
        **Catatan:** Prediksi ini berdasarkan analisis AI dan sebaiknya dikombinasikan dengan pengamatan lapangan dan konsultasi dengan ahli pertanian lokal.
        """)

elif choice == "Ask AI (Gemini)":
    st.header("🤖 Ask AI Using Gemini")
    st.markdown("Enter any question or prompt related to farming:")
    user_prompt = st.text_area("Prompt", placeholder="Example: How do you care for chili plants to get maximum harvest results?")
    if st.button("Gemini asked"):
        if user_prompt.strip() == "":
            st.warning("Please enter a prompt first.")
        else:
            # Menampilkan indikator loading
            with st.spinner("Currently processing inquiries..."):
                # Periksa apakah user menanyakan tentang waktu atau tanggal
                waktu_keywords = ["hour", "time", "date", "today", "now", "what day", "what month", "what year", "what time"]
                pertanyaan_waktu = any(keyword in user_prompt.lower() for keyword in waktu_keywords)
                # Jika menanyakan tentang waktu, siapkan informasi waktu
                if pertanyaan_waktu:
                    import datetime
                    import pytz
                    # Gunakan timezone default Indonesia (WIB)
                    timezone_code = "Asia/Jakarta"
                    timezone_label = "WIB (Waktu Indonesia Barat)"
                    try:
                        # Dapatkan waktu saat ini berdasarkan timezone
                        tz = pytz.timezone(timezone_code)
                        current_time = datetime.datetime.now(tz)
                        # Format waktu dengan berbagai format yang mungkin diperlukan
                        waktu_lengkap = current_time.strftime("%A, %d %B %Y, %H:%M:%S %Z")
                        jam = current_time.strftime("%H:%M")
                        tanggal = current_time.strftime("%d %B %Y")
                        hari = current_time.strftime("%A")
                        # Terjemahkan nama hari dan bulan ke Bahasa Indonesia
                        hari_indo = {
                            "Monday": "Senin", "Tuesday": "Selasa", "Wednesday": "Rabu",
                            "Thursday": "Kamis", "Friday": "Jumat", "Saturday": "Sabtu", "Sunday": "Minggu"
                        }
                        bulan_indo = {
                            "January": "Januari", "February": "Februari", "March": "Maret", "April": "April",
                            "May": "Mei", "June": "Juni", "July": "Juli", "August": "Agustus",
                            "September": "September", "October": "Oktober", "November": "November", "December": "Desember"
                        }
                        for eng, indo in hari_indo.items():
                            hari = hari.replace(eng, indo)
                        for eng, indo in bulan_indo.items():
                            tanggal = tanggal.replace(eng, indo)
                            waktu_lengkap = waktu_lengkap.replace(eng, indo)
                        # Tambahkan informasi waktu ke dalam prompt hanya jika user menanyakan waktu
                        context_prompt = f"""
                        CURRENT TIME INFORMATION:
                        - Currently it is: {hari}, {tanggal}, jam {jam} {timezone_label}
                        - Time: {jam}
                        - Date: {tanggal}
                        - Day: {hari}
                        - Timezone: {timezone_label}
                       IMPORTANT: Use the time information above when providing your answer. If a user asks for the current time or date, you MUST use the time data provided above, rather than suggesting a Google search.
                       USER QUESTIONS:
                        {user_prompt}
                        """
                    except Exception as e:
                        st.error(f"Error getting time information: {str(e)}")
                        context_prompt = user_prompt
                else:
                    # Jika tidak menanyakan tentang waktu, gunakan prompt user langsung
                    context_prompt = user_prompt
                # API key Gemini
                api_key = "AIzaSyAqdG2ufJDIOGEPmd0JhEMEc7RbBwloZVU"  # Ganti jika perlu
                # Gunakan endpoint yang benar untuk Gemini API
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
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
                            st.success("Answer from Gemini:")
                            st.markdown(ai_jawaban)
                        except Exception as e:
                            st.error(f"There was an error reading the response from Gemini: {str(e)}")
                            st.code(hasil)
                    else:
                        st.error(f"Failed to contact Gemini API. Status code: {response.status_code}")
                        st.code(response.text)
                except Exception as e:
                    st.error(f"There is an error: {str(e)}")

elif choice == "Plant Disease Detection":
    st.header("🌱 Plant Disease Detection")
    try:
        model = load_model("models/plant_disease_cnn.h5")
    except Exception as e:
        st.error("Model not found! Make sure the model is in the 'models' folder.")
    
    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Images", use_container_width=True)
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 150, 150, 3)
        prediction = model.predict(img_array)
        classes = ['Bacterial Spot', 'Healthy', 'Leaf Mold', 'Target Spot']
        max_index = np.argmax(prediction)
        
        if max_index < len(classes):
            result = classes[max_index]
            st.success(f"Prediction Results: {result}")
        else:
            st.error("Invalid prediction, check your model.")
        
        if result != "0":  # Hanya proses jika bukan kelas "0"
            if result == "Healthy":
                st.info("Tanaman sehat. Tidak perlu tindakan apa pun.")
            else:
                # Menggunakan AI Gemini untuk mendapatkan informasi tambahan
                prompt = f"Memberikan informasi tentang penyebab dan metode perawatan untuk tanaman yang terserang {result}."
                api_key = "AIzaSyAqdG2ufJDIOGEPmd0JhEMEc7RbBwloZVU"  # Ganti jika perlu
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
                headers = {
                    "Content-Type": "application/json"
                }
                data = {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": prompt
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
                            st.warning(f"Causes and Treatment for {result}:")
                            st.info(ai_jawaban)
                        except Exception as e:
                            st.error(f"There was an error reading the response from Gemini: {str(e)}")
                            st.code(hasil)
                    else:
                        st.error(f"Failed to contact Gemini API. Status code:{response.status_code}")
                        st.code(response.text)
                except Exception as e:
                    st.error(f"There is an error: {str(e)}")

elif choice == "Soil Type Detection":
    st.header("🪵 Soil Type Detection & Fertilizer Recommendations")
    try:
        model = load_model("models/soil_classifier_cnn.h5")
    except Exception as e:
        st.error("Model not found! Make sure the model is in the 'models' folder.")
    
    uploaded_file = st.file_uploader("Upload Land Image", type=["jpg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Images", use_container_width=True)
        
        # Preprocess gambar
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 150, 150, 3)
        
        # Prediksi menggunakan model
        prediction = model.predict(img_array)
        classes = ["0", "aluvial", "andosol", "chalk", "entisol", "humus", "inceptisol", "laterit", "sand"]
        fertile_soils = ["humus", "aluvial", "andosol", "inceptisol"]  # Tanah yang subur
        max_index = np.argmax(prediction)
        
        if max_index < len(classes):
            result = classes[max_index]
            st.success(f"Prediction Results: {result}")
        else:
            st.error("Invalid prediction, check your model.")
        
        if result != "0":  # Hanya proses jika bukan kelas "0"
            # Cek apakah jenis tanah subur atau tidak
            if result in fertile_soils:
                st.success(f"✅ {result.title()} tanahnya subur dan bagus untuk perkebunan!")
                # Prompt untuk tanah subur
                prompt = f"Memberikan informasi tentang karakteristik {result} tanah. Sertakan atribut kesuburannya, tanaman ideal yang tumbuh baik di dalamnya, dan kiat perawatan umum untuk menjaga kesuburannya. Format jawaban Anda dengan judul dan poin-poin yang jelas."
            else:
                st.warning(f"⚠️ {result.title()} memiliki kesuburan lebih rendah dan memerlukan perawatan untuk pertumbuhan tanaman yang optimal.")
                # Prompt untuk tanah tidak subur
                prompt = f"Memberikan informasi tentang karakteristik {result} tanah, mengapa kesuburannya rendah, dan langkah-langkah terperinci tentang cara meningkatkan kesuburannya. Sertakan rekomendasi pupuk tertentu, amandemen tanah yang diperlukan, dan teknik khusus untuk meningkatkan strukturnya. Format jawaban Anda dengan judul dan poin-poin yang jelas."
            
            api_key = "AIzaSyAqdG2ufJDIOGEPmd0JhEMEc7RbBwloZVU"  # Ganti jika perlu
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
            headers = {
                "Content-Type": "application/json"
            }
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
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
                        if result in fertile_soils:
                            st.info(f"Information for {result.title()} Soil:")
                            st.info(ai_jawaban)
                            
                            # Tambahkan rekomendasi khusus untuk tanah subur
                            st.success("Fertilizer Recommendations:")
                            st.markdown("""
                            For fertile soil like this, you only need to maintain its condition with:
                            - Light organic fertilizer application
                            - Regular organic matter addition
                            - Proper crop rotation
                            - Minimal soil disturbance
                            """)
                        else:
                            st.error(f"Soil Treatment Required for {result.title()}:")
                            st.info(ai_jawaban)
                            
                            # Tambahkan rekomendasi khusus untuk tanah tidak subur
                            st.warning("Critical Treatment Needed:")
                            st.markdown("""
                            This soil type requires significant intervention:
                            - Heavy fertilizer application
                            - Soil pH adjustment
                            - Organic matter incorporation
                            - Possible drainage improvement
                            - Regular soil testing
                            """)
                    except Exception as e:
                        st.error(f"There was an error reading the response from Gemini: {str(e)}")
                        st.code(hasil)
                else:
                    st.error(f"Failed to contact Gemini API. Status code: {response.status_code}")
                    st.code(response.text)
            except Exception as e:
                st.error(f"There is an error: {str(e)}")

elif choice == "Harvest Prediction":
    st.header("🌾 Harvest Prediction (ton/ha)")
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
        if st.button("Prediction"):
            hasil = model.predict(input_data)[0]
            st.success(f"Estimated Harvest Results: {hasil:.2f} ton/ha")
    except Exception as e:
        st.error("Model not found! Make sure the model is in the 'models' folder.")