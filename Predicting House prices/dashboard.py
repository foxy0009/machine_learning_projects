import streamlit as st
import requests
import json

# --- VIBE CONFIG ---
st.set_page_config(page_title="House Price AI", layout="centered")
st.title("🏡 House Price Predictor")
st.caption("Powered by XGBoost & FastAPI")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("📍 Location & Details")
    city = st.selectbox("City", ["Istanbul", "Ankara", "Izmir", "Other"])
    heating = st.selectbox("Heating Type", ["Kombi Doğalgaz", "Merkezi", "None"])
    
    st.divider()
    st.write("🔧 Technical Specs")
    age = st.slider("Building Age", 0, 50, 5)
    floor = st.number_input("Floor Number", min_value=0, value=2)
    total_floors = st.number_input("Total Floors in Building", min_value=1, value=5)

# --- MAIN PAGE INPUTS ---
col1, col2 = st.columns(2)
with col1:
    net_sqm = st.number_input("Net Area (m²)", min_value=10, value=100)
    rooms = st.number_input("Rooms", min_value=1.0, value=3.0, step=0.5)

with col2:
    gross_sqm = st.number_input("Gross Area (m²)", min_value=10, value=120)
    bathrooms = st.number_input("Bathrooms", min_value=1, value=1)

# --- LOGIC TO MAP INPUTS TO API FORMAT ---
def build_payload():
    # Start with base numbers
    data = {
        "Net_Metrekare": net_sqm,
        "Brüt_Metrekare": gross_sqm,
        "Oda_Sayısı": rooms,
        "Bulunduğu_Kat": floor,
        "Binanın_Yaşı": age,
        "Binanın_Kat_Sayısı": total_floors,
        "Banyo_Sayısı": bathrooms,
    }
    
    # Handle the One-Hot Encoding (The boolean columns)
    # We just set the specific one to 1, API handles the rest as 0
    if city != "Other":
        data[f"Şehir_{city.lower()}"] = 1
        
    if heating == "Kombi Doğalgaz":
        data["Isıtma_Tipi_Kombi Doğalgaz"] = 1
    elif heating == "Merkezi":
        data["Isıtma_Tipi_Merkezi Doğalgaz"] = 1
        
    return {"data": data}

# --- PREDICTION BUTTON ---
if st.button("✨ Vibe Check Price", type="primary", use_container_width=True):
    payload = build_payload()
    
    try:
        # Talk to your API
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()
        
        if "error" in result:
            st.error(f"API Error: {result['error']}")
        else:
            price = result['predicted_price']
            st.balloons()
            st.success(f"💰 Estimated Price: {price:,.0f} TL")
            
            # Show raw data for debugging
            with st.expander("See API Details"):
                st.json(result)
                
    except Exception as e:
        st.error("⚠️ Is your API running? I can't connect to 127.0.0.1:8000")
        st.error(str(e))