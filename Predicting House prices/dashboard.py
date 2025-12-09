import streamlit as st
import requests

# --- VIBE CONFIG ---
st.set_page_config(page_title="House Price AI", layout="centered")
st.title("🏡 House Price Predictor")
st.caption("Powered by XGBoost & FastAPI")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("📍 Location & Details")
    city = st.selectbox("City", ["Istanbul", "Ankara", "Izmir", "Antalya", "Other"])
    heating = st.selectbox("Heating Type", ["Kombi Doğalgaz", "Merkezi Doğalgaz", "Other"])

    st.divider()
    st.write("🔧 Technical Specs")
    age = st.slider("Building Age", 0, 50, 5)
    floor = st.number_input("Floor Number", min_value=-3, value=2)
    total_floors = st.number_input("Total Floors in Building", min_value=1, value=5)

# --- MAIN PAGE INPUTS ---
col1, col2 = st.columns(2)
with col1:
    net_sqm = st.number_input("Net Area (m²)", min_value=10, value=100)
    rooms = st.number_input("Rooms", min_value=1.0, value=3.0, step=0.5)

with col2:
    bathrooms = st.number_input("Bathrooms", min_value=0, value=1)


# --- LOGIC TO MAP INPUTS TO API FORMAT ---
def build_payload():
    data = {
        "Net_Metrekare": net_sqm,
        "Oda_Sayısı": rooms,
        "Bulunduğu_Kat": floor,
        "Binanın_Yaşı": age,
        "Binanın_Kat_Sayısı": total_floors,
        "Banyo_Sayısı": bathrooms,
        "Sehir": city.lower() if city != "Other" else "other",
        "Isitma_Tipi": heating if heating != "Other" else "Bilinmiyor",
    }
    return data


# --- PREDICTION BUTTON ---
if st.button("✨ Vibe Check Price", type="primary", use_container_width=True):
    payload = build_payload()

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()

        if response.status_code != 200:
            st.error(f"API Error ({response.status_code}): {result.get('detail', result)}")
        else:
            price = result.get("prediction_tl")
            if price is None:
                st.error("API did not return a prediction. Check server logs.")
            else:
                st.balloons()
                st.success(f"💰 Estimated Price: {price:,.0f} TL")

            # Show raw data for debugging
            with st.expander("See API Details"):
                st.json({"request": payload, "response": result})

    except Exception as e:
        st.error("⚠️ Is your API running? I can't connect to 127.0.0.1:8000")
        st.error(str(e))
