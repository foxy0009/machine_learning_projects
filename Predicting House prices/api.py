import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# --- 1. Load Artifacts Once (Global) ---
try:
    model = joblib.load('xgb_model.joblib')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')  # List of columns the model expects
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model artifacts. {e}")
    # In production, you might want to exit here if models aren't found

app = FastAPI(title="House Price Predictor API")

# --- 2. Strict Input Schema (The "Contract") ---
class HouseInput(BaseModel):
    # Numerical features
    Net_Metrekare: float = Field(..., gt=10, description="Net area in m2")
    Brüt_Metrekare: float = Field(..., gt=10, description="Gross area in m2")
    Oda_Sayısı: float = Field(..., gt=0, description="Number of rooms")
    Banyo_Sayısı: float = Field(..., ge=0, description="Number of bathrooms")
    Binanın_Yaşı: float = Field(..., ge=0, description="Age of building (0-8 encoded)")
    Binanın_Kat_Sayısı: int = Field(..., gt=0, description="Total floors in building")
    Bulunduğu_Kat: int = Field(..., description="Floor of the flat")

    # Categorical features (User sends strings, we convert)
    Sehir: str = Field(..., example="antalya", description="City name (lowercase)")
    Isitma_Tipi: str = Field(..., example="Kombi Doğalgaz", description="Heating type")

# --- 3. Helper: Preprocessing Logic ---
def _preprocess_input(input_data: HouseInput) -> pd.DataFrame:
    """Convert raw user input into the exact format the model expects."""

    # 1. Create DataFrame from raw input
    data_dict = input_data.dict()

    # normalize case-sensitive categories
    data_dict["Sehir"] = data_dict["Sehir"].lower()
    data_dict["Isitma_Tipi"] = data_dict["Isitma_Tipi"].strip()

    df = pd.DataFrame([data_dict])

    # 2. One-hot encode city/heating according to training columns
    city_cols = [c for c in model_columns if c.startswith("Şehir_")]
    heat_cols = [c for c in model_columns if c.startswith("Isıtma_Tipi_")]

    for col in city_cols + heat_cols:
        df[col] = 0

    target_city_col = f"Şehir_{data_dict['Sehir']}"
    target_heat_col = f"Isıtma_Tipi_{data_dict['Isitma_Tipi']}"

    if target_city_col in city_cols:
        df[target_city_col] = 1
    if target_heat_col in heat_cols:
        df[target_heat_col] = 1

    # 3. Align with Model Columns (creates any other missing columns and orders correctly)
    df = df.reindex(columns=model_columns, fill_value=0)

    # 4. Scaling
    scale_cols = [
        'Net_Metrekare', 'Brüt_Metrekare', 'Oda_Sayısı', 'Bulunduğu_Kat',
        'Binanın_Yaşı', 'Binanın_Kat_Sayısı', 'Banyo_Sayısı'
    ]
    valid_scale_cols = [c for c in scale_cols if c in df.columns]
    if valid_scale_cols:
        df[valid_scale_cols] = scaler.transform(df[valid_scale_cols])

    return df

# --- 4. The Endpoint ---
@app.post("/predict")
def predict(house: HouseInput):
    try:
        # Preprocess
        processed_df = _preprocess_input(house)

        # Predict
        log_pred = model.predict(processed_df)[0]
        real_price = np.expm1(log_pred)

        return {
            "prediction_tl": round(float(real_price), 2),
            "status": "success"
        }

    except Exception as e:
        # Log the error internally here
        print(f"Error during prediction: {e}")
        # Return a proper HTTP 500
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
