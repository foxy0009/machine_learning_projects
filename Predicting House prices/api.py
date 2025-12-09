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
    model_columns = joblib.load('model_columns.pkl') # List of columns the model expects
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model artifacts. {e}")
    # In production, you might want to exit here if models aren't found

app = FastAPI(title="House Price Predictor API")

# --- 2. Strict Input Schema (The "Contract") ---
class HouseInput(BaseModel):
    # Numerical features
    Net_Metrekare: float = Field(..., gt=10, description="Net area in m2")
    Oda_Sayısı: float = Field(..., gt=0, description="Number of rooms")
    Banyo_Sayısı: float = Field(..., ge=0, description="Number of bathrooms")
    Binanın_Yaşı: float = Field(..., ge=0, description="Age of building (0-8 encoded)")
    Binanın_Kat_Sayısı: int = Field(..., gt=0, description="Total floors in building")
    Bulunduğu_Kat: int = Field(..., description="Floor of the flat")
    
    # Categorical features (User sends strings, we convert)
    Sehir: str = Field(..., example="antalya", description="City name (lowercase)")
    Isitma_Tipi: str = Field(..., example="Kombi Doğalgaz", description="Heating type")
    # Add other categorical fields you used in training (Takas, Tapu, etc.) if they were important

# --- 3. Helper: Preprocessing Logic ---
def _preprocess_input(input_data: HouseInput) -> pd.DataFrame:
    """
    Converts raw user input into the exact format the model expects.
    """
    # 1. Create DataFrame from raw input
    data_dict = input_data.dict()
    df = pd.DataFrame([data_dict])
    
    # 2. Feature Engineering (Match your training logic!)
    # Example: Calculate Avg_Room_Size if your model uses it
    df['Avg_Room_Size'] = df['Net_Metrekare'] / df['Oda_Sayısı']
    df['Is_Basement'] = (df['Bulunduğu_Kat'] < 0).astype(int)

    # 3. Handle Categorical / One-Hot Encoding manually
    # The user sends "Sehir": "antalya". We need to turn that into "Şehir_antalya": 1
    # We do this by reindexing against the model_columns list.
    
    # First, rename inputs to match training prefixes if necessary
    # (e.g. if you trained on 'Şehir_antalya', map 'Sehir' value to that column)
    target_city_col = f"Şehir_{input_data.Sehir.lower()}"
    target_heat_col = f"Isıtma_Tipi_{input_data.Isitma_Tipi}"
    
    # Create the One-Hot columns manually for this single row
    if target_city_col in model_columns:
        df[target_city_col] = 1
    if target_heat_col in model_columns:
        df[target_heat_col] = 1

    # 4. Align with Model Columns (Crucial Step)
    # This ensures all missing columns (other cities, other heating types) are created and set to 0
    df = df.reindex(columns=model_columns, fill_value=0)

    # 5. Scaling
    # ONLY scale the columns that were scaled in training.
    # Note: Removed Brüt_Metrekare as requested.
    scale_cols = ['Net_Metrekare', 'Oda_Sayısı', 'Bulunduğu_Kat', 
                  'Binanın_Yaşı', 'Binanın_Kat_Sayısı', 'Banyo_Sayısı']
    
    # Filter to ensure we only scale existing columns
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