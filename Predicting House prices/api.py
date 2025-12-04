import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

app = FastAPI()

# Load artifacts
model = joblib.load('xgb_model.joblib')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

class HouseData(BaseModel):
    data: Dict[str, Any]

@app.post("/predict")
def predict_price(house: HouseData):
    try:
        input_data = house.data
        df = pd.DataFrame([input_data])

        # Align columns (Fill missing bools with 0)
        df = df.reindex(columns=model_columns, fill_value=0)

        # Scaling
        scale_cols = ['Net_Metrekare', 'Brüt_Metrekare', 'Oda_Sayısı', 'Bulunduğu_Kat', 
                      'Binanın_Yaşı', 'Binanın_Kat_Sayısı', 'Banyo_Sayısı']
        
        # Ensure columns exist before scaling
        present_scale_cols = [col for col in scale_cols if col in df.columns]
        if present_scale_cols:
            df[present_scale_cols] = scaler.transform(df[present_scale_cols])

        # Predict (Returns Log Price)
        log_pred = model.predict(df)[0]

        # Reverse Log (Get Real Price)
        real_price = np.expm1(log_pred)

        return {
            "predicted_price": float(real_price),
            "log_price": float(log_pred)
        }

    except Exception as e:
        return {"error": str(e)}
