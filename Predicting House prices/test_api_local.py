from api import predict_price, HouseData
import pandas as pd
import numpy as np

# Create dummy data based on columns
# We need to provide data that matches the expected input
# The model expects specific columns.
# Let's create a dictionary with some sample values.

sample_data = {
    'Net_Metrekare': 100,
    'Brüt_Metrekare': 120,
    'Oda_Sayısı': 3,
    'Bulunduğu_Kat': 2,
    'Binanın_Yaşı': 5,
    'Binanın_Kat_Sayısı': 10,
    'Banyo_Sayısı': 1,
    'Isıtma_Tipi_Kombi Doğalgaz': True,
    'Tapu_Durumu_Kat Mülkiyeti': True,
    'Şehir_ankara': True
}

house = HouseData(data=sample_data)

print("Testing predict_price...")
try:
    result = predict_price(house)
    print("Result:", result)
    
    if "predicted_price" in result and "log_price" in result:
        print("SUCCESS: API returned valid prediction.")
    else:
        print("FAILURE: API returned unexpected format.")
        
except Exception as e:
    print(f"FAILURE: Exception occurred: {e}")
