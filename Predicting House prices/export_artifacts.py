import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# 1. Load Data
print("Loading data...")
df = pd.read_csv('cleaned_data.csv')

# 2. Prepare Data
print("Preparing data...")
X = df.drop('Fiyat', axis=1)
y = df['Fiyat']

# Save model columns
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print(f"Saved model_columns.pkl with {len(model_columns)} columns")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log transform target
y_train = np.log1p(y_train)
# y_test = np.log1p(y_test) # Not strictly needed for training

# Scaling
scale_cols = ['Net_Metrekare', 'Brüt_Metrekare', 'Oda_Sayısı', 'Bulunduğu_Kat', 
              'Binanın_Yaşı', 'Binanın_Kat_Sayısı', 'Banyo_Sayısı']

scaler = StandardScaler()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])

# Save scaler
joblib.dump(scaler, 'scaler.pkl')
print("Saved scaler.pkl")

# 3. Train Model
print("Training XGBoost model...")
# Best XGB Params from notebook: {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 1000, 'subsample': 0.8}
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'xgb_model.joblib')
print("Saved xgb_model.joblib")

print("Done!")
