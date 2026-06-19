from typing import Optional

import pandas as pd
import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

# -----------------------------
# Premium Model (Quotation Data)
# -----------------------------
df_car = pd.read_csv("C:/Users/priya/OneDrive/Desktop/acko/DataSet/claim_Quotation_datas/Quotation/acko_car_quotation.csv")
df_bike = pd.read_csv("C:/Users/priya/OneDrive/Desktop/acko/DataSet/claim_Quotation_datas/Quotation/acko_bike_quotation.csv")

# Add vehicle_type column
df_car["vehicle_type"] = "car"
df_bike["vehicle_type"] = "bike"

df = pd.concat([df_car, df_bike], ignore_index=True)

# Feature engineering
df["vehicle_age"] = 2026 - df["manufacturing_year"]

X = df[["vehicle_type", "vehicle_make", "vehicle_model", "fuel_type", "city",
        "manufacturing_year", "vehicle_age", "idv", "ncb_percent"]]
y = df["annual_premium"]

categorical = ["vehicle_type", "vehicle_make", "vehicle_model", "fuel_type", "city"]
numerical = ["manufacturing_year", "vehicle_age", "idv", "ncb_percent"]

from sklearn.preprocessing import OrdinalEncoder

preprocessor_premium = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
    ("num", StandardScaler(), numerical)
])


premium_model = Pipeline([
    ("preprocessor", preprocessor_premium),
    ("regressor", HistGradientBoostingRegressor(max_depth=5, max_iter=50))
])

premium_model.fit(X, y)

# Save premium model
os.makedirs("models", exist_ok=True)
joblib.dump(premium_model, "models/premium_model.pkl")

