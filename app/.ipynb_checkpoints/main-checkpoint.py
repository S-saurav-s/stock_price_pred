from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import numpy as np

# -----------------------------
# LOAD MODEL & METADATA
# -----------------------------
MODEL_PATH = "models/v1/xgb_vol_5d.pkl"
META_PATH = "models/v1/metadata.json"

model = joblib.load(MODEL_PATH)

with open(META_PATH, "r") as f:
    metadata = json.load(f)

FEATURE_COLS = metadata["features"]

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI(
    title="Volatility Forecasting API",
    description="5-day volatility forecasting using XGBoost",
    version="1.0"
)

# -----------------------------
# INPUT SCHEMA
# -----------------------------
class VolatilityRequest(BaseModel):
    features: list  # must be in same order as FEATURE_COLS

# -----------------------------
# ENDPOINT
# -----------------------------
@app.post("/predict")
def predict_volatility(request: VolatilityRequest):
    X = np.array(request.features).reshape(1, -1)
    pred = model.predict(X)[0]

    return {
        "predicted_volatility": float(pred),
        "model_version": "v1",
        "target": metadata["target"]
    }
