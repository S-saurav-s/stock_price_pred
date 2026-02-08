from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

# ===============================
# APP INITIALIZATION
# ===============================
app = FastAPI(
    title="Market Volatility Forecast API",
    description="Multi-horizon volatility forecasting service",
    version="1.0"
)

# ===============================
# MODEL LOADING
# ===============================
BASE_MODEL_PATH = "models/v1"

model_5d = joblib.load(os.path.join(BASE_MODEL_PATH, "xgb_vol_5d.pkl"))
model_10d = joblib.load(os.path.join(BASE_MODEL_PATH, "xgb_vol_10d.pkl"))

# ===============================
# REQUEST SCHEMA
# ===============================
class FeatureRequest(BaseModel):
    features: list[float]

# ===============================
# HEALTH CHECK
# ===============================
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model_version": "v1",
        "models_loaded": ["vol_5d", "vol_10d"]
    }

# ===============================
# PREDICTION ENDPOINT
# ===============================
@app.post("/predict")
def predict(request: FeatureRequest):
    """
    Expects a list of engineered features in the same order
    used during training.
    """
    X = np.array(request.features).reshape(1, -1)

    vol_5d = model_5d.predict(X)[0]
    vol_10d = model_10d.predict(X)[0]

    return {
        "vol_5d": float(vol_5d),
        "vol_10d": float(vol_10d),
        "model_version": "v1"
    }
