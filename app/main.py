from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Argo CD MLOps Demo", version="1.0.0")

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "house_price_model.joblib"
)

model = joblib.load(MODEL_PATH)


class HouseFeatures(BaseModel):
    area: float = Field(gt=0)
    bedrooms: int = Field(ge=1)


@app.get("/")
def home():
    return {
        "message": "ML Prediction API is running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(features: HouseFeatures):
    X = np.array([[features.area, features.bedrooms]])
    prediction = model.predict(X)[0]

    return {
        "predicted_price": round(float(prediction), 2),
        "currency": "INR",
        "model_version": "1.0.0",
    }
