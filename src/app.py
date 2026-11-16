"""
FastAPI Model Serving Endpoint
Internship Track: Week 4 - Task 2
Author: Sayantani Adhikary
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")

app = FastAPI(
    title="Iris Species Prediction API",
    description="Containerized Random Forest classifier served via FastAPI",
    version="1.0.0",
)

# Load trained model artifact on startup
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run src/train_model.py first.")

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
target_names = artifact["target_names"]


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, example=5.1)
    sepal_width: float = Field(..., gt=0, example=3.5)
    petal_length: float = Field(..., gt=0, example=1.4)
    petal_width: float = Field(..., gt=0, example=0.2)


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(features: IrisFeatures):
    try:
        X_input = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]])
        pred_idx = int(model.predict(X_input)[0])
        probs = model.predict_proba(X_input)[0]
        return {
            "predicted_class": target_names[pred_idx],
            "class_index": pred_idx,
            "confidence": round(float(np.max(probs)), 4),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
