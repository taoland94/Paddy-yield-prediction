# main.py
from turtle import st

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import Dict, Any

app = FastAPI(
    title="Paddy Yield Prediction API",
    description="API backend for serving Paddy Yield ML Pipeline predictions.",
    version="1.0.0")

# -------------------------------------------------------------
# 1. CORS SETTINGS (Allows Streamlit to talk to FastAPI)
# -------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any frontend port/domain
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, etc.
    allow_headers=["*"],)

# -------------------------------------------------------------
# 2. Load the trained pipeline model
pipeline = joblib.load("paddy_yield_pipeline.pkl")
# -------------------------------------------------------------
# 3. REQUEST BODY SCHEMA
# -------------------------------------------------------------
class FeaturePayload(BaseModel):
    # Accepts a flexible dictionary of features
    inputs: Dict[str, Any]


# -------------------------------------------------------------
# 4. API ENDPOINTS
# -------------------------------------------------------------
@app.get("/")
def home():
    return {"status": "online", "message": "Paddy Yield Prediction API is running!"}


@app.post("/predict")
def predict_yield(payload: FeaturePayload):
    try:
        # Convert incoming JSON dictionary to a single-row DataFrame
        input_df = pd.DataFrame([payload.inputs])
        
        # Enforce column order if saved in the payload
        if feature_names:
            input_df = input_df[feature_names]

        # Make prediction (Kg/Ha)
        prediction = pipeline.predict(input_df)[0]
        
        return {
            "success": True,
            "predicted_yield_per_ha": float(prediction)}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))