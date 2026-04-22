# predict.py
# This is a FastAPI web server. When it starts, it loads the trained model.
# It then waits for HTTP POST requests and returns predictions.
# Think of it like your Laravel API routes, but in Python.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel   # validates incoming JSON (like a request DTO)
import pickle
import pandas as pd
import os

# ── STARTUP: Load the model once when the server starts ───────────────────────
# We don't reload it on every request — that would be slow (392KB file)
# Same pattern as a DB connection pool: open once, reuse forever
app = FastAPI(title="Titanic Survival Predictor", version="1.0.0")

MODEL_PATH = os.getenv("MODEL_PATH", "/model/model.pkl")  # configurable via env var

model = None  # will be loaded on startup

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {MODEL_PATH}")

# ── REQUEST SCHEMA ─────────────────────────────────────────────────────────────
# Pydantic validates that incoming JSON has these exact fields with correct types
# If a field is missing or wrong type, FastAPI auto-returns a 422 error
# Same concept as request validation middleware in Laravel
class PassengerInput(BaseModel):
    pclass: int        # 1, 2, or 3
    sex: int           # 0=female, 1=male
    age: float         # age in years
    sibsp: int         # siblings/spouses aboard
    parch: int         # parents/children aboard
    fare: float        # ticket price
    embarked: int      # 0=Cherbourg, 1=Queenstown, 2=Southampton

# ── RESPONSE SCHEMA ───────────────────────────────────────────────────────────
class PredictionOutput(BaseModel):
    survived: bool
    confidence: float
    message: str

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    # Kubernetes will call this to check if the container is alive
    # Same as the health checks you configure in ECS task definitions
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/")
def root():
    return {"service": "Titanic Survival Predictor", "docs": "/docs"}

@app.post("/predict", response_model=PredictionOutput)
def predict(passenger: PassengerInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build a DataFrame from the request (model.predict expects a DataFrame)
    input_df = pd.DataFrame([{
        'Pclass':   passenger.pclass,
        'Sex':      passenger.sex,
        'Age':      passenger.age,
        'SibSp':    passenger.sibsp,
        'Parch':    passenger.parch,
        'Fare':     passenger.fare,
        'Embarked': passenger.embarked
    }])

    prediction = model.predict(input_df)[0]          # 0 or 1
    probability = model.predict_proba(input_df)[0]   # [prob_died, prob_survived]
    confidence = float(probability[1])               # probability of surviving

    return PredictionOutput(
        survived=bool(prediction),
        confidence=round(confidence, 4),
        message="Survived" if prediction == 1 else "Did not survive"
    )