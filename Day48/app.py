
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel


# --------------------------------------------------
# 1. Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="SMS Spam Detection API",
    description="API for predicting whether an SMS is spam or ham",
    version="1.0.0"
)


# --------------------------------------------------
# 2. Locate saved model files
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "Day47" / "models"

MODEL_PATH = MODEL_DIR / "spam_classifier.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"


# --------------------------------------------------
# 3. Load the trained model and vectorizer
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# --------------------------------------------------
# 4. Define the request format
# --------------------------------------------------

class MessageRequest(BaseModel):
    message: str


# --------------------------------------------------
# 5. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "SMS Spam Detection API is running"
    }


# --------------------------------------------------
# 6. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: MessageRequest):

    message = request.message

    message_tfidf = vectorizer.transform([message])

    prediction = model.predict(message_tfidf)[0]

    probability = model.predict_proba(message_tfidf)[0][1]

    if prediction == 1:
        label = "Spam"
    else:
        label = "Ham"

    return {
        "message": message,
        "prediction": label,
        "spam_probability": round(float(probability), 4)
    }
