from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("churn_model.pkl")


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}


@app.get("/predict")
def predict():
    prediction = model.predict([[0]*model.n_features_in_])

    return {
        "prediction": int(prediction[0])
    }
