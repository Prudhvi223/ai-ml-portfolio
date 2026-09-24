
# Day 48 — Project 3: Serve SMS Spam Detection

## Overview

On Day 48, I deployed my SMS Spam Detection machine learning model using FastAPI and Streamlit.

The trained Logistic Regression model and TF-IDF vectorizer from Day 47 were loaded and connected to a FastAPI backend.

A Streamlit frontend was created so users can enter SMS messages and receive predictions.

## Project Architecture

User
↓
Streamlit UI
↓
FastAPI Backend
↓
TF-IDF Vectorizer
↓
Logistic Regression Model
↓
Spam or Ham Prediction

## Features

- FastAPI backend for prediction.
- Streamlit user interface.
- SMS text input.
- Spam and Ham classification.
- Spam probability display.
- Saved model and vectorizer loading.
- Interactive API documentation using Swagger UI.

## API Endpoint

### POST /predict

Example request:

```json
{
  "message": "Congratulations! You have won a free prize!"
}
```

Example response:

```json
{
  "message": "Congratulations! You have won a free prize!",
  "prediction": "Spam",
  "spam_probability": 0.95
}
```

The actual prediction and probability depend on the trained model.

## Technologies Used

- Python
- FastAPI
- Uvicorn
- Streamlit
- Requests
- Scikit-learn
- Joblib
- TF-IDF
- Logistic Regression

## How to Run

### Start FastAPI

From the project root:

```bash
python -m uvicorn Day48.app:app --reload
```

### Start Streamlit

Open another terminal:

```bash
python -m streamlit run Day48/ui.py
```

### Open the Application

Streamlit UI:

```text
http://localhost:8501
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

## What I Learned

- How to serve a machine learning model using FastAPI.
- How to create API endpoints.
- How to validate requests using Pydantic.
- How to connect a frontend to a backend using Requests.
- How to build a simple Streamlit interface.
- How to load saved machine learning models.
- How to test an end-to-end machine learning application.

## Conclusion

Day 48 completed the serving stage of Project 3.

The SMS Spam Detection model is now connected to a FastAPI backend and a Streamlit user interface.
