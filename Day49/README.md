
# Day 49 — Project 3: Ship It

## Project Title

SMS Spam Detection using Machine Learning, FastAPI, and Streamlit.

## Project Overview

Project 3 is an end-to-end machine learning application that classifies SMS messages as Spam or Ham.

The project includes data preparation, model training, evaluation, API development, and a Streamlit user interface.

## Project Workflow

1. Prepared the SMS spam dataset.
2. Converted text into numerical features using TF-IDF.
3. Trained a Logistic Regression classification model.
4. Evaluated the trained model.
5. Saved the model and TF-IDF vectorizer using Joblib.
6. Created a FastAPI backend.
7. Created a Streamlit frontend.
8. Connected the frontend to the backend.
9. Tested predictions using real SMS examples.
10. Documented and pushed the project to GitHub.

## Project Structure

- Day46: Dataset preparation and exploration.
- Day47: Model training, tuning, and evaluation.
- Day48: FastAPI backend and Streamlit UI.
- Day49: Final documentation and project shipping.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- FastAPI
- Uvicorn
- Streamlit
- Requests
- Joblib
- Git and GitHub

## Features

- SMS spam classification.
- FastAPI prediction endpoint.
- Interactive Streamlit interface.
- Spam probability output.
- Saved model and vectorizer.
- Swagger API documentation.

## How to Run

### Start the FastAPI Backend

```bash
python -m uvicorn Day48.app:app --reload
```

### Start the Streamlit Frontend

Open another terminal:

```bash
python -m streamlit run Day48/ui.py
```

### Open the Applications

Streamlit:

```text
http://localhost:8501
```

FastAPI Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Project Result

The trained model was successfully connected to FastAPI and Streamlit.

The application accepts SMS messages and returns a Spam or Ham prediction with a spam probability.

## What I Learned

- How to build an end-to-end machine learning project.
- How to save and load trained models.
- How to serve ML models using FastAPI.
- How to create a frontend using Streamlit.
- How to connect a frontend with a backend API.
- How to document and ship a project using GitHub.

## Milestone

Project 3 completed successfully.

Day 46 → Dataset Preparation

Day 47 → Train and Tune

Day 48 → Serve the Model

Day 49 → Ship the Project
