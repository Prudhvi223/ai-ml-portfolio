# Customer Churn Prediction

## Problem Statement

Customer churn is one of the biggest challenges for telecom companies.
The goal of this project is to predict whether a customer will leave the company based on customer information.

---

## Dataset

- IBM Telco Customer Churn Dataset
- Rows: 7043
- Columns: 21

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- FastAPI
- Streamlit
- Joblib

---

## Workflow

1. Data Cleaning
2. Exploratory Data Analysis
3. Feature Engineering
4. Logistic Regression Model
5. Hyperparameter Tuning using GridSearchCV
6. Save Model using Joblib
7. FastAPI API
8. Streamlit Interface
9. Unit Testing using Pytest

---

## Model

Logistic Regression

Best Parameters

- C = 0.1
- Solver = lbfgs

---

## Performance

Accuracy: XX%

Confusion Matrix

(Add Screenshot)

Classification Report

(Add Screenshot)

---

## Streamlit Demo

(Add Screenshot)

---

## Run Locally

Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-ml-portfolio.git
```

Install packages

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

Run FastAPI

```bash
uvicorn app:app --reload
```

Run Tests

```bash
pytest
```

---

## Folder Structure

```
Day21/
│
├── app.py
├── churn_model.pkl
├── customer_churn.csv
├── README.md
├── requirements.txt
└── test_app.py
```

---

## Author

Prudhvi Raj Kumar