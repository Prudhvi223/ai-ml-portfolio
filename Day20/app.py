import streamlit as st
import joblib
import numpy as np

model = joblib.load("churn_model.pkl")

st.title("Customer Churn Prediction")

st.write("Click the button to predict using dummy data.")

if st.button("Predict"):

    sample = np.zeros((1, model.n_features_in_))

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.error("Customer will Churn")
    else:
        st.success("Customer will Stay")
