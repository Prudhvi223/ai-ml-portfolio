
import requests
import streamlit as st


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)


# --------------------------------------------------
# 2. Application title
# --------------------------------------------------

st.title("📩 SMS Spam Detection")
st.write("Enter an SMS message to check whether it is Spam or Ham.")


# --------------------------------------------------
# 3. Text input
# --------------------------------------------------

message = st.text_area(
    "Enter your SMS message:",
    placeholder="Example: Congratulations! You won a free prize!"
)


# --------------------------------------------------
# 4. Prediction button
# --------------------------------------------------

if st.button("Predict"):

    if not message.strip():

        st.warning("Please enter an SMS message.")

    else:

        try:

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"message": message},
                timeout=10
            )

            response.raise_for_status()

            result = response.json()

            prediction = result["prediction"]
            probability = result["spam_probability"]

            if prediction == "Spam":

                st.error(f"Prediction: {prediction}")

            else:

                st.success(f"Prediction: {prediction}")

            st.write(
                f"Spam probability: {probability:.4f}"
            )

        except requests.exceptions.RequestException:

            st.error(
                "Could not connect to the FastAPI server. "
                "Make sure the API is running."
            )
