import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("fraud_model.pkl")

# App title
st.title("Credit Card Fraud Detection System")

st.write("Upload a CSV file to detect fraudulent transactions.")

# Upload CSV file
uploaded_file = st.file_uploader("Choose CSV File", type=["csv"])

# If user uploads file
if uploaded_file is not None:

    # Read CSV
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.write(data.head())

    try:

        # Remove Class column if exists
        if 'Class' in data.columns:
            data = data.drop('Class', axis=1)

        # Predict fraud
        predictions = model.predict(data)

        # Add predictions to dataset
        data['Prediction'] = predictions

        st.subheader("Prediction Results")
        st.write(data.head())

        # Count fraud vs normal
        fraud_count = data['Prediction'].value_counts()

        st.subheader("Fraud vs Normal Transactions")

        # Show chart
        st.bar_chart(fraud_count)

        # Show counts
        st.write(fraud_count)

    except Exception as e:

        st.error(f"Error: {e}")