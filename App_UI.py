import streamlit as st
import requests
import pandas as pd


st.title("📈 Stock Prediction App")

# User inputs
ticker = st.text_input("Enter stock ticker (e.g. AAPL):")
num_days = st.number_input("Enter number of days to predict:", min_value=1, step=1)

# Predict button
if st.button("Predict"):
    if not ticker or not num_days:
        st.warning("Please enter both stock ticker and number of days.")
    else:
        try:
            with st.spinner("Fetching prediction..."):
                # Make the API call with both ticker and number of days
                response = requests.get(
                    f"http://localhost:5000/predict?symbol={ticker}&days={num_days}"
                )
                data = response.json()
                dates = data["dates"]
                prices = data["predicted_prices"]

                # Create and display a DataFrame
                df = pd.DataFrame({"Date": dates, "Predicted Price": prices})
                df["Date"] = pd.to_datetime(df["Date"])

                st.line_chart(df.set_index("Date"))

        except Exception as e:
            st.error(f"Failed to fetch prediction: {e}")
