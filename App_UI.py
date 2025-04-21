import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.title("📈 Stock Prediction App")

ticker = st.text_input("Enter stock ticker (e.g. AAPL):")

if st.button("Predict"):
    if not ticker:
        st.warning("Please enter a stock ticker.")
    else:
        try:
            with st.spinner("Fetching prediction..."):
                response = requests.get(f"http://localhost:5000/predict?symbol={ticker}")
                data = response.json()
                dates = data["dates"]
                prices = data["predicted_prices"]

                df = pd.DataFrame({"Date": dates, "Predicted Price": prices})
                df["Date"] = pd.to_datetime(df["Date"])

                st.line_chart(df.set_index("Date"))

        except Exception as e:
            st.error(f"Failed to fetch prediction: {e}")
