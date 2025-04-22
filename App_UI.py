import streamlit as st
import requests
import pandas as pd
from stock_prediction_deep_learning_inference import *


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
                fig = give_preds_and_plots(ticker, num_days)
                
                if fig:
                    # Display the plot
                    st.pyplot(fig)  # Render the plot in Streamlit
                else:
                    st.error("Failed to generate prediction plot. Please try again.")

        except Exception as e:
            st.error(f"Failed to fetch prediction: {e}")
