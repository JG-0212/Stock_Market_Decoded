import streamlit as st
import socket
from prometheus_client import Summary, Counter, disable_created_metrics, start_http_server
from backend import give_preds_and_plots

@st.cache_resource
def start_metrics_server():
    start_http_server(18000)  # Starts /metrics endpoint on localhost:18000
    disable_created_metrics()
    api_usage = Summary('ab', 'API run time monitoring')
    counter = Counter('cd', 'Number of times that API is called', ['client'])
    print("✅ Prometheus metrics server started on http://localhost:18000")
    
    return api_usage, counter

api_usage, counter = start_metrics_server()
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
                print("Incrementing metrics")
                hostname = socket.gethostname()
                ip = socket.gethostbyname(hostname)
                counter.labels(client=ip).inc()
                fig = api_usage.time()(give_preds_and_plots)(ticker, num_days)
                
                if fig:
                    # Display the plot
                    st.pyplot(fig)  # Render the plot in Streamlit
                else:
                    st.error("Failed to generate prediction plot. Please try again.")

        except Exception as e:
            st.error(f"Failed to fetch prediction: {e}")
