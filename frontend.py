import streamlit as st
import socket
from prometheus_client import Summary, Counter, disable_created_metrics, start_http_server
from frontend_utils import plot_preds, compare_preds, TICKER_PORT_MAP  # Add your multi-stock function

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

available_tickers = list(TICKER_PORT_MAP.keys())

selected_tickers = st.multiselect("Select one or more stock tickers:", options=available_tickers, max_selections=None)

num_days = st.number_input("Enter number of days to predict:", min_value=1, step=1)

# Predict button
if st.button("Predict"):
    if not selected_tickers or not num_days:
        st.warning("Please select at least one stock and number of days.")
    else:
        try:
            with st.spinner("Fetching prediction..."):
                # Metrics logging
                hostname = socket.gethostname()
                ip = socket.gethostbyname(hostname)
                counter.labels(client=ip).inc()

                # Single-stock prediction
                if len(selected_tickers) == 1:
                    fig = api_usage.time()(plot_preds)(selected_tickers[0], num_days)
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.error("Failed to generate prediction plot.")
                else:
                    # Multi-stock prediction
                    fig = api_usage.time()(compare_preds)(selected_tickers, num_days)
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.error("Failed to generate plots for selected stocks.")
        except Exception as e:
            st.error(f"Error: {e}")
