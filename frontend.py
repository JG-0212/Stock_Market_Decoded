import streamlit as st
import socket
import logging
from prometheus_client import Summary, Counter, disable_created_metrics, start_http_server
from frontend_utils import plot_preds, compare_preds, TICKER_PORT_MAP

# Logger setup
logger = logging.getLogger("frontend")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="application_logs.log",
    filemode="w"
)

@st.cache_resource
def start_metrics_server():
    try:
        start_http_server(18000)
        disable_created_metrics()
        api_usage = Summary('api_runtime', 'API run time monitoring')
        counter = Counter('api_call_counter', 'Number of times that API is called', ['client'])
        logger.info("Prometheus metrics server started on http://localhost:18000")
        return api_usage, counter
    except Exception:
        logger.exception("Failed to start Prometheus metrics server.")

api_usage, counter = start_metrics_server()

st.title("📈 Stock Prediction App")

try:
    available_tickers = list(TICKER_PORT_MAP.keys())
    selected_tickers = st.multiselect("Select one or more stock tickers:", options=available_tickers)
    num_days = st.number_input("Enter number of days to predict:", min_value=1, step=1)
except Exception:
    logger.exception("Failed to load ticker options or user input.")

if st.button("Predict"):
    if not selected_tickers or not num_days:
        st.warning("Please select at least one stock and number of days.")
    else:
        try:
            with st.spinner("Fetching prediction..."):
                hostname = socket.gethostname()
                ip = socket.gethostbyname(hostname)
                counter.labels(client=ip).inc()
                logger.info(f"Prediction requested by client {ip} for: {selected_tickers}, days: {num_days}")

                if len(selected_tickers) == 1:
                    fig = api_usage.time()(plot_preds)(selected_tickers[0], num_days)
                    if fig:
                        st.pyplot(fig)
                        logger.info(f"Successfully plotted prediction for {selected_tickers[0]}")
                    else:
                        st.error("Failed to generate prediction plot.")
                        logger.info(f"Plot for {selected_tickers[0]} was None")
                else:
                    fig = api_usage.time()(compare_preds)(selected_tickers, num_days)
                    if fig:
                        st.pyplot(fig)
                        logger.info(f"Successfully compared predictions for {selected_tickers}")
                    else:
                        st.error("Failed to generate plots for selected stocks.")
                        logger.info("Multi-stock plot was None")

        except Exception:
            logger.exception("Prediction failed.")
            st.error("An error occurred while generating predictions.")
