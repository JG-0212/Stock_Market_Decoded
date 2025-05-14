import threading
import mlflow.pyfunc
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import numpy as np
import logging
import traceback
import os

# Setup logger
logger = logging.getLogger("backend")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="application_logs.log",
    filemode="w"
)

# Ticker to port mapping
TICKER_PORT_MAP = {
    "AAPL": 8000,
    "GOOG": 8001,
    "MSFT": 8002
}

# MLflow Tracking URI
MLFLOW_TRACKING_URI = "http://backend:5000"
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info("Mlflow tracker succesfully set")
except Exception as e:
    print(e)
    logger.exception("Error in setting up mlflow tracker")

class InputData(BaseModel):
    inputs: list

def create_app(model):
    app = FastAPI()

    @app.post("/invocations")
    async def invoke(payload: InputData, request: Request):
        try:
            client_ip = request.client.host
            preds = model.predict(np.array(payload.inputs)).tolist()
            logger.info(f"Prediction successful from {client_ip}")
            return {
                "predictions": preds,
                "client_ip": client_ip
            }
        except Exception as e:
            logger.error("Prediction failed", exc_info=True)
            return {"error": str(e)}

    return app

def run_model_server(ticker, port):
    try:
        version_path = f"data/{ticker}/{ticker}_latest.txt"
        if not os.path.exists(version_path):
            raise FileNotFoundError(f"Version file not found at {version_path}")

        with open(version_path) as f:
            version = int(f.read().strip())

        model_name = f"{ticker}PredModel"
        model_uri = f"models:/{model_name}/{version}"

        logger.info(f"Loading model for {ticker} from {model_uri}")
        model = mlflow.keras.load_model(model_uri)

        app = create_app(model)
        logger.info(f"Serving {ticker} model at http://0.0.0.0:{port}/invocations")

        uvicorn.run(app, host='0.0.0.0', port=port)

    except Exception as e:
        logger.error(f"Failed to serve model for {ticker} on port {port}", exc_info=True)

def main():
    logger.info("Starting backend_server")
    threads = []
    for ticker, port in TICKER_PORT_MAP.items():
        thread = threading.Thread(target=run_model_server, args=(ticker, port), daemon=True)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
