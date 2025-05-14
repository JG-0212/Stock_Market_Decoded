import os
import time
import logging
import yfinance as yf

from frontend_utils import give_preds, TICKER_PORT_MAP

# Logger setup
logger = logging.getLogger("evaluator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="evaluator_logs.log",
    filemode="w"
)

GAMMA = 0.99


def eval_today_se(ticker, ticker_data):
    try:
        data = ticker_data.reset_index()[['Close']]
        today_actual = data.values[-1][0]
        _, _, today_predicted, _ = give_preds(ticker, 1)
        return (today_actual - today_predicted[0]) ** 2
    except Exception:
        logger.exception(f"Failed to evaluate squared error for {ticker}")
        return 0.0


if __name__ == '__main__':
    all_data = yf.download(
        list(TICKER_PORT_MAP.keys()), period='2d', interval='1d', group_by='tickers'
    )
    
    for ticker in list(TICKER_PORT_MAP.keys()):
        try:
            score_path = f"evaluation/{ticker}/score.txt"
            if os.path.exists(score_path):
                with open(score_path, "r") as f:
                    cur_score = float(f.read().strip())
            else:
                cur_score = 0.0

            updated_score = cur_score * GAMMA + eval_today_se(ticker, all_data[ticker])

            with open(score_path, "w") as f:
                f.write(str(updated_score))

            logger.info(f"Updated score for {ticker}: {updated_score}")
        except Exception:
            logger.exception(f"Failed to update score for {ticker}")

