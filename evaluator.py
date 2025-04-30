import os

import yfinance as yf

from frontend_utils import give_preds, TICKER_PORT_MAP

GAMMA = 0.99

def eval_today_se(ticker):
    data = yf.download(ticker, period='2d', interval='1d').reset_index()[['Close']]
    today_actual = data.values[-1][0]
    _,_, today_predicted = give_preds(ticker, 1)
    return (today_actual-today_predicted[0])**2

if __name__ == '__main__':
    for ticker in list(TICKER_PORT_MAP.keys()):
        score_path = f"evaluation/{ticker}/score.txt"
        if os.path.exists(score_path):
            with open(score_path, "r") as f:
                cur_score = float(f.read().strip())
        else:
            cur_score = 0.0
        
        updated_score = cur_score*GAMMA+eval_today_se(ticker)
        with open(score_path,"w") as f:
            f.write(str(updated_score))
    