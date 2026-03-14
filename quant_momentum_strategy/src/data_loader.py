import yfinance as yf
import pandas as pd

STOCKS = [
    "AAPL","MSFT","GOOGL","AMZN","META",
    "TSLA","JPM","V","JNJ","BRK-B"
]

START = "2017-01-01"
END = "2025-01-01"


def download_data():

    print("Downloading data from Yahoo Finance...")

    data = yf.download(STOCKS, start=START, end=END)

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    dataset = {}

    for stock in STOCKS:

        df = pd.DataFrame()

        df["close"] = close[stock]
        df["high"] = high[stock]
        df["low"] = low[stock]
        df["volume"] = volume[stock]

        dataset[stock] = df

    return dataset