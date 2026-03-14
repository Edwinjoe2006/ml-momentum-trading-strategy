import pandas as pd
import ta

def create_features(dataset):

    frames = []

    for stock, df in dataset.items():

        df = df.copy()

        # Returns
        df["ret1"] = df["close"].pct_change()
        df["ret5"] = df["close"].pct_change(5)
        df["ret10"] = df["close"].pct_change(10)

        # Momentum signals
        df["momentum_20"] = df["close"] / df["close"].shift(20) - 1
        df["momentum_60"] = df["close"] / df["close"].shift(60) - 1
        df["momentum_120"] = df["close"] / df["close"].shift(120) - 1

        # Volatility
        df["volatility"] = df["ret1"].rolling(20).std()
        df["volatility_60"] = df["ret1"].rolling(60).std()

        # Moving averages
        df["ma10"] = df["close"].rolling(10).mean()
        df["ma50"] = df["close"].rolling(50).mean()
        df["ma200"] = df["close"].rolling(200).mean()

        df["ma_ratio"] = df["ma10"] / df["ma50"]

        # Trend strength
        df["trend_strength"] = df["close"] / df["ma200"]

        # RSI
        df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()

        # MACD
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()

        # Volume features
        df["volume_ma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

        # Price position
        df["price_position"] = (
            (df["close"] - df["low"].rolling(20).min()) /
            (df["high"].rolling(20).max() - df["low"].rolling(20).min())
        )

        # Target
        df["future_return"] = df["close"].pct_change(5).shift(-5)
        df["target"] = (df["future_return"] > 0).astype(int)

        df["stock"] = stock

        frames.append(df)

    data = pd.concat(frames)

    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    data = data.dropna()

    return data
