# ML Momentum Trading Strategy

This project implements a **Machine Learning based momentum trading strategy** using historical stock market data.

The system predicts short-term stock returns using technical indicators and machine learning models, then constructs a portfolio based on predicted probabilities.

---

## Strategy Pipeline

1. Data Collection  
Historical OHLCV data is downloaded from **Yahoo Finance** using the `yfinance` API.

2. Feature Engineering  
Technical indicators and statistical signals are generated including:

- Momentum indicators (20, 60, 120 periods)
- Volatility signals
- Moving average trend indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Volume indicators
- Price position signals

3. Machine Learning Models  

The following models are trained:

- Logistic Regression
- Random Forest
- XGBoost

Predictions from all models are combined using an **ensemble approach**.

4. Portfolio Construction  

Stocks are ranked by predicted probability and the **top opportunities are selected weekly**.

Portfolio weights are assigned based on prediction confidence.

5. Backtesting  

The strategy is evaluated using historical simulation.

Performance metrics include:

- Annual Return
- Volatility
- Sharpe Ratio
- Maximum Drawdown
- Hit Rate

---

## Project Structure
