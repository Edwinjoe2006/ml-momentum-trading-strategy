Machine Learning Momentum Strategy

Overview
This project implements a machine learning driven momentum trading strategy for equities. The goal is to predict short-term stock returns using a combination of technical indicators and ensemble machine learning models.

The system ranks stocks weekly based on predicted return probabilities and allocates capital to the highest confidence opportunities.

Data Source
Yahoo Finance API (yfinance)

Universe
AAPL, MSFT, GOOGL, AMZN, META, TSLA, JPM, V, JNJ, BRK.B

Workflow

1. Data Collection
Historical OHLCV stock data is downloaded using Yahoo Finance.

2. Feature Engineering
Multiple technical indicators are generated including:

• Multi-horizon momentum signals  
• Volatility measures  
• Moving average trend indicators  
• RSI and MACD signals  
• Volume-based indicators  
• Price positioning signals  

3. Model Training
Three machine learning models are trained:

• Logistic Regression  
• Random Forest  
• XGBoost  

Predictions from the models are combined using an ensemble approach.

4. Portfolio Construction
Stocks are ranked based on predicted probabilities and the top opportunities are selected each week. Portfolio weights are assigned based on prediction confidence.

5. Backtesting
The strategy is evaluated using historical simulation with transaction costs.

Performance Metrics

• Cumulative Return  
• Annualized Return  
• Volatility  
• Sharpe Ratio  
• Sortino Ratio  
• Maximum Drawdown  
• Hit Rate  

Outputs

The system generates:

results/performance.png – strategy performance chart  
results/feature_importance.png – feature importance visualization  
results/predictions.csv – model predictions  
results/weekly_returns.csv – weekly strategy returns

Objective

This project demonstrates how machine learning techniques can be applied to systematic trading strategies by combining financial feature engineering, predictive modeling, and portfolio optimization.