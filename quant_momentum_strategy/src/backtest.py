import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


def run_backtest(portfolio):

    # Calculate strategy returns
    portfolio["strategy_return"] = portfolio["future_return"] * portfolio["weight"]

    weekly_returns = portfolio.groupby("week")["strategy_return"].sum()

    # Simulated transaction cost
    weekly_returns = weekly_returns - 0.002

    # Cumulative performance
    cumulative = (1 + weekly_returns).cumprod()

    # Basic performance statistics
    annual_return = weekly_returns.mean() * 52
    volatility = weekly_returns.std() * np.sqrt(52)

    sharpe = annual_return / volatility

    # Downside risk for Sortino ratio
    downside = weekly_returns[weekly_returns < 0].std()
    sortino = annual_return / (downside * np.sqrt(52))

    drawdown = cumulative / cumulative.cummax() - 1
    max_dd = drawdown.min()

    hit_rate = (weekly_returns > 0).mean()

    print("\n===== STRATEGY PERFORMANCE =====")
    print("Annual Return:", annual_return)
    print("Volatility:", volatility)
    print("Sharpe Ratio:", sharpe)
    print("Sortino Ratio:", sortino)
    print("Max Drawdown:", max_dd)
    print("Hit Rate:", hit_rate)


    # Benchmark comparison (S&P 500)


    benchmark = yf.download("^GSPC", start="2017-01-01", end="2025-01-01")["Close"]

    benchmark_returns = benchmark.pct_change().dropna()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()

    # Plot performance
 
    plt.figure(figsize=(12,6))

    plt.plot(
        cumulative.index.astype(str),
        cumulative.values,
        label="ML Strategy",
        linewidth=2
    )

    plt.plot(
        benchmark_cumulative.index.astype(str),
        benchmark_cumulative.values,
        label="S&P 500 Benchmark",
        alpha=0.7
    )

    plt.title("ML Momentum Strategy vs Benchmark")
    plt.xlabel("Week")
    plt.ylabel("Cumulative Return")

    plt.xticks(rotation=45)

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig("results/performance.png")

    plt.show()

    return weekly_returns