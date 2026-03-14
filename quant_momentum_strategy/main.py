from src.data_loader import download_data
from src.feature_engineering import create_features
from src.model_training import train_models
from src.portfolio import build_portfolio
from src.backtest import run_backtest


def main():

    print("STEP 1: Downloading data")

    data = download_data()

    print("STEP 2: Feature engineering")

    features = create_features(data)

    print("STEP 3: Training ML models")

    prob, predictions = train_models(features)

    print("STEP 4: Portfolio construction")

    portfolio = build_portfolio(predictions)

    print("STEP 5: Running backtest")

    results = run_backtest(portfolio)

    portfolio.to_csv("results/predictions.csv")

    print("Results saved successfully.")


if __name__ == "__main__":
    main()