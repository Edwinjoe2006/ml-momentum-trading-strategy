import pandas as pd


def build_portfolio(predictions):

    predictions["week"] = predictions.index.to_period("W")

    portfolios = []

    for week, group in predictions.groupby("week"):

        ranked = group.sort_values(
            "probability",
            ascending=False
        )

        top2 = ranked.head(2).copy()

        top2["weight"] = 0.5

        portfolios.append(top2)

    portfolio = pd.concat(portfolios)

    return portfolio