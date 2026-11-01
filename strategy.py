from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "train.csv"
MAX_DAYS = 2500

REQUIRED_COLUMNS = {
    "day", "stock", "open", "high", "low", "close", "volume"
}


def load_data(
    path: str | Path = DATA_PATH,
    max_days: int | None = MAX_DAYS,
) -> pd.DataFrame:
    """Read the stock data and check that it has the columns we need."""
    frame = pd.read_csv(path)

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )

    if frame.empty:
        raise ValueError("The dataset is empty")

    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Some required columns contain missing values")

    frame = frame.sort_values(["stock", "day"]).reset_index(drop=True)

    # Keep only the first N days when a limit is set.
    if max_days is not None:
        if max_days < 1:
            raise ValueError("max_days must be positive or None")

        days = sorted(frame["day"].unique())[:max_days]
        frame = frame[frame["day"].isin(days)].reset_index(drop=True)

    return frame


def backtest(
    data: pd.DataFrame,
    fast_window: int = 10,
    slow_window: int = 30,
) -> pd.DataFrame:
    """Test the moving-average strategy using the next day's return."""
    if fast_window < 1 or slow_window < 2 or fast_window >= slow_window:
        raise ValueError("Use windows where 1 <= fast_window < slow_window")

    required = {"day", "stock", "close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"Data is missing required columns: {', '.join(sorted(missing))}"
        )

    result = data.sort_values(["stock", "day"]).copy()
    grouped = result.groupby("stock", sort=False)["close"]

    result["fast_ma"] = grouped.transform(
        lambda prices: prices.rolling(
            fast_window, min_periods=fast_window
        ).mean()
    )
    result["slow_ma"] = grouped.transform(
        lambda prices: prices.rolling(
            slow_window, min_periods=slow_window
        ).mean()
    )

    # Hold the stock when the fast average is above the slow average.
    result["signal"] = (
        result["fast_ma"] > result["slow_ma"]
    ).astype(float)

    # Use yesterday's signal for today's return.
    result["position"] = (
        result.groupby("stock", sort=False)["signal"]
        .shift(1)
        .fillna(0.0)
    )

    result["market_return"] = (
        result.groupby("stock", sort=False)["close"]
        .pct_change()
        .fillna(0.0)
    )
    result["strategy_return"] = (
        result["position"] * result["market_return"]
    )

    result["equity"] = result.groupby("stock", sort=False)[
        "strategy_return"
    ].transform(lambda returns: (1.0 + returns).cumprod())

    result["buy_hold_equity"] = result.groupby("stock", sort=False)[
        "market_return"
    ].transform(lambda returns: (1.0 + returns).cumprod())

    return result.reset_index(drop=True)


def summarize(backtest_data: pd.DataFrame) -> pd.DataFrame:
    """Show the return, volatility, and largest drawdown for each stock."""
    summaries = []

    for stock, frame in backtest_data.groupby("stock", sort=True):
        returns = frame["strategy_return"]
        equity = frame["equity"]

        if len(returns) > 1:
            volatility = returns.std(ddof=1) * (252 ** 0.5)
        else:
            volatility = 0.0

        drawdown = equity / equity.cummax() - 1.0

        summaries.append({
            "stock": stock,
            "strategy_return": equity.iloc[-1] - 1.0,
            "buy_hold_return": frame["buy_hold_equity"].iloc[-1] - 1.0,
            "annualized_volatility": volatility,
            "max_drawdown": drawdown.min(),
        })

    return pd.DataFrame(summaries)


if __name__ == "__main__":
    prices = load_data()
    tested = backtest(prices)

    print(
        f"Rows: {len(prices):,} | "
        f"Stocks: {prices['stock'].nunique()} | "
        f"Days: {prices['day'].nunique():,}"
    )

    print(
        summarize(tested).to_string(
            index=False,
            float_format=lambda value: f"{value:.2%}",
        )
    )