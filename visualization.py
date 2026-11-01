"""Create charts for the project's stock data and backtests."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strategy import backtest, load_data
from models import (
    MODELS,
    TRAIN_RATIO,
    backtest_model,
    chronological_split,
    run_evaluation,
)

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "outputs"


def create_charts(output_dir: str | Path = OUTPUT_DIR) -> list[Path]:
    """Create the stock-price and moving-average equity charts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    data = load_data()
    results = backtest(data)

    fig, ax = plt.subplots(figsize=(11, 6))

    for stock, frame in data.groupby("stock", sort=True):
        prices = frame["close"] / frame["close"].iloc[0] * 100
        ax.plot(frame["day"], prices, linewidth=1, alpha=0.8, label=stock)

    ax.set_title("Stock prices (normalized to 100)")
    ax.set_xlabel("Day")
    ax.set_ylabel("Index")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    prices_path = output / "price_overview.png"
    fig.savefig(prices_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))

    for stock, frame in results.groupby("stock", sort=True):
        ax.plot(frame["day"], frame["equity"], linewidth=1, label=stock)

    ax.set_title("10/30-day moving-average strategy equity")
    ax.set_xlabel("Day")
    ax.set_ylabel("Growth of 1.0")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    equity_path = output / "strategy_equity.png"
    fig.savefig(equity_path, dpi=160)
    plt.close(fig)

    return [prices_path, equity_path]


def create_model_chart(output_dir: str | Path = OUTPUT_DIR) -> Path:
    """Compare the selected models on the holdout data."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    selected, _ = run_evaluation(output_dir=str(output))
    train, test = chronological_split(load_data(), TRAIN_RATIO)

    fig, ax = plt.subplots(figsize=(11, 6))

    for name in selected["model"]:
        result = backtest_model(test, MODELS[name], warmup_data=train)
        daily_returns = result.groupby("day")["strategy_return"].mean()
        equity = (1 + daily_returns).cumprod()

        ax.plot(equity.index, equity.values, linewidth=1.7, label=name)

    market = test.sort_values(["stock", "day"]).copy()
    market["market_return"] = (
        market.groupby("stock")["close"].pct_change().fillna(0)
    )
    market_returns = market.groupby("day")["market_return"].mean()
    market_equity = (1 + market_returns).cumprod()

    ax.plot(
        market_equity.index,
        market_equity.values,
        linestyle="--",
        color="black",
        label="equal-weight buy-and-hold",
    )

    ax.set_title("Top three strategies vs. buy-and-hold benchmark (holdout)")
    ax.set_xlabel("Day")
    ax.set_ylabel("Growth of 1.0")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    chart_path = output / "five_model_test_equity.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)

    return chart_path


if __name__ == "__main__":
    print(f"Saved {create_model_chart().relative_to(ROOT)}")