"""Compare baseline strategy performance across stocks."""

from pathlib import Path

from strategy import backtest, load_data, summarize


def compare() -> tuple:
    """Return summary metrics ranked by strategy return and their CSV output path."""
    results = summarize(backtest(load_data())).sort_values("strategy_return", ascending=False)
    output = Path(__file__).parent / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "strategy_comparison.csv"
    results.to_csv(report_path, index=False)
    return results.reset_index(drop=True), report_path


if __name__ == "__main__":
    results, path = compare()
    print(results.to_string(index=False, float_format=lambda value: f"{value:.2%}"))
    print(f"\nSaved {path.relative_to(Path(__file__).parent)}")
