# Alpha Arena

Alpha Arena is a Python project for trying out stock-trading strategies using historical stock prices. It includes a moving-average strategy, other sample strategies, and tools to compare their results and make charts.

## Project files

* `data/train.csv` — daily price and volume data for 12 stocks. The project uses the first 2,500 days by default.
* `Alpha_Arena_Official_Rulebook.pdf` — the competition rulebook.
* `strategy.py` — checks the data and runs the moving-average strategy.
* `models.py` — tests five trading strategies.
* `compare.py` — compares the moving-average strategy with buy-and-hold.
* `visualization.py` — creates charts of the results.
* `outputs/` — stores the reports and charts created by the scripts.

## Setup

Python 3.10 or newer is recommended. Open PowerShell in the project folder and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the project

Run these commands in PowerShell:

```powershell
python strategy.py
python compare.py
python models.py
python visualization.py
```

## How it works

The basic strategy compares the 10-day and 30-day moving averages. It checks the signal using the day’s closing price, then applies it to the next day’s return.

The project also tests momentum, mean reversion, breakout, and RSI strategies. It uses the first 70% of the 2,500-day window to rank the five strategies by Sharpe ratio. The remaining 30% is used to check how they perform on later data. The three highest-ranked strategies are included in the results.

For these tests, the signal is delayed by one day and a 0.3% cost is applied when a position changes. The buy-and-hold benchmark gives each stock an equal share. Results include return and risk measures such as Sharpe ratio and information ratio.

The scripts save their results in the `outputs/` folder:

* `five_model_comparison.csv` — results for the selected three strategies.
* `five_model_test_returns.csv` — daily returns during the holdout period.
* `best_model.txt` — the strategy with the highest training Sharpe ratio.

## Competition notes

The settings in this project are for local testing. The official competition may use different costs, limits, and scoring rules, so check the organizer’s instructions before submitting. Do not use hidden evaluation data or information from the future. Disclose AI or outside help if the organizer asks you to.

## Data format

The CSV contains `day`, `stock`, `open`, `high`, `low`, `close`, and `volume`.

The dataset has 44,496 rows for 12 stocks and no missing values in these columns. The `day` column is an integer time index, not a calendar date. By default, the scripts use days 0 through 2,499.