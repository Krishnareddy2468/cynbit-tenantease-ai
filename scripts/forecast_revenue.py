"""Forecast monthly rental revenue and print the result.

Run from the project root:

    python scripts/forecast_revenue.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data_generator, forecasting  # noqa: E402


def main() -> None:
    portfolio = data_generator.load()
    result = forecasting.forecast(portfolio.monthly_revenue)

    print(f"engine            : {result.engine}")
    print(f"history months    : {len(result.history)}")
    print(
        "backtest MAPE     : "
        + (f"{result.backtest_mape:.2f}%" if result.backtest_mape is not None else "n/a")
    )

    last = result.history.tail(3)
    print("\nrecent actuals")
    print("-" * 36)
    for row in last.itertuples():
        print(f"  {row.month:%Y-%m}   INR {row.actual:>12,.0f}")

    print(f"\n{forecasting.config.FORECAST_HORIZON_MONTHS}-month forecast (80% interval)")
    print("-" * 46)
    for row in result.forecast.itertuples():
        print(
            f"  {row.month:%Y-%m}   INR {row.yhat:>12,.0f}   "
            f"[{row.yhat_lower:>11,.0f} .. {row.yhat_upper:>11,.0f}]"
        )


if __name__ == "__main__":
    main()
