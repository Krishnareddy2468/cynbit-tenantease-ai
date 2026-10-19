"""Forecasting layer: output contract holds under both engines, backtest sane."""

from __future__ import annotations

from src import config, forecasting


def test_forecast_contract(portfolio):
    result = forecasting.forecast(portfolio.monthly_revenue)
    assert result.engine in {"prophet", "holt-winters"}
    assert len(result.forecast) == config.FORECAST_HORIZON_MONTHS
    assert list(result.forecast.columns) == ["month", "yhat", "yhat_lower", "yhat_upper"]


def test_confidence_band_ordering(portfolio):
    """Lower bound <= point <= upper bound for every forecast month."""
    fc = forecasting.forecast(portfolio.monthly_revenue).forecast
    assert (fc["yhat_lower"] <= fc["yhat"]).all()
    assert (fc["yhat"] <= fc["yhat_upper"]).all()


def test_forecast_values_are_positive(portfolio):
    """Revenue can't be negative - guards against the overfit-Prophet failure."""
    fc = forecasting.forecast(portfolio.monthly_revenue).forecast
    assert (fc["yhat"] > 0).all()


def test_fallback_contract_matches(portfolio):
    """Forcing the Holt-Winters fallback yields the identical contract."""
    result = forecasting.forecast(portfolio.monthly_revenue, prefer_prophet=False)
    assert result.engine == "holt-winters"
    assert len(result.forecast) == config.FORECAST_HORIZON_MONTHS


def test_backtest_is_reasonable(portfolio):
    mape = forecasting.backtest(portfolio.monthly_revenue)
    assert mape is not None
    assert 0 <= mape < 30, f"backtest MAPE {mape}% is implausibly high"
