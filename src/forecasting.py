"""Layer 2 - Monthly rental-revenue forecasting.

TenantEase shows landlords what they *collected*. It cannot tell them what
next quarter's cash flow will look like, so they discover a shortfall only
after it lands. This layer forecasts collected revenue `FORECAST_HORIZON_MONTHS`
ahead with confidence bands, and decomposes the series into trend and
seasonality so the "why" is visible, not just the number.

Design choices worth defending in the technical report:

* **Prophet as the primary model.** The series is monthly with real seasonal
  structure (festival-season squeeze, financial-year-end outflow). Prophet
  handles seasonality, trend changepoints and missing periods with sane priors
  and, critically, emits uncertainty intervals out of the box - a point
  forecast with no band is useless for cash-flow planning.
* **A graceful fallback.** Prophet depends on a compiled backend (cmdstan)
  that can fail to build on a slim deploy host such as Streamlit Cloud. If the
  import or the fit fails, we fall back to statsmodels Holt-Winters
  (exponential smoothing with an additive seasonal term) rather than letting
  the dashboard throw. The output contract is identical either way.
* **Honest backtesting.** We hold out the last `FORECAST_HORIZON_MONTHS`
  observed months, forecast them from the rest, and report MAPE. A forecast
  with no error estimate is decoration.

Caveat stated plainly in the report: 24 monthly points is thin for any
seasonal model. In production this would run on daily collection data, which
gives Prophet far more to work with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

#: What we forecast. The collected-revenue series is the cash-flow line a
#: landlord actually cares about.
TARGET_COLUMN = "rent_collected"
DATE_COLUMN = "month"


@dataclass
class ForecastResult:
    """A forecast plus the context needed to render and trust it."""

    history: pd.DataFrame          # month, actual
    forecast: pd.DataFrame         # month, yhat, yhat_lower, yhat_upper
    components: pd.DataFrame       # month, trend, seasonal (in-sample)
    engine: str                    # "prophet" | "holt-winters"
    backtest_mape: float | None    # % error on a held-out tail, if computable


# --------------------------------------------------------------------------
# Prophet path
# --------------------------------------------------------------------------

def _prophet_available() -> bool:
    try:
        import prophet  # noqa: F401

        return True
    except Exception:
        return False


def _prepare_prophet_backend() -> None:
    """Work around a Prophet packaging bug so it uses its standalone binary.

    Prophet's `CmdStanPyBackend.__init__` unconditionally calls
    `set_cmdstan_path` on its *bundled* cmdstan directory. On several wheels
    that directory ships without a makefile, so the call raises and Prophet
    silently falls back with a misleading "no attribute 'stan_backend'" error.

    The compiled `prophet_model.bin` runs standalone and needs no cmdstan
    toolchain to execute, so we point `CMDSTAN_VERSION` at a directory that
    does not exist: Prophet then skips the broken `set_cmdstan_path` call and
    loads the prebuilt model directly. This keeps Prophet working on a clean
    `pip install` with no separate cmdstan install.
    """
    try:
        from prophet.models import CmdStanPyBackend

        version_dir = (
            __import__("importlib.resources", fromlist=["files"]).files("prophet")
            / "stan_model"
            / f"cmdstan-{CmdStanPyBackend.CMDSTAN_VERSION}"
        )
        # Only neutralise when the bundled dir is present but invalid (no makefile).
        if version_dir.exists() and not (version_dir / "makefile").exists():
            CmdStanPyBackend.CMDSTAN_VERSION = "__standalone__"
    except Exception:
        pass  # if anything is unexpected, the caller still has the fallback


def _fit_prophet(series: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit Prophet and return (forecast_tail, in_sample_components)."""
    _prepare_prophet_backend()
    from prophet import Prophet

    df = series.rename(columns={DATE_COLUMN: "ds", TARGET_COLUMN: "y"})[["ds", "y"]]

    # With only ~24 monthly points, Prophet's default yearly seasonality (10
    # Fourier terms ~= 20 parameters) massively overfits - it will happily
    # forecast negative revenue. We add a low-order, strongly-regularised
    # yearly term instead: enough to capture the festival / FY-end shape, not
    # enough to overfit. (Backtest MAPE ~1-2% vs ~79% with the default.)
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=0.80,
        changepoint_prior_scale=0.05,
    )
    model.add_seasonality("yearly", period=365.25, fourier_order=3, prior_scale=1.0)
    model.fit(df)

    future = model.make_future_dataframe(periods=horizon, freq="MS")
    fc = model.predict(future)

    forecast = (
        fc.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        .rename(columns={"ds": DATE_COLUMN})
        .reset_index(drop=True)
    )
    seasonal_col = "yearly" if "yearly" in fc.columns else "additive_terms"
    components = fc[["ds", "trend", seasonal_col]].rename(
        columns={"ds": DATE_COLUMN, seasonal_col: "seasonal"}
    )
    return forecast, components


# --------------------------------------------------------------------------
# Holt-Winters fallback
# --------------------------------------------------------------------------

def _fit_holt_winters(series: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exponential-smoothing fallback with an identical output contract."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y = series[TARGET_COLUMN].to_numpy(dtype=float)
    months = pd.to_datetime(series[DATE_COLUMN])

    # Seasonality needs >= two full cycles; otherwise fall back to trend only.
    seasonal = "add" if len(y) >= 24 else None
    periods = 12 if seasonal else None

    fit = ExponentialSmoothing(
        y, trend="add", seasonal=seasonal, seasonal_periods=periods,
        initialization_method="estimated",
    ).fit()

    mean = fit.forecast(horizon)
    resid_std = float(np.std(fit.resid)) if hasattr(fit, "resid") else float(np.std(y) * 0.1)
    z = 1.28  # ~80% interval, matching Prophet's interval_width
    future_months = pd.date_range(
        months.iloc[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS"
    )

    forecast = pd.DataFrame(
        {
            DATE_COLUMN: future_months,
            "yhat": mean,
            "yhat_lower": mean - z * resid_std,
            "yhat_upper": mean + z * resid_std,
        }
    )
    components = pd.DataFrame(
        {
            DATE_COLUMN: months,
            "trend": fit.level if hasattr(fit, "level") else y,
            "seasonal": (fit.season if seasonal and hasattr(fit, "season") else np.zeros(len(y))),
        }
    )
    return forecast, components


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def _run(series: pd.DataFrame, horizon: int, prefer_prophet: bool) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Dispatch to Prophet, falling back to Holt-Winters on any failure."""
    if prefer_prophet and _prophet_available():
        try:
            fc, comp = _fit_prophet(series, horizon)
            return fc, comp, "prophet"
        except Exception:
            pass  # fall through to the robust path
    fc, comp = _fit_holt_winters(series, horizon)
    return fc, comp, "holt-winters"


def backtest(monthly: pd.DataFrame, horizon: int = config.FORECAST_HORIZON_MONTHS) -> float | None:
    """MAPE from forecasting the last `horizon` observed months out-of-sample."""
    series = monthly.sort_values(DATE_COLUMN).reset_index(drop=True)
    if len(series) <= horizon + 6:
        return None

    train = series.iloc[:-horizon]
    actual = series.iloc[-horizon:][TARGET_COLUMN].to_numpy(dtype=float)

    fc, _, _ = _run(train, horizon, prefer_prophet=_prophet_available())
    pred = fc["yhat"].to_numpy(dtype=float)
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    return round(mape, 2)


def forecast(
    monthly: pd.DataFrame,
    horizon: int = config.FORECAST_HORIZON_MONTHS,
    *,
    prefer_prophet: bool = True,
) -> ForecastResult:
    """Forecast monthly collected revenue `horizon` months ahead."""
    series = monthly.sort_values(DATE_COLUMN).reset_index(drop=True)

    fc, comp, engine = _run(series, horizon, prefer_prophet)
    history = series[[DATE_COLUMN, TARGET_COLUMN]].rename(columns={TARGET_COLUMN: "actual"})

    return ForecastResult(
        history=history,
        forecast=fc,
        components=comp,
        engine=engine,
        backtest_mape=backtest(series, horizon),
    )
