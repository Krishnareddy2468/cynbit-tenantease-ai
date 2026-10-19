"""Synthetic tenant and payment-ledger generator.

TenantEase already stores everything this module fabricates: tenant profiles,
unit assignments, monthly rent demands and the payment audit trail. Cynbit's
production data is not public, so the proof of concept simulates a portfolio
with the same schema and with behavioural structure a real rent roll shows.

The generative process is deliberately causal rather than random:

    latent reliability  ->  payment delay distribution  ->  observed ledger
                                      ^
                        seasonal pressure + rent burden + tenure

Because the label (future default) is produced by the same latent trait that
drives the observable history, a model *can* learn the relationship - but only
through noisy proxies, which is exactly the situation in production.

The ledger is split in time:

    months  1..21   observation window  -> features
    months 22..24   label window        -> ground-truth default flag

Features never read the label window, so there is no temporal leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from faker import Faker

from . import config


@dataclass(frozen=True)
class GeneratedPortfolio:
    """The three tables the rest of the pipeline consumes."""

    tenants: pd.DataFrame
    payments: pd.DataFrame
    monthly_revenue: pd.DataFrame


def _month_starts(n_months: int, end: pd.Timestamp) -> pd.DatetimeIndex:
    """The first of each month for the `n_months` ending at `end` inclusive."""
    return pd.date_range(end=end, periods=n_months, freq="MS")


def _snapshot_date() -> pd.Timestamp:
    """The 'today' the simulated portfolio is observed from."""
    return pd.Timestamp.today().normalize().replace(day=config.SNAPSHOT_DAY)


def _build_tenants(rng: np.random.Generator, fake: Faker) -> pd.DataFrame:
    """Create the tenant roster with latent behavioural traits."""
    units: list[tuple[str, str, str]] = []
    for building, (city, count) in config.BUILDINGS.items():
        for i in range(count):
            floor = i // 6 + 1
            unit_no = f"{chr(65 + (i // 6) % 6)}-{floor}{i % 6 + 1:02d}"
            units.append((building, city, unit_no))

    rng.shuffle(units)
    units = units[: config.N_TENANTS]

    rows = []
    for idx, (building, city, unit_no) in enumerate(units, start=1):
        prop_type = rng.choice(
            config.PROPERTY_TYPES, p=[0.26, 0.28, 0.14, 0.12, 0.14, 0.06]
        )
        mean_rent, std_rent = config.RENT_PROFILE[prop_type]
        rent = float(np.clip(rng.normal(mean_rent, std_rent), mean_rent * 0.55, None))
        rent = round(rent / 500) * 500

        # Latent reliability in [0, 1]: the trait the model must recover.
        # Beta(5, 2) -> most tenants pay well, a meaningful tail does not.
        reliability = float(rng.beta(5.0, 2.0))

        # Rent burden: rent as a share of declared income. High burden erodes
        # reliability in a way the ledger will eventually reveal.
        income = float(rng.normal(rent * 3.6, rent * 1.1))
        income = max(income, rent * 1.4)
        rent_to_income = rent / income

        tenure_months = int(rng.integers(1, config.HISTORY_MONTHS + 13))
        deposit_months = float(rng.choice([0.0, 1.0, 2.0, 3.0], p=[0.08, 0.30, 0.46, 0.16]))

        rows.append(
            {
                "tenant_id": f"TN{idx:04d}",
                "tenant_name": fake.name(),
                "phone": f"+91{rng.integers(70, 99)}{rng.integers(10**7, 10**8)}",
                "building": building,
                "city": city,
                "unit_no": unit_no,
                "property_type": prop_type,
                "monthly_rent": rent,
                "declared_income": round(income, -2),
                "rent_to_income": round(rent_to_income, 4),
                "deposit_months": deposit_months,
                "deposit_paid": bool(deposit_months > 0),
                "deposit_amount": round(rent * deposit_months, -2),
                "tenure_months": tenure_months,
                "id_verified": bool(rng.random() < 0.82),
                "latent_reliability": reliability,
            }
        )

    return pd.DataFrame(rows)


def _payment_delay(
    rng: np.random.Generator,
    reliability: float,
    seasonal: float,
    rent_to_income: float,
    months_tenured: int,
) -> int | None:
    """Days past due for one rent cycle, or None if the tenant did not pay.

    Delay and miss probability are both driven by a single per-cycle risk
    term, `risk = 1 - effective reliability`. Both responses are convex in
    risk: reliable tenants cluster tightly around the due date while the
    unreliable tail degrades sharply. That convexity is what gives the
    classifier a learnable signal.
    """
    burden_penalty = float(np.clip((rent_to_income - 0.30) * 0.55, 0.0, 0.18))
    tenure_bonus = min(months_tenured, 36) / 36 * 0.06
    eff = float(np.clip(reliability * seasonal - burden_penalty + tenure_bonus, 0.05, 0.99))
    risk = 1.0 - eff

    # Probability the cycle is missed outright.
    if rng.random() < 0.55 * risk**2.2:
        return None

    # Days past due: tight for reliable tenants, heavy-tailed for risky ones.
    scale = 0.9 + risk**2.1 * 44.0
    delay = rng.exponential(scale) - rng.uniform(0.0, 2.5)
    return int(np.clip(round(delay), 0, 75))


def _build_payments(rng: np.random.Generator, tenants: pd.DataFrame) -> pd.DataFrame:
    """Expand the roster into a month-by-month payment ledger."""
    snapshot = _snapshot_date()
    cycles = _month_starts(config.HISTORY_MONTHS, snapshot - pd.DateOffset(months=1))

    records = []
    for tenant in tenants.itertuples(index=False):
        # A tenant only has ledger rows for cycles after they moved in.
        active_from = max(0, config.HISTORY_MONTHS - tenant.tenure_months)
        for offset, due_date in enumerate(cycles):
            if offset < active_from:
                continue

            seasonal = config.SEASONAL_PAYMENT_PRESSURE[due_date.month]
            months_tenured = offset - active_from
            delay = _payment_delay(
                rng,
                tenant.latent_reliability,
                seasonal,
                tenant.rent_to_income,
                months_tenured,
            )

            paid = delay is not None
            records.append(
                {
                    "tenant_id": tenant.tenant_id,
                    "building": tenant.building,
                    "cycle_month": due_date,
                    "due_date": due_date + pd.Timedelta(days=4),
                    "amount_due": tenant.monthly_rent,
                    "amount_paid": tenant.monthly_rent if paid else 0.0,
                    "paid": paid,
                    "days_past_due": int(delay) if paid else config.DEFAULT_DAYS_PAST_DUE,
                    "payment_date": (
                        due_date + pd.Timedelta(days=4 + int(delay)) if paid else pd.NaT
                    ),
                }
            )

    payments = pd.DataFrame(records)
    payments["is_late"] = payments["days_past_due"] > config.LATE_THRESHOLD_DAYS
    return payments.sort_values(["tenant_id", "cycle_month"]).reset_index(drop=True)


def _build_monthly_revenue(payments: pd.DataFrame, tenants: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the ledger into the monthly series Prophet consumes."""
    total_units = sum(count for _, count in config.BUILDINGS.values())

    monthly = (
        payments.groupby("cycle_month")
        .agg(
            rent_demanded=("amount_due", "sum"),
            rent_collected=("amount_paid", "sum"),
            active_tenants=("tenant_id", "nunique"),
            late_payments=("is_late", "sum"),
            missed_payments=("paid", lambda s: int((~s).sum())),
        )
        .reset_index()
    )

    monthly["collection_rate"] = monthly["rent_collected"] / monthly["rent_demanded"]
    monthly["occupancy_rate"] = monthly["active_tenants"] / min(total_units, len(tenants))
    monthly = monthly.rename(columns={"cycle_month": "month"})
    return monthly


def generate(seed: int = config.RANDOM_SEED) -> GeneratedPortfolio:
    """Generate a complete synthetic TenantEase portfolio."""
    rng = np.random.default_rng(seed)
    fake = Faker("en_IN")
    Faker.seed(seed)

    tenants = _build_tenants(rng, fake)
    payments = _build_payments(rng, tenants)
    monthly = _build_monthly_revenue(payments, tenants)

    return GeneratedPortfolio(tenants=tenants, payments=payments, monthly_revenue=monthly)


def write(portfolio: GeneratedPortfolio) -> None:
    """Persist the portfolio to data/ as CSV."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    portfolio.tenants.to_csv(config.TENANTS_CSV, index=False)
    portfolio.payments.to_csv(config.PAYMENTS_CSV, index=False)
    portfolio.monthly_revenue.to_csv(config.MONTHLY_CSV, index=False)


def load() -> GeneratedPortfolio:
    """Read the portfolio from data/, generating it first if absent."""
    if not (config.TENANTS_CSV.exists() and config.PAYMENTS_CSV.exists()):
        portfolio = generate()
        write(portfolio)
        return portfolio

    tenants = pd.read_csv(config.TENANTS_CSV)
    payments = pd.read_csv(
        config.PAYMENTS_CSV, parse_dates=["cycle_month", "due_date", "payment_date"]
    )
    monthly = pd.read_csv(config.MONTHLY_CSV, parse_dates=["month"])
    return GeneratedPortfolio(tenants=tenants, payments=payments, monthly_revenue=monthly)
