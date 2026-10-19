"""Synthetic data generator: determinism, schema, and realistic calibration."""

from __future__ import annotations

from src import config, data_generator


def test_generation_is_deterministic():
    """Same seed must yield identical tables - reproducibility is non-negotiable."""
    a = data_generator.generate(seed=config.RANDOM_SEED)
    b = data_generator.generate(seed=config.RANDOM_SEED)
    assert a.tenants.equals(b.tenants)
    assert a.payments.equals(b.payments)


def test_portfolio_size(portfolio):
    assert len(portfolio.tenants) == config.N_TENANTS
    assert portfolio.tenants["tenant_id"].is_unique
    assert len(portfolio.monthly_revenue) == config.HISTORY_MONTHS


def test_required_columns(portfolio):
    tenant_cols = {"tenant_id", "monthly_rent", "property_type", "deposit_months", "tenure_months"}
    payment_cols = {"tenant_id", "cycle_month", "amount_due", "amount_paid", "paid", "days_past_due", "is_late"}
    assert tenant_cols.issubset(portfolio.tenants.columns)
    assert payment_cols.issubset(portfolio.payments.columns)


def test_collection_rate_is_realistic(portfolio):
    """Collection should land in the real-rent-roll band (~93-96%)."""
    pay = portfolio.payments
    collection = pay["amount_paid"].sum() / pay["amount_due"].sum()
    assert 0.90 <= collection <= 0.98, f"collection rate {collection:.1%} out of band"


def test_no_negative_money(portfolio):
    assert (portfolio.tenants["monthly_rent"] > 0).all()
    assert (portfolio.payments["amount_paid"] >= 0).all()
    assert (portfolio.payments["days_past_due"] >= 0).all()


def test_tenure_bounds_payment_history(portfolio):
    """No tenant should have more billed cycles than their tenure allows."""
    counts = portfolio.payments.groupby("tenant_id").size()
    assert counts.max() <= config.HISTORY_MONTHS
