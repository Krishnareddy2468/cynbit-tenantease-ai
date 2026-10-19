"""Feature engineering for the tenant default-risk model.

The ledger is split on the time axis:

    observation window   all cycles up to `cutoff`   -> features
    label window         the `LABEL_HORIZON_MONTHS`  -> ground truth
                         cycles after `cutoff`

Every feature is computed strictly inside the observation window, so the
model is asked the question it would face in production: *given what this
tenant has done so far, will they default in the next quarter?*

A tenant is labelled a default if, within the label window, they accumulate
`DEFAULT_MIN_BAD_CYCLES` delinquent cycles (missed outright, or run
`DELINQUENCY_DPD`+ days past due). Default is a repeated pattern, not a single
unlucky month - a cleaner, more realistic target than a one-cycle trigger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

#: Numeric features fed to the model, in a fixed order.
NUMERIC_FEATURES = [
    "on_time_rate",
    "late_rate",
    "late_count_6m",
    "missed_count",
    "avg_days_past_due",
    "max_days_past_due",
    "std_days_past_due",
    "delay_trend_3m",
    "days_since_last_payment",
    "cycles_observed",
    "monthly_rent",
    "rent_to_income",
    "deposit_months",
    "tenure_months",
]

#: Boolean / categorical features one-hot encoded alongside the numerics.
BOOLEAN_FEATURES = ["deposit_paid", "id_verified"]
CATEGORICAL_FEATURES = ["property_type"]

#: Human-readable labels for the feature-importance chart.
FEATURE_LABELS = {
    "on_time_rate": "On-time payment rate",
    "late_rate": "Late payment rate",
    "late_count_6m": "Late payments (last 6 cycles)",
    "missed_count": "Missed cycles",
    "avg_days_past_due": "Avg days past due",
    "max_days_past_due": "Worst days past due",
    "std_days_past_due": "Payment volatility",
    "delay_trend_3m": "Delay trend (deteriorating)",
    "days_since_last_payment": "Days since last payment",
    "cycles_observed": "Billing cycles observed",
    "monthly_rent": "Monthly rent",
    "rent_to_income": "Rent-to-income ratio",
    "deposit_months": "Deposit (months of rent)",
    "tenure_months": "Tenure (months)",
    "deposit_paid": "Deposit paid",
    "id_verified": "ID verified",
}


def split_cutoff(payments: pd.DataFrame) -> pd.Timestamp:
    """The last cycle that belongs to the observation window."""
    cycles = np.sort(payments["cycle_month"].unique())
    return pd.Timestamp(cycles[-config.LABEL_HORIZON_MONTHS - 1])


def _safe_std(series: pd.Series) -> float:
    return float(series.std()) if len(series) > 1 else 0.0


def _tenant_features(group: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    """Summarise one tenant's observation-window ledger."""
    group = group.sort_values("cycle_month")
    n = len(group)

    paid = group[group["paid"]]
    last_6 = group.tail(6)
    last_3 = group.tail(3)
    prior_3 = group.tail(6).head(3)

    if len(paid):
        last_payment = paid["payment_date"].max()
        days_since = (cutoff + pd.DateOffset(months=1) - last_payment).days
    else:
        days_since = int(config.HISTORY_MONTHS * 30)

    # Positive trend = tenant is getting slower, i.e. deteriorating.
    trend = float(last_3["days_past_due"].mean() - prior_3["days_past_due"].mean()) if len(prior_3) else 0.0

    return {
        "on_time_rate": float((~group["is_late"]).mean()),
        "late_rate": float(group["is_late"].mean()),
        "late_count_6m": int(last_6["is_late"].sum()),
        "missed_count": int((~group["paid"]).sum()),
        "avg_days_past_due": float(group["days_past_due"].mean()),
        "max_days_past_due": float(group["days_past_due"].max()),
        "std_days_past_due": _safe_std(group["days_past_due"]),
        "delay_trend_3m": trend,
        "days_since_last_payment": float(max(days_since, 0)),
        "cycles_observed": n,
    }


def build(
    tenants: pd.DataFrame,
    payments: pd.DataFrame,
    cutoff: pd.Timestamp | None = None,
    *,
    with_label: bool = True,
) -> pd.DataFrame:
    """Build the modelling table: one row per tenant, features + label.

    Parameters
    ----------
    cutoff:
        Last cycle in the observation window. Defaults to the split that
        reserves `LABEL_HORIZON_MONTHS` cycles for labelling.
    with_label:
        When False, features are built over the *entire* ledger and no label
        is produced - this is the inference path the dashboard uses to score
        tenants as of today.
    """
    payments = payments.copy()
    payments["cycle_month"] = pd.to_datetime(payments["cycle_month"])
    payments["payment_date"] = pd.to_datetime(payments["payment_date"])

    if with_label:
        cutoff = cutoff or split_cutoff(payments)
        observed = payments[payments["cycle_month"] <= cutoff]
        labelled = payments[payments["cycle_month"] > cutoff]
    else:
        cutoff = pd.Timestamp(payments["cycle_month"].max())
        observed = payments
        labelled = None

    rows = []
    for tenant_id, group in observed.groupby("tenant_id", sort=False):
        if group.empty:
            continue
        row = {"tenant_id": tenant_id}
        row.update(_tenant_features(group, cutoff))
        rows.append(row)

    behavioural = pd.DataFrame(rows)

    profile_cols = [
        "tenant_id", "tenant_name", "building", "city", "unit_no",
        "property_type", "monthly_rent", "rent_to_income", "deposit_months",
        "deposit_paid", "deposit_amount", "id_verified", "tenure_months",
    ]
    table = behavioural.merge(tenants[profile_cols], on="tenant_id", how="inner")

    if with_label and labelled is not None:
        # Count delinquent cycles (missed, or DELINQUENCY_DPD+ days past due)
        # per tenant in the label window, then flag a default once the count
        # reaches DEFAULT_MIN_BAD_CYCLES - a pattern of trouble, not one month.
        defaulted = (
            labelled.assign(
                _bad=lambda d: (~d["paid"]) | (d["days_past_due"] >= config.DELINQUENCY_DPD)
            )
            .groupby("tenant_id")["_bad"]
            .sum()
            .ge(config.DEFAULT_MIN_BAD_CYCLES)
            .rename("will_default")
            .astype(int)
        )
        table = table.merge(defaulted, on="tenant_id", how="inner")

    return table.reset_index(drop=True)


def to_matrix(table: pd.DataFrame) -> pd.DataFrame:
    """Project the modelling table onto the model's input matrix."""
    numeric = table[NUMERIC_FEATURES].astype(float)
    boolean = table[BOOLEAN_FEATURES].astype(int)
    categorical = pd.get_dummies(
        table[CATEGORICAL_FEATURES].astype(str), prefix=CATEGORICAL_FEATURES, dtype=int
    )
    matrix = pd.concat([numeric, boolean, categorical], axis=1)
    return matrix


def align_matrix(matrix: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Reindex a matrix onto the exact column set the model was trained on."""
    return matrix.reindex(columns=expected, fill_value=0).astype(float)


def pretty(feature: str) -> str:
    """Human-readable name for a (possibly one-hot encoded) feature."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if feature.startswith("property_type_"):
        return f"Property type: {feature.split('_', 2)[-1]}"
    return feature.replace("_", " ").capitalize()
