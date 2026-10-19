"""Central configuration for the TenantEase AI Intelligence Engine.

Every tunable constant lives here so the synthetic data generator, the models
and the Streamlit app agree on schema, paths and business thresholds.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

TENANTS_CSV = DATA_DIR / "tenants.csv"
PAYMENTS_CSV = DATA_DIR / "payments.csv"
MONTHLY_CSV = DATA_DIR / "monthly_revenue.csv"

RISK_MODEL_PATH = MODEL_DIR / "risk_model.joblib"
RISK_METRICS_PATH = MODEL_DIR / "risk_metrics.json"

# --- Reproducibility -------------------------------------------------------

RANDOM_SEED = 42

# --- Synthetic portfolio shape --------------------------------------------

N_TENANTS = 520
HISTORY_MONTHS = 24
#: The ledger ends on the 1st of this month; "today" is the snapshot date.
SNAPSHOT_DAY = 1

PROPERTY_TYPES = ("1BHK", "2BHK", "3BHK", "Studio", "PG Shared", "Commercial")

#: Base monthly rent in INR by property type (mean, std).
RENT_PROFILE = {
    "1BHK": (12_000, 2_500),
    "2BHK": (19_000, 4_000),
    "3BHK": (28_000, 6_000),
    "Studio": (9_500, 2_000),
    "PG Shared": (6_500, 1_200),
    "Commercial": (45_000, 12_000),
}

#: Buildings in the simulated portfolio: name -> (city, unit count).
BUILDINGS = {
    "Shanti Residency": ("Jaipur", 90),
    "Malviya Heights": ("Jaipur", 85),
    "Vaishali Enclave": ("Jaipur", 80),
    "Prithvi Towers": ("Udaipur", 70),
    "Aravalli Court": ("Jodhpur", 65),
    "Nirman Plaza": ("Jaipur", 70),
    "Sarthak Nivas": ("Ajmer", 60),
}

#: Months where Indian rental collection is seasonally stronger/weaker.
#: Keyed by calendar month; value is a multiplier on on-time payment odds.
#: Oct/Nov = festival spend squeeze, Apr/Jun = academic intake & appraisals.
SEASONAL_PAYMENT_PRESSURE = {
    1: 1.00, 2: 1.02, 3: 0.96,   # Mar: financial year-end tax outflow
    4: 1.04, 5: 1.01, 6: 1.03,   # Apr-Jun: appraisals, academic intake
    7: 0.99, 8: 0.98, 9: 0.97,
    10: 0.92, 11: 0.93,          # Oct-Nov: Diwali spend squeeze
    12: 1.01,
}

# --- Business rules --------------------------------------------------------

#: A payment is "late" once it crosses this many days past the due date.
LATE_THRESHOLD_DAYS = 5

#: Days-past-due value stamped on a missed rent cycle in the ledger. A miss is
#: treated as this delinquent for downstream aggregation.
DEFAULT_DAYS_PAST_DUE = 30

#: A single cycle counts as "delinquent" once it is missed outright or runs
#: this many days past due.
DELINQUENCY_DPD = 15

#: Ground-truth default label: a tenant is flagged as a default risk only when
#: they rack up this many delinquent cycles inside the label window. Default is
#: a *pattern* of trouble, not one unlucky month - this matches how a landlord
#: actually decides a tenant is a problem, and gives the model a cleaner signal
#: than a single-cycle trigger (which is dominated by month-to-month noise).
DEFAULT_MIN_BAD_CYCLES = 2

#: Forward-looking window (months) the risk model predicts over.
LABEL_HORIZON_MONTHS = 3

#: Risk score -> category thresholds (score is 0-100).
RISK_BANDS = (
    (0, 25, "Low"),
    (25, 50, "Medium"),
    (50, 75, "High"),
    (75, 101, "Critical"),
)

RISK_COLORS = {
    "Low": "#2BB673",
    "Medium": "#E8B339",
    "High": "#E8743B",
    "Critical": "#B5253A",
}

# --- Forecasting -----------------------------------------------------------

FORECAST_HORIZON_MONTHS = 3

# --- LLM layer -------------------------------------------------------------

LLM_PROVIDER = os.getenv("TENANTEASE_LLM_PROVIDER", "auto").strip().lower()
LLM_MODEL = os.getenv("TENANTEASE_LLM_MODEL", "").strip()

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

LLM_MAX_TOKENS = 700
LLM_TEMPERATURE = 0.3


def risk_category(score: float) -> str:
    """Map a 0-100 risk score onto its business-facing band."""
    for low, high, label in RISK_BANDS:
        if low <= score < high:
            return label
    return "Critical"
