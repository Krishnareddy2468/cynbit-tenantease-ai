"""Regenerate the synthetic TenantEase portfolio into data/."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, data_generator  # noqa: E402


def main() -> None:
    portfolio = data_generator.generate()
    data_generator.write(portfolio)

    print(f"tenants         : {len(portfolio.tenants):>6,} rows -> {config.TENANTS_CSV.name}")
    print(f"payment ledger  : {len(portfolio.payments):>6,} rows -> {config.PAYMENTS_CSV.name}")
    print(f"monthly revenue : {len(portfolio.monthly_revenue):>6,} rows -> {config.MONTHLY_CSV.name}")

    pay = portfolio.payments
    print()
    print(f"on-time rate    : {(~pay['is_late']).mean():.1%}")
    print(f"missed cycles   : {(~pay['paid']).mean():.1%}")
    print(f"collection rate : {pay['amount_paid'].sum() / pay['amount_due'].sum():.1%}")


if __name__ == "__main__":
    main()
