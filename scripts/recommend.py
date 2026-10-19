"""Generate smart recommendations for the highest-risk tenants.

Run from the project root:

    python scripts/recommend.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data_generator, features, recommendations, risk_model  # noqa: E402


def main() -> None:
    portfolio = data_generator.load()
    model = risk_model.TrainedRiskModel.load()

    table = features.build(portfolio.tenants, portfolio.payments, with_label=False)
    scored = model.score(table)

    # The feature table already carries tenant_name / building / unit_no.
    enriched = table.merge(scored, on="tenant_id").sort_values(
        "risk_score", ascending=False
    )

    engine = recommendations.RecommendationEngine()
    print(f"LLM source: {engine.source_label}\n")

    for row in enriched.head(5).itertuples():
        ctx = {c: getattr(row, c) for c in enriched.columns}
        rec = engine.for_tenant(ctx)
        print(f"{row.tenant_name}  ({row.building} {row.unit_no})")
        print(f"  risk      : {row.risk_score}/100 [{row.risk_category}]  via {rec.source}")
        print(f"  action    : {rec.action}")
        print(f"  reason    : {rec.reason}")
        print(f"  urgency   : {rec.urgency}")
        if rec.suggested_message:
            print(f"  message   : {rec.suggested_message[:120]}...")
        print()


if __name__ == "__main__":
    main()
