"""Train the Layer 1 tenant risk model and persist it to models/.

Run from the project root:

    python scripts/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data_generator, risk_model  # noqa: E402


def _bar(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def main() -> None:
    portfolio = data_generator.load()
    trained = risk_model.train(portfolio.tenants, portfolio.payments)
    m = trained.metrics

    _bar("Dataset")
    print(f"tenants            : {m['n_tenants']}")
    print(f"features           : {m['n_features']}")
    print(f"default base rate  : {m['default_rate']:.1%}")
    print(f"held-out test size : {m['test_size']}")

    _bar("Model comparison (ROC-AUC)")
    print(f"{'':20s}{'CV':>10}{'test':>10}")
    print(f"{'logistic baseline':20s}{m['baseline_logreg_cv_auc']:>10.3f}{m['baseline_logreg_test_auc']:>10.3f}")
    print(f"{'XGBoost':20s}{m['cv_auc_mean']:>10.3f}{m['test_auc']:>10.3f}")
    print(f"  (XGBoost CV std   : +/- {m['cv_auc_std']:.3f})")
    if m.get("oracle_auc") is not None:
        print(f"  oracle ceiling    : {m['oracle_auc']:.3f}  (Bayes-optimal; model should sit below)")
    lift = m["test_auc"] - m["baseline_logreg_test_auc"]
    print(f"  lift over baseline: {lift:+.3f} (test)")

    _bar(f"Held-out metrics @ tuned threshold {m['operating_threshold']:.2f}")
    print(f"average precision  : {m['test_average_precision']:.3f}")
    print(f"precision          : {m['test_precision']:.3f}")
    print(f"recall             : {m['test_recall']:.3f}")
    print(f"F1                 : {m['test_f1']:.3f}")

    _bar("Confusion matrix (rows=actual, cols=predicted)")
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    print("              pred: no   pred: default")
    print(f"actual no       {tn:>4}        {fp:>4}")
    print(f"actual default  {fn:>4}        {tp:>4}")

    _bar("Top 8 features by gain")
    for row in trained.feature_importance.head(8).itertuples():
        print(f"  {row.importance:6.3f}  {row.label}")

    trained.save()
    _bar("Saved")
    print(f"model   -> {risk_model.config.RISK_MODEL_PATH}")
    print(f"metrics -> {risk_model.config.RISK_METRICS_PATH}")

    if m["test_auc"] > 0.95:
        print("\nWARNING: test AUC > 0.95 - inspect the feature pipeline for leakage.")


if __name__ == "__main__":
    main()
