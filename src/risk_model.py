"""Layer 1 - Tenant default-risk classifier.

Scores every tenant 0-100 on the probability they default in the next quarter,
using the temporally-split feature table from `features.py`.

Design choices worth defending in the technical report:

* **XGBoost, benchmarked honestly against a linear baseline.** Payment
  behaviour carries non-linear interactions (a heavy rent-to-income ratio only
  bites once the delay trend is also rising). Gradient-boosted trees capture
  that and give per-feature gain importances for free. We train a logistic
  regression alongside and report *both* their cross-validated and held-out AUC,
  so the XGBoost number is never quoted in a vacuum.
* **Calibrated probabilities, tuned threshold.** We do not use
  `scale_pos_weight`; instead we keep the raw (well-calibrated) probabilities -
  which become the 0-100 risk score - and choose the default/no-default
  operating threshold on *out-of-fold training* predictions by maximising F1.
  The test set never touches threshold selection.
* **Honest evaluation.** 20% is held out and never seen in training or
  threshold tuning; 5-fold stratified CV gives the AUC a mean and a std. On
  synthetic data we also report the *oracle* AUC (how well the true hidden
  reliability trait predicts the label) as the Bayes-optimal ceiling - the
  model lands below it, confirming the pipeline is not leaking the label.

A test AUC in the 0.85-0.90 band, below the oracle ceiling, is the target.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config, features

LABEL_COLUMN = "will_default"


@dataclass
class TrainedRiskModel:
    """A fitted classifier plus everything the app needs to use and explain it."""

    model: XGBClassifier
    columns: list[str]
    metrics: dict
    threshold: float = 0.5
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)

    # -- inference ---------------------------------------------------------

    def score(self, table: pd.DataFrame) -> pd.DataFrame:
        """Return tenant_id, calibrated probability, 0-100 score, band, flag."""
        matrix = features.align_matrix(features.to_matrix(table), self.columns)
        proba = self.model.predict_proba(matrix)[:, 1]
        scores = np.rint(proba * 100).astype(int)

        out = pd.DataFrame(
            {
                "tenant_id": table["tenant_id"].to_numpy(),
                "default_probability": proba,
                "risk_score": scores,
            }
        )
        out["risk_category"] = out["risk_score"].map(config.risk_category)
        out["flagged_default"] = (proba >= self.threshold).astype(int)
        return out

    # -- persistence -------------------------------------------------------

    def save(self) -> None:
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "columns": self.columns, "threshold": self.threshold},
            config.RISK_MODEL_PATH,
        )
        payload = {
            "metrics": self.metrics,
            "threshold": self.threshold,
            "feature_importance": self.feature_importance.to_dict(orient="records"),
        }
        config.RISK_METRICS_PATH.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls) -> "TrainedRiskModel":
        bundle = joblib.load(config.RISK_MODEL_PATH)
        payload = json.loads(config.RISK_METRICS_PATH.read_text())
        importance = pd.DataFrame(payload.get("feature_importance", []))
        return cls(
            model=bundle["model"],
            columns=bundle["columns"],
            metrics=payload.get("metrics", {}),
            threshold=bundle.get("threshold", 0.5),
            feature_importance=importance,
        )


def _baseline(X_tr, y_tr, X_te, y_te, cv, seed) -> dict:
    """Cross-validated and held-out AUC for a logistic-regression baseline."""
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    cv_auc = cross_val_score(pipe, X_tr, y_tr, scoring="roc_auc", cv=cv)
    pipe.fit(X_tr, y_tr)
    test_auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    return {
        "cv_auc_mean": round(float(cv_auc.mean()), 4),
        "test_auc": round(float(test_auc), 4),
    }


def _build_classifier(seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2.0,
        gamma=0.2,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )


def _oracle_auc(table: pd.DataFrame, tenants: pd.DataFrame, y: pd.Series) -> float | None:
    """Bayes-optimal ceiling: AUC of the true latent reliability vs the label.

    Only available on synthetic data (production has no hidden ground truth).
    """
    if "latent_reliability" not in tenants.columns:
        return None
    merged = table[["tenant_id"]].merge(
        tenants[["tenant_id", "latent_reliability"]], on="tenant_id", how="left"
    )
    return round(float(roc_auc_score(y, -merged["latent_reliability"])), 4)


def _best_f1_threshold(y_true, proba) -> float:
    """Threshold on the PR curve that maximises F1."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thr[int(np.argmax(f1[:-1]))])


def train(
    tenants: pd.DataFrame | None = None,
    payments: pd.DataFrame | None = None,
    *,
    seed: int = config.RANDOM_SEED,
) -> TrainedRiskModel:
    """Train the risk classifier end to end and return the fitted bundle."""
    from . import data_generator

    if tenants is None or payments is None:
        portfolio = data_generator.load()
        tenants, payments = portfolio.tenants, portfolio.payments

    table = features.build(tenants, payments)
    X = features.to_matrix(table)
    y = table[LABEL_COLUMN].astype(int)
    columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    baseline = _baseline(X_train, y_train, X_test, y_test, cv, seed)

    model = _build_classifier(seed)
    cv_auc = cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=cv)

    # Out-of-fold train predictions choose the operating threshold (no leakage).
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = _best_f1_threshold(y_train, oof)

    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    cm = confusion_matrix(y_test, pred_test)
    test_auc = float(roc_auc_score(y_test, proba_test))

    metrics = {
        "n_tenants": int(len(table)),
        "n_features": len(columns),
        "default_rate": round(float(y.mean()), 4),
        "test_size": int(len(y_test)),
        "operating_threshold": round(threshold, 4),
        "oracle_auc": _oracle_auc(table, tenants, y),
        "baseline_logreg_cv_auc": baseline["cv_auc_mean"],
        "baseline_logreg_test_auc": baseline["test_auc"],
        "cv_auc_mean": round(float(cv_auc.mean()), 4),
        "cv_auc_std": round(float(cv_auc.std()), 4),
        "test_auc": round(test_auc, 4),
        "test_average_precision": round(float(average_precision_score(y_test, proba_test)), 4),
        "test_precision": round(float(precision_score(y_test, pred_test, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, pred_test, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_test, pred_test, zero_division=0)), 4),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()},
    }

    importance = (
        pd.DataFrame({"feature": columns, "importance": model.feature_importances_})
        .assign(label=lambda d: d["feature"].map(features.pretty))
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return TrainedRiskModel(
        model=model,
        columns=columns,
        metrics=metrics,
        threshold=threshold,
        feature_importance=importance,
    )
