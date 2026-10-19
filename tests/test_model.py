"""Risk model: score bounds, band mapping, metric sanity, and inference contract."""

from __future__ import annotations

import pytest

from src import config, features, risk_model


@pytest.fixture(scope="module")
def trained(request):
    portfolio = request.getfixturevalue("portfolio")
    return risk_model.train(portfolio.tenants, portfolio.payments)


def test_risk_band_boundaries():
    """Bands must map correctly, including at the exact thresholds."""
    assert config.risk_category(0) == "Low"
    assert config.risk_category(24) == "Low"
    assert config.risk_category(25) == "Medium"
    assert config.risk_category(49) == "Medium"
    assert config.risk_category(50) == "High"
    assert config.risk_category(74) == "High"
    assert config.risk_category(75) == "Critical"
    assert config.risk_category(100) == "Critical"


def test_scores_are_bounded(trained, portfolio):
    table = features.build(portfolio.tenants, portfolio.payments, with_label=False)
    scored = trained.score(table)
    assert scored["risk_score"].between(0, 100).all()
    assert scored["default_probability"].between(0, 1).all()
    assert scored["risk_category"].isin(config.RISK_COLORS).all()
    assert len(scored) == len(portfolio.tenants)


def test_model_beats_baseline(trained):
    """XGBoost should at least match the logistic baseline on test AUC."""
    m = trained.metrics
    assert m["test_auc"] >= m["baseline_logreg_test_auc"] - 0.01


def test_no_leakage_auc_below_oracle(trained):
    """A test AUC above the oracle ceiling would mean the pipeline is leaking."""
    m = trained.metrics
    assert m["test_auc"] < 0.95, "suspiciously high AUC - check for leakage"
    if m.get("oracle_auc"):
        assert m["test_auc"] <= m["oracle_auc"] + 0.02


def test_threshold_is_valid(trained):
    assert 0.0 < trained.metrics["operating_threshold"] < 1.0


def test_save_and_load_roundtrip(trained, tmp_path, monkeypatch):
    """Persisted model must reload with identical columns and threshold."""
    monkeypatch.setattr(config, "RISK_MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(config, "RISK_METRICS_PATH", tmp_path / "m.json")
    trained.save()
    reloaded = risk_model.TrainedRiskModel.load()
    assert reloaded.columns == trained.columns
    assert reloaded.threshold == pytest.approx(trained.threshold)
    assert reloaded.metrics["test_auc"] == trained.metrics["test_auc"]
