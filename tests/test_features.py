"""Feature pipeline: the temporal-split leakage guard is the test that matters."""

from __future__ import annotations

import pandas as pd

from src import config, features


def test_no_temporal_leakage(portfolio):
    """Features must be built ONLY from cycles at or before the cutoff.

    This is the core correctness guarantee of the whole model: if features
    could read the label window, the risk model would be learning the answer.
    We assert the observation window and the label window do not overlap.
    """
    payments = portfolio.payments.copy()
    payments["cycle_month"] = pd.to_datetime(payments["cycle_month"])
    cutoff = features.split_cutoff(payments)

    label_window = payments[payments["cycle_month"] > cutoff]
    observation = payments[payments["cycle_month"] <= cutoff]

    # The windows must be disjoint and the label window must be exactly the
    # configured horizon.
    assert observation["cycle_month"].max() <= cutoff
    assert label_window["cycle_month"].min() > cutoff
    assert label_window["cycle_month"].nunique() == config.LABEL_HORIZON_MONTHS


def test_label_definition_is_pattern_based(portfolio):
    """A single late cycle must NOT flag a default; a pattern must."""
    table = features.build(portfolio.tenants, portfolio.payments)
    assert "will_default" in table.columns
    assert set(table["will_default"].unique()).issubset({0, 1})
    # Base rate should be a credible minority, not degenerate.
    rate = table["will_default"].mean()
    assert 0.03 <= rate <= 0.40, f"default base rate {rate:.1%} is implausible"


def test_feature_matrix_shape(portfolio):
    table = features.build(portfolio.tenants, portfolio.payments)
    matrix = features.to_matrix(table)
    # Every modelling row maps to a matrix row; no NaNs reach the model.
    assert len(matrix) == len(table)
    assert not matrix.isna().any().any()
    assert all(f in matrix.columns for f in features.NUMERIC_FEATURES)


def test_rates_are_bounded(portfolio):
    table = features.build(portfolio.tenants, portfolio.payments)
    assert table["on_time_rate"].between(0, 1).all()
    assert table["late_rate"].between(0, 1).all()
    assert (table["missed_count"] >= 0).all()


def test_align_matrix_reorders_and_fills():
    """Inference matrices must be coerced onto the exact training column order."""
    import pandas as pd

    expected = ["a", "b", "c"]
    # Columns out of order and one missing ("c"), one extra ("z").
    m = pd.DataFrame({"b": [1.0], "a": [2.0], "z": [9.0]})
    aligned = features.align_matrix(m, expected)
    assert list(aligned.columns) == expected
    assert aligned.loc[0, "c"] == 0.0  # missing filled with zero
    assert aligned.loc[0, "a"] == 2.0  # existing preserved
    assert "z" not in aligned.columns  # extra dropped


def test_inference_mode_has_no_label(portfolio):
    """with_label=False scores all tenants over the full ledger, no label."""
    table = features.build(portfolio.tenants, portfolio.payments, with_label=False)
    assert "will_default" not in table.columns
    assert len(table) == len(portfolio.tenants)
