"""TenantEase AI Intelligence Engine — Streamlit dashboard.

Four tabs over one synthetic portfolio:

    1. Tenant Risk       - XGBoost risk scores, bands, sortable table, drill-down
    2. Revenue Forecast  - Prophet 3-month forecast, confidence band, decomposition
    3. AI Recommendations- LLM (or rule-based) action per tenant, provider badge
    4. Model Performance - ROC, confusion matrix, feature importance, baselines

The data is synthetic (see the sidebar note): Cynbit's production data is not
public, so the app runs on a generated portfolio with realistic behaviour.
"""

from __future__ import annotations

import streamlit as st

from src import config, data_generator, features, forecasting, recommendations, risk_model, ui

st.set_page_config(
    page_title="TenantEase AI Intelligence Engine",
    page_icon="🏠",
    layout="wide",
)


# --------------------------------------------------------------------------
# Cached loaders - data and model load once, not on every widget interaction.
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_portfolio():
    return data_generator.load()


@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return risk_model.TrainedRiskModel.load()
    except Exception:
        return risk_model.train()


@st.cache_data(show_spinner=False)
def scored_tenants():
    portfolio = load_portfolio()
    model = load_model()
    table = features.build(portfolio.tenants, portfolio.payments, with_label=False)
    scored = model.score(table)
    return table.merge(scored, on="tenant_id")


@st.cache_data(show_spinner="Forecasting revenue…")
def revenue_forecast():
    return forecasting.forecast(load_portfolio().monthly_revenue)


@st.cache_resource(show_spinner=False)
def rec_engine():
    return recommendations.RecommendationEngine()


def inr(value: float) -> str:
    return f"₹{value:,.0f}"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

portfolio = load_portfolio()
model = load_model()
enriched = scored_tenants()

with st.sidebar:
    st.title("🏠 TenantEase AI")
    st.caption("Predictive intelligence layer for Cynbit's TenantEase")
    st.divider()
    st.metric("Tenants scored", len(enriched))
    st.metric("Buildings", portfolio.tenants["building"].nunique())
    st.metric("Flagged for action", int(enriched["flagged_default"].sum()))
    st.divider()
    st.info(
        "**Synthetic data.** Cynbit's production data is not public, so this "
        "demo runs on a generated portfolio with realistic Indian rental "
        "patterns (520 tenants, 24 months).",
        icon="ℹ️",
    )


st.title("TenantEase AI Intelligence Engine")
st.caption(
    "Three ML layers on top of TenantEase: default-risk scoring (XGBoost), "
    "revenue forecasting (Prophet), and smart recommendations (LLM)."
)

tab_risk, tab_forecast, tab_reco, tab_model = st.tabs(
    ["🎯 Tenant Risk", "📈 Revenue Forecast", "💡 AI Recommendations", "🔬 Model Performance"]
)


# --------------------------------------------------------------------------
# Tab 1 - Tenant Risk
# --------------------------------------------------------------------------

with tab_risk:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Low", int((enriched["risk_category"] == "Low").sum()))
    c2.metric("Medium", int((enriched["risk_category"] == "Medium").sum()))
    c3.metric("High", int((enriched["risk_category"] == "High").sum()))
    c4.metric("Critical", int((enriched["risk_category"] == "Critical").sum()))

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(ui.risk_distribution(enriched), use_container_width=True)
    with right:
        st.plotly_chart(ui.band_breakdown(enriched), use_container_width=True)

    st.subheader("Tenant risk register")
    bands = st.multiselect(
        "Filter by risk band", list(config.RISK_COLORS), default=["High", "Critical"]
    )
    view = enriched[enriched["risk_category"].isin(bands)] if bands else enriched
    view = view.sort_values("risk_score", ascending=False)

    table_cols = {
        "tenant_name": "Tenant", "building": "Building", "unit_no": "Unit",
        "property_type": "Type", "monthly_rent": "Rent", "risk_score": "Risk",
        "risk_category": "Band", "on_time_rate": "On-time", "late_count_6m": "Late (6m)",
    }
    display = view[list(table_cols)].rename(columns=table_cols)
    st.dataframe(
        display, use_container_width=True, hide_index=True,
        column_config={
            "Rent": st.column_config.NumberColumn(format="₹%d"),
            "Risk": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
            "On-time": st.column_config.NumberColumn(format="%.0f%%", help="Share of on-time payments"),
        },
    )

    st.subheader("Tenant drill-down")
    names = view["tenant_name"].tolist()
    if names:
        picked = st.selectbox("Select a tenant", names)
        row = view[view["tenant_name"] == picked].iloc[0]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Risk score", f"{int(row['risk_score'])}/100", row["risk_category"])
        d2.metric("On-time rate", f"{row['on_time_rate'] * 100:.0f}%")
        d3.metric("Late (last 6)", int(row["late_count_6m"]))
        d4.metric("Avg days late", f"{row['avg_days_past_due']:.1f}")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Monthly rent", inr(row["monthly_rent"]))
        e2.metric("Deposit (months)", f"{row['deposit_months']:.0f}")
        e3.metric("Tenure (months)", int(row["tenure_months"]))
        e4.metric("Missed cycles", int(row["missed_count"]))


# --------------------------------------------------------------------------
# Tab 2 - Revenue Forecast
# --------------------------------------------------------------------------

with tab_forecast:
    result = revenue_forecast()

    m1, m2, m3 = st.columns(3)
    next_month = result.forecast.iloc[0]
    m1.metric("Next-month forecast", inr(next_month["yhat"]))
    m2.metric(
        "Backtest MAPE",
        f"{result.backtest_mape:.2f}%" if result.backtest_mape is not None else "n/a",
        help="Mean absolute % error on a held-out 3-month tail",
    )
    m3.metric("Forecast engine", result.engine.title())

    st.plotly_chart(
        ui.revenue_forecast(result.history, result.forecast), use_container_width=True
    )

    st.subheader("3-month projection")
    proj = result.forecast.copy()
    proj["month"] = proj["month"].dt.strftime("%b %Y")
    proj = proj.rename(columns={
        "month": "Month", "yhat": "Forecast", "yhat_lower": "Low (80%)", "yhat_upper": "High (80%)"
    })
    st.dataframe(
        proj, use_container_width=True, hide_index=True,
        column_config={c: st.column_config.NumberColumn(format="₹%d")
                       for c in ["Forecast", "Low (80%)", "High (80%)"]},
    )

    st.plotly_chart(ui.decomposition(result.components), use_container_width=True)
    if result.engine == "holt-winters":
        st.caption(
            "Prophet's backend was unavailable, so the forecast used the "
            "Holt-Winters fallback (identical output contract)."
        )


# --------------------------------------------------------------------------
# Tab 3 - AI Recommendations
# --------------------------------------------------------------------------

with tab_reco:
    engine = rec_engine()
    live = engine.client.is_live
    badge = "🟢 Live LLM" if live else "⚪ Rule-based fallback"
    st.markdown(f"**Recommendation source:** {badge} — `{engine.source_label}`")
    if not live:
        st.caption(
            "No API key detected. Recommendations are generated by the "
            "deterministic rule-based engine. Set ANTHROPIC_API_KEY or "
            "OPENAI_API_KEY to enable live LLM generation."
        )

    top_n = st.slider("How many top-risk tenants to advise", 3, 15, 5)
    cohort = enriched.sort_values("risk_score", ascending=False).head(top_n)

    urgency_color = config.RISK_COLORS
    for _, row in cohort.iterrows():
        ctx = row.to_dict()
        rec = engine.for_tenant(ctx)
        color = urgency_color.get(rec.urgency, ui.INK_MUTED)
        with st.container(border=True):
            head, score = st.columns([4, 1])
            head.markdown(
                f"**{row['tenant_name']}** · {row['building']} {row['unit_no']} "
                f"· {row['property_type']} · {inr(row['monthly_rent'])}/mo"
            )
            score.markdown(
                f"<div style='text-align:right'><span style='color:{color};"
                f"font-weight:700'>{rec.urgency}</span><br>"
                f"<span style='color:{ui.INK_MUTED};font-size:0.8em'>risk "
                f"{int(row['risk_score'])}/100</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Action:** {rec.action}")
            st.markdown(f"**Why:** {rec.reason}")
            if rec.suggested_message:
                with st.expander("Suggested message to tenant"):
                    st.write(rec.suggested_message)


# --------------------------------------------------------------------------
# Tab 4 - Model Performance
# --------------------------------------------------------------------------

with tab_model:
    m = model.metrics
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Test AUC", f"{m.get('test_auc', 0):.3f}")
    k2.metric(
        "CV AUC", f"{m.get('cv_auc_mean', 0):.3f}",
        help=f"± {m.get('cv_auc_std', 0):.3f} across 5 folds",
    )
    k3.metric(
        "vs baseline", f"{m.get('test_auc', 0) - m.get('baseline_logreg_test_auc', 0):+.3f}",
        help=f"Logistic baseline test AUC {m.get('baseline_logreg_test_auc', 0):.3f}",
    )
    oracle = m.get("oracle_auc")
    k4.metric(
        "Oracle ceiling", f"{oracle:.3f}" if oracle else "n/a",
        help="Bayes-optimal AUC from the true latent trait; the model sits below it (no leakage)",
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(ui.roc_curve(m), use_container_width=True)
    with right:
        st.plotly_chart(ui.confusion_matrix(m), use_container_width=True)

    st.plotly_chart(ui.feature_importance(model.feature_importance), use_container_width=True)

    with st.expander("How the model is evaluated"):
        oracle_txt = f"{oracle:.3f}" if oracle else "n/a"
        st.markdown(
            f"""
- **Default label**: a tenant is a default if they have
  **{config.DEFAULT_MIN_BAD_CYCLES}+ delinquent cycles** (missed or
  {config.DELINQUENCY_DPD}+ days late) in the forward
  {config.LABEL_HORIZON_MONTHS}-month window — a pattern, not one unlucky month.
- **No leakage**: features are built strictly from the observation window;
  the test AUC ({m.get('test_auc', 0):.3f}) sits below the oracle ceiling
  ({oracle_txt}).
- **Operating threshold** {m.get('operating_threshold', 0.5):.2f} was tuned on
  out-of-fold training predictions (max F1), never on the test set.
- **Recall {m.get('test_recall', 0):.2f}** at that threshold — catching real
  defaulters matters more than the occasional false flag.
"""
        )
