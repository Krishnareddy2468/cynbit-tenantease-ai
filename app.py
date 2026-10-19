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

import os

import streamlit as st

# set_page_config must be the very first Streamlit command.
st.set_page_config(
    page_title="TenantEase AI Intelligence Engine",
    page_icon=":material/apartment:",
    layout="wide",
)

# On Streamlit Community Cloud, API keys are set as Secrets (not a .env file).
# Bridge them into the environment so the provider-agnostic LLMClient — which
# reads os.getenv — finds them exactly as it does locally. Harmless locally
# (st.secrets is empty) and runs before the LLM client is constructed.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass

from src import config, data_generator, features, forecasting, recommendations, risk_model, ui  # noqa: E402


# Chrome tokens per theme. `set_theme()` in src/ui.py handles the chart palette;
# these drive the Streamlit surfaces (page, sidebar, cards, text).
_UI_TOKENS = {
    "dark": {
        "bg": "#141414", "surface": "#1c1c1b", "border": "#2c2c2a",
        "ink": "#ECECEA", "muted": "#9a9a97", "accent": "#5B8DEF",
        "label": "#8a8a87", "value": "#F2F2F0", "link": "#7FA8F5",
    },
    "light": {
        "bg": "#F6F8FA", "surface": "#FFFFFF", "border": "#E2E5EA",
        "ink": "#1A1D23", "muted": "#687280", "accent": "#3A6FD0",
        "label": "#687280", "value": "#111827", "link": "#2C5FC4",
    },
}


def _inject_css(mode: str) -> None:
    """Design layer, themed: typography, card metrics, muted chrome, polished tabs."""
    t = _UI_TOKENS.get(mode, _UI_TOKENS["dark"])
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"], .stMarkdown, .stMetric, button, input, textarea, select {{
            font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
        }}

        /* Base surfaces */
        .stApp, [data-testid="stAppViewContainer"] {{ background: {t['bg']}; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{ background: {t['surface']}; border-right: 1px solid {t['border']}; }}
        .stApp, .stMarkdown, p, span, label, li {{ color: {t['ink']}; }}

        /* Remove stock chrome */
        #MainMenu, footer, [data-testid="stDecoration"] {{ display: none !important; }}
        [data-testid="stToolbar"] {{ right: 1rem; }}

        /* Tighten the main canvas */
        .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }}

        /* Metrics as cards */
        [data-testid="stMetric"] {{
            background: {t['surface']};
            border: 1px solid {t['border']};
            border-radius: 12px;
            padding: 14px 16px;
        }}
        [data-testid="stMetricLabel"] p {{
            font-size: 0.72rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {t['label']} !important;
        }}
        [data-testid="stMetricValue"] {{
            font-size: 1.7rem !important;
            font-weight: 700 !important;
            color: {t['value']} !important;
        }}
        [data-testid="stMetricDelta"] {{ font-size: 0.8rem !important; }}

        /* Tabs: roomier, accent underline on the active tab */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {t['border']}; }}
        [data-testid="stTabs"] [data-baseweb="tab"] {{
            height: 44px; padding: 0 18px; font-weight: 600; font-size: 0.92rem; color: {t['muted']};
        }}
        [data-testid="stTabs"] [aria-selected="true"] {{ color: {t['accent']} !important; }}

        /* Section subheaders + links */
        h1, h2, h3 {{ color: {t['ink']}; letter-spacing: -0.01em; }}
        a {{ color: {t['link']} !important; }}

        /* Input widgets follow the theme (multiselect, selectbox) */
        [data-baseweb="select"] > div {{
            background: {t['surface']} !important;
            border-color: {t['border']} !important;
        }}
        [data-testid="stExpander"] {{
            background: {t['surface']};
            border: 1px solid {t['border']};
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _header(mode: str) -> None:
    """A branded header band instead of the default st.title."""
    t = _UI_TOKENS.get(mode, _UI_TOKENS["dark"])
    st.markdown(
        f"""
        <div style="padding: 6px 0 2px 0;">
          <div style="display:flex; align-items:center; gap:12px;">
            <div style="width:38px;height:38px;border-radius:10px;
                        background:linear-gradient(135deg,#5B8DEF,#2BB6A8);
                        display:flex;align-items:center;justify-content:center;
                        font-size:15px;font-weight:700;color:#fff;">TE</div>
            <div>
              <div style="font-size:1.55rem;font-weight:700;color:{t['value']};
                          line-height:1.1;">TenantEase AI Intelligence Engine</div>
              <div style="font-size:0.9rem;color:{t['muted']};margin-top:2px;">
                Predictive AI layer for Cynbit's TenantEase · risk scoring ·
                revenue forecasting · smart recommendations
              </div>
            </div>
          </div>
        </div>
        <div style="height:1px;background:{t['border']};margin:16px 0 20px 0;"></div>
        """,
        unsafe_allow_html=True,
    )


# Resolve the active theme once per run, before any CSS or chart is rendered.
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"
THEME_MODE = st.session_state.theme_mode
ui.set_theme(THEME_MODE)
_inject_css(THEME_MODE)


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
        # First run (no artifact yet): train once and persist so later launches
        # load instantly. This is what makes a single command "just work".
        with st.spinner("Training the risk model (first run only)…"):
            trained = risk_model.train()
            trained.save()
        return trained


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
    _title_color = _UI_TOKENS[THEME_MODE]["value"]
    _sub_color = _UI_TOKENS[THEME_MODE]["muted"]
    st.markdown(
        f"<div style='font-size:1.15rem;font-weight:700;color:{_title_color};'>"
        "TenantEase AI</div>"
        f"<div style='font-size:0.8rem;color:{_sub_color};margin:2px 0 4px;'>"
        "for Cynbit Technologies</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.radio(
        "Appearance",
        options=["dark", "light"],
        format_func=lambda m: "Dark" if m == "dark" else "Light",
        key="theme_mode",
        horizontal=True,
    )
    st.divider()
    st.metric("Tenants scored", len(enriched))
    st.metric("Buildings", portfolio.tenants["building"].nunique())
    st.metric("Flagged for action", int(enriched["flagged_default"].sum()))
    st.divider()
    st.info(
        "**Synthetic data.** Cynbit's production data is not public, so this "
        "demo runs on a generated portfolio with realistic Indian rental "
        "patterns (520 tenants, 24 months).",
    )


_header(THEME_MODE)

tab_risk, tab_forecast, tab_reco, tab_model = st.tabs(
    ["Tenant Risk", "Revenue Forecast", "AI Recommendations", "Model Performance"]
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
    st.markdown(ui.risk_register_html(view), unsafe_allow_html=True)

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
    proj[["yhat", "yhat_lower", "yhat_upper"]] = proj[
        ["yhat", "yhat_lower", "yhat_upper"]
    ].round().astype(int)
    proj = proj.rename(columns={
        "month": "Month", "yhat": "Forecast", "yhat_lower": "Low (80%)", "yhat_upper": "High (80%)"
    })
    st.markdown(
        ui.plain_table_html(proj, money_cols=("Forecast", "Low (80%)", "High (80%)")),
        unsafe_allow_html=True,
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
    top_n = st.slider("How many top-risk tenants to advise", 3, 15, 5)
    cohort = enriched.sort_values("risk_score", ascending=False).head(top_n)

    # Generate first, so the badge can reflect the *actual* source.
    recommendations_out = [(row, engine.for_tenant(row.to_dict())) for _, row in cohort.iterrows()]
    live_sources = {"anthropic", "openai", "gemini"}
    used_live = any(rec.source in live_sources for _, rec in recommendations_out)
    configured = engine.client.provider

    if used_live:
        dot_color, badge_text = "#2BB673", f"Live LLM — {engine.source_label}"
        caption = None
    elif engine.client.is_live:
        # A provider is configured but every call fell back (e.g. a transient
        # outage); be honest rather than claiming a live result.
        dot_color = "#E8B339"
        badge_text = f"{configured} unavailable — showing rule-based fallback"
        caption = "The configured LLM could not be reached; recommendations use the deterministic engine."
    else:
        dot_color, badge_text = ui.INK_MUTED, "Rule-based fallback"
        caption = (
            "No API key detected. Set GEMINI_API_KEY (or ANTHROPIC_API_KEY / "
            "OPENAI_API_KEY) in your .env to enable live LLM generation."
        )

    st.markdown(
        f"<span style='display:inline-block;width:9px;height:9px;border-radius:50%;"
        f"background:{dot_color};margin-right:7px;vertical-align:middle;'></span>"
        f"<b>Recommendation source:</b> {badge_text}",
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)

    urgency_color = config.RISK_COLORS
    for row, rec in recommendations_out:
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
