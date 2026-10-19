"""Plotly chart builders and a shared dark theme for the dashboard.

Charting follows a few rules kept consistent across every tab:

* **One axis per chart.** No dual-axis plots - two measures of different scale
  get two charts, never two y-scales.
* **Colour by job, not decoration.** Revenue is a single metric over time, so
  history and forecast share one hue (solid vs dashed) rather than reading as
  two categories. Risk bands use the ordered severity palette from config, and
  every band is shown with its text label, so colour is never the only signal.
* **Recessive grid and axes, thin marks, legend whenever two series appear.**
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import config

# -- palette ---------------------------------------------------------------

RISK_COLORS = config.RISK_COLORS

# Two palettes; `set_theme()` swaps the module-level tokens the chart builders
# read at call time, so toggling the app theme re-colours the charts too.
THEMES = {
    "dark": {
        "SURFACE": "#1c1c1b",
        "INK": "#ECECEA",
        "INK_MUTED": "#9a9a97",
        "GRID": "#323230",
        "ACCENT": "#5B8DEF",
        "ACCENT_SOFT": "rgba(91,141,239,0.18)",
    },
    "light": {
        "SURFACE": "#FFFFFF",
        "INK": "#1A1D23",
        "INK_MUTED": "#687280",
        "GRID": "#E6E8EC",
        "ACCENT": "#3A6FD0",
        "ACCENT_SOFT": "rgba(58,111,208,0.14)",
    },
}

# Current palette (defaults to dark; overwritten by set_theme()).
SURFACE = INK = INK_MUTED = GRID = ACCENT = ACCENT_SOFT = ""


def set_theme(mode: str = "dark") -> None:
    """Point the module-level colour tokens at the chosen palette."""
    global SURFACE, INK, INK_MUTED, GRID, ACCENT, ACCENT_SOFT
    t = THEMES.get(mode, THEMES["dark"])
    SURFACE = t["SURFACE"]
    INK = t["INK"]
    INK_MUTED = t["INK_MUTED"]
    GRID = t["GRID"]
    ACCENT = t["ACCENT"]
    ACCENT_SOFT = t["ACCENT_SOFT"]


set_theme("dark")


def _base_layout(title: str, height: int = 360) -> dict:
    """Shared layout: dark surface, recessive axes, legend top-right."""
    return dict(
        title=dict(text=title, font=dict(size=15, color=INK)),
        height=height,
        margin=dict(l=60, r=24, t=48, b=48),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=INK, size=12),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, color=INK_MUTED),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, color=INK_MUTED),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color=INK_MUTED, size=11),
        ),
        hoverlabel=dict(bgcolor="#2a2a28", font=dict(color=INK, size=12)),
    )


# -- Tab 1: tenant risk ----------------------------------------------------

def risk_distribution(scored: pd.DataFrame) -> go.Figure:
    """Histogram of 0-100 risk scores, coloured by band with a shared legend."""
    fig = go.Figure()
    for band, color in RISK_COLORS.items():
        sub = scored[scored["risk_category"] == band]
        if sub.empty:
            continue
        fig.add_trace(
            go.Histogram(
                x=sub["risk_score"], name=band, marker_color=color,
                xbins=dict(start=0, end=100, size=5), marker_line_width=0,
                hovertemplate=f"{band}<br>score %{{x}}<br>%{{y}} tenants<extra></extra>",
            )
        )
    fig.update_layout(
        **_base_layout("Tenant risk score distribution"),
        barmode="stack", bargap=0.04,
    )
    fig.update_xaxes(title_text="Risk score (0-100)")
    fig.update_yaxes(title_text="Tenants")
    return fig


def band_breakdown(scored: pd.DataFrame) -> go.Figure:
    """Count of tenants per risk band, bars in severity colours, labelled."""
    order = list(RISK_COLORS)
    counts = scored["risk_category"].value_counts().reindex(order).fillna(0).astype(int)
    fig = go.Figure(
        go.Bar(
            x=order, y=counts.values,
            marker_color=[RISK_COLORS[b] for b in order],
            text=counts.values, textposition="outside",
            marker_line_width=0,
            hovertemplate="%{x}<br>%{y} tenants<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout("Tenants by risk band", height=320), showlegend=False)
    fig.update_yaxes(title_text="Tenants")
    return fig


# -- Tab 2: revenue forecast -----------------------------------------------

def revenue_forecast(history: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    """One metric over time: solid history, dashed forecast, shaded 80% band."""
    fig = go.Figure()

    # Confidence band (drawn first so lines sit on top).
    fig.add_trace(
        go.Scatter(
            x=list(forecast["month"]) + list(forecast["month"][::-1]),
            y=list(forecast["yhat_upper"]) + list(forecast["yhat_lower"][::-1]),
            fill="toself", fillcolor=ACCENT_SOFT, line=dict(width=0),
            name="80% interval", hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history["month"], y=history["actual"], mode="lines",
            name="Collected (actual)", line=dict(color=ACCENT, width=2),
            hovertemplate="%{x|%b %Y}<br>INR %{y:,.0f}<extra></extra>",
        )
    )
    # Bridge the last actual into the forecast so the dashed line connects.
    bridge = pd.concat([history.tail(1).rename(columns={"actual": "yhat"})[["month", "yhat"]], forecast[["month", "yhat"]]])
    fig.add_trace(
        go.Scatter(
            x=bridge["month"], y=bridge["yhat"], mode="lines+markers",
            name="Forecast", line=dict(color=ACCENT, width=2, dash="dash"),
            marker=dict(size=7, color=ACCENT),
            hovertemplate="%{x|%b %Y}<br>forecast INR %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout("Monthly collected revenue — 3-month forecast", height=400))
    fig.update_yaxes(title_text="INR")
    return fig


def decomposition(components: pd.DataFrame) -> go.Figure:
    """Trend and seasonal components as two stacked single-metric panels."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
        subplot_titles=("Trend", "Seasonality (INR vs baseline)"),
    )
    fig.add_trace(
        go.Scatter(x=components["month"], y=components["trend"], mode="lines",
                   line=dict(color=ACCENT, width=2), name="Trend",
                   hovertemplate="%{x|%b %Y}<br>INR %{y:,.0f}<extra></extra>"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=components["month"], y=components["seasonal"], mode="lines",
                   line=dict(color=ACCENT, width=2), name="Seasonal",
                   hovertemplate="%{x|%b %Y}<br>INR %{y:,.0f}<extra></extra>"),
        row=2, col=1,
    )
    layout = _base_layout("Revenue decomposition", height=420)
    layout.pop("xaxis")
    layout.pop("yaxis")
    fig.update_layout(**layout, showlegend=False)
    fig.update_xaxes(gridcolor=GRID, color=INK_MUTED)
    fig.update_yaxes(gridcolor=GRID, color=INK_MUTED)
    for ann in fig.layout.annotations:
        ann.font.color = INK_MUTED
        ann.font.size = 12
    return fig


# -- Tab 4: model performance ----------------------------------------------

def roc_curve(metrics: dict) -> go.Figure:
    """ROC curve with the no-skill diagonal and the AUC annotated."""
    roc = metrics.get("roc_curve", {"fpr": [0, 1], "tpr": [0, 1]})
    auc = metrics.get("test_auc", 0)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="No skill",
                   line=dict(color=INK_MUTED, width=1, dash="dot"), hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                   name=f"XGBoost (AUC {auc:.3f})", line=dict(color=ACCENT, width=2.5),
                   hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>")
    )
    fig.update_layout(**_base_layout("ROC curve — held-out test set"))
    fig.update_xaxes(title_text="False positive rate", range=[0, 1])
    fig.update_yaxes(title_text="True positive rate", range=[0, 1])
    return fig


def confusion_matrix(metrics: dict) -> go.Figure:
    """Confusion matrix as a single-hue sequential heatmap with count labels."""
    (tn, fp), (fn, tp) = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    z = [[tn, fp], [fn, tp]]
    labels_x = ["Pred: no default", "Pred: default"]
    labels_y = ["Actual: no default", "Actual: default"]
    fig = go.Figure(
        go.Heatmap(
            z=z, x=labels_x, y=labels_y,
            colorscale=[[0, SURFACE], [1, ACCENT]], showscale=False,
            text=[[f"{tn}", f"{fp}"], [f"{fn}", f"{tp}"]],
            texttemplate="%{text}", textfont=dict(size=22, color=INK),
            hovertemplate="%{y}<br>%{x}<br>%{z} tenants<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout("Confusion matrix (tuned threshold)", height=340))
    fig.update_yaxes(autorange="reversed")
    return fig


def feature_importance(importance: pd.DataFrame, top: int = 10) -> go.Figure:
    """Horizontal bars, single hue, highest gain on top."""
    data = importance.head(top).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=data["importance"], y=data["label"], orientation="h",
            marker_color=ACCENT, marker_line_width=0,
            hovertemplate="%{y}<br>gain %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout(f"Top {top} features by gain", height=400), showlegend=False)
    fig.update_xaxes(title_text="Importance (gain)")
    return fig


# -- Themed HTML tables (so they follow the light/dark toggle, unlike the
#    glide-grid st.dataframe, which only reads Streamlit's static config theme) --

def _table_shell(head_html: str, body_html: str, max_height: int = 460) -> str:
    return f"""
    <div style="border:1px solid {GRID};border-radius:10px;overflow:hidden;
                max-height:{max_height}px;overflow-y:auto;">
      <table style="width:100%;border-collapse:collapse;font-size:0.86rem;
                    color:{INK};font-family:'Inter',sans-serif;">
        <thead style="position:sticky;top:0;">{head_html}</thead>
        <tbody>{body_html}</tbody>
      </table>
    </div>
    """


def risk_register_html(df) -> str:
    """Render the tenant risk register as a themed table with risk bars + pills."""
    cols = ["Tenant", "Building", "Unit", "Type", "Rent", "Risk", "Band", "On-time", "Late (6m)"]
    head = (
        "<tr style='background:" + SURFACE + ";'>"
        + "".join(
            f"<th style='text-align:left;padding:9px 12px;font-size:0.72rem;"
            f"text-transform:uppercase;letter-spacing:0.04em;color:{INK_MUTED};"
            f"border-bottom:1px solid {GRID};'>{c}</th>"
            for c in cols
        )
        + "</tr>"
    )

    rows = []
    for r in df.itertuples(index=False):
        band = r.risk_category
        color = RISK_COLORS.get(band, ACCENT)
        score = int(r.risk_score)
        bar = (
            f"<div style='display:flex;align-items:center;gap:10px;'>"
            f"<div style='width:84px;height:8px;border-radius:4px;"
            f"background:linear-gradient(90deg,{color} {score}%,{GRID} {score}%);'></div>"
            f"<span style='color:{INK};font-variant-numeric:tabular-nums;"
            f"min-width:20px;'>{score}</span></div>"
        )
        pill = (
            f"<span style='color:{color};font-weight:600;font-size:0.82rem;'>{band}</span>"
        )
        cells = [
            f"<span style='font-weight:500;'>{r.tenant_name}</span>",
            r.building,
            r.unit_no,
            r.property_type,
            f"₹{int(r.monthly_rent):,}",
            bar,
            pill,
            f"{r.on_time_rate * 100:.0f}%",
            str(int(r.late_count_6m)),
        ]
        tds = "".join(
            f"<td style='padding:8px 12px;border-bottom:1px solid {GRID};'>{c}</td>"
            for c in cells
        )
        rows.append(f"<tr>{tds}</tr>")

    return _table_shell(head, "".join(rows))


def plain_table_html(df, money_cols=()) -> str:
    """Render a small dataframe as a themed table (money columns get ₹ + commas)."""
    head = (
        "<tr style='background:" + SURFACE + ";'>"
        + "".join(
            f"<th style='text-align:left;padding:9px 12px;font-size:0.72rem;"
            f"text-transform:uppercase;letter-spacing:0.04em;color:{INK_MUTED};"
            f"border-bottom:1px solid {GRID};'>{c}</th>"
            for c in df.columns
        )
        + "</tr>"
    )
    rows = []
    for r in df.itertuples(index=False):
        cells = []
        for col, val in zip(df.columns, r):
            if col in money_cols:
                cells.append(f"₹{int(val):,}")
            else:
                cells.append(str(val))
        tds = "".join(
            f"<td style='padding:8px 12px;border-bottom:1px solid {GRID};'>{c}</td>"
            for c in cells
        )
        rows.append(f"<tr>{tds}</tr>")
    return _table_shell(head, "".join(rows), max_height=260)
