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

SURFACE = "#1a1a19"
INK = "#e8e8e6"
INK_MUTED = "#9a9a97"
GRID = "#2e2e2c"
ACCENT = "#5B8DEF"          # primary single-metric hue (revenue, bars, ROC)
ACCENT_SOFT = "rgba(91,141,239,0.18)"  # confidence band / fills
RISK_COLORS = config.RISK_COLORS


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
