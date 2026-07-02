from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def timeseries_chart(df: pd.DataFrame, x: str = "date", y: str = "value", color: str | None = None, title: str | None = None) -> go.Figure:
    fig = px.line(df, x=x, y=y, color=color, title=title)
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def correlation_heatmap(frame: pd.DataFrame, columns: list[str]) -> go.Figure:
    correlation = frame[columns].corr(numeric_only=True)
    fig = px.imshow(correlation, color_continuous_scale="viridis", zmin=-1, zmax=1, title="Correlation heatmap")
    return fig


def actual_vs_predicted_chart(y_true, y_pred, dates=None) -> go.Figure:
    if dates is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=y_true, mode="lines+markers", name="Actual"))
        fig.add_trace(go.Scatter(x=dates, y=y_pred, mode="lines+markers", name="Predicted"))
        fig.update_layout(title="Actual vs predicted", xaxis_title="Date", yaxis_title="Value")
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="Predictions"))
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect fit", line=dict(dash="dash")))
    fig.update_layout(title="Actual vs predicted", xaxis_title="Actual", yaxis_title="Predicted")
    return fig


def forecast_fan_chart(predictions_df: pd.DataFrame) -> go.Figure:
    historical = predictions_df[~predictions_df["is_forecast"]]
    forecast = predictions_df[predictions_df["is_forecast"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical["period"], y=historical["actual"], mode="lines", name="Historical actual"))

    if not forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecast["period"], forecast["period"][::-1]]),
                y=pd.concat([forecast["upper"], forecast["lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(99, 110, 250, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="Confidence interval",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["period"],
                y=forecast["prediction"],
                mode="lines",
                name="Forecast",
                line=dict(dash="dash"),
            )
        )
        fig.add_vline(x=historical["period"].max(), line_dash="dot", annotation_text="Last observed")

    fig.update_layout(title="Forecast", xaxis_title="Period", yaxis_title="Value")
    fig.update_xaxes(rangeslider_visible=True)
    return fig
