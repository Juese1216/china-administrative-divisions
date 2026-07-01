from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _trim_for_hover(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def build_map(
    panel: pd.DataFrame,
    output_html: str | Path,
    color_column: str,
    size_column: str,
    geojson_path: str | Path | None = None,
    geojson_id_property: str = "unit_id",
    map_style: str = "carto-positron",
    center_lat: float = 35.0,
    center_lon: float = 104.0,
    zoom: float = 3.2,
) -> None:
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    plot_df = panel.copy()
    plot_df["year"] = plot_df["year"].astype(str)
    hover_cols = _trim_for_hover(
        plot_df,
        [
            "unit_name",
            "level",
            "province",
            "city",
            "county",
            "population",
            "nightlight",
            "builtup_area",
            "vitality_index",
            "change_intensity_index",
            "change_type",
            "row_stage",
        ],
    )

    if geojson_path:
        geojson = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
        try:
            fig = px.choropleth_map(
                plot_df,
                geojson=geojson,
                locations="unit_id",
                featureidkey=f"properties.{geojson_id_property}",
                color=color_column,
                animation_frame="year",
                hover_data=hover_cols,
                map_style=map_style,
                center={"lat": center_lat, "lon": center_lon},
                zoom=zoom,
                opacity=0.72,
                color_continuous_scale="Turbo",
            )
        except AttributeError:
            fig = px.choropleth_mapbox(
                plot_df,
                geojson=geojson,
                locations="unit_id",
                featureidkey=f"properties.{geojson_id_property}",
                color=color_column,
                animation_frame="year",
                hover_data=hover_cols,
                mapbox_style=map_style,
                center={"lat": center_lat, "lon": center_lon},
                zoom=zoom,
                opacity=0.72,
                color_continuous_scale="Turbo",
            )
    else:
        if not {"lon", "lat"}.issubset(plot_df.columns):
            raise ValueError("No geojson was provided and panel lacks lon/lat point columns.")
        size_values = plot_df[size_column].fillna(plot_df[size_column].median())
        plot_df["_plot_size"] = size_values.clip(lower=1)
        try:
            fig = px.scatter_map(
                plot_df,
                lat="lat",
                lon="lon",
                color=color_column,
                size="_plot_size",
                animation_frame="year",
                hover_data=hover_cols,
                map_style=map_style,
                center={"lat": center_lat, "lon": center_lon},
                zoom=zoom,
                color_continuous_scale="Turbo",
                opacity=0.68,
            )
        except AttributeError:
            fig = px.scatter_mapbox(
                plot_df,
                lat="lat",
                lon="lon",
                color=color_column,
                size="_plot_size",
                animation_frame="year",
                hover_data=hover_cols,
                mapbox_style=map_style,
                center={"lat": center_lat, "lon": center_lon},
                zoom=zoom,
                color_continuous_scale="Turbo",
                opacity=0.68,
            )

    fig.update_layout(
        title="China Rural Time-Series Atlas: observed comparison and 2027-2035 forecast",
        margin={"r": 0, "t": 48, "l": 0, "b": 0},
        coloraxis_colorbar={"title": color_column},
    )
    fig.write_html(output_html, include_plotlyjs="cdn")


def build_dashboard(
    panel: pd.DataFrame,
    national_summary: pd.DataFrame,
    output_html: str | Path,
    metrics: list[str],
) -> None:
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "National aggregate metrics",
            "Mean indices",
            "Units by change type",
            "Data stage share",
        ],
    )

    for metric in metrics:
        col = f"{metric}_sum"
        if col in national_summary.columns:
            fig.add_trace(
                go.Scatter(
                    x=national_summary["year"],
                    y=national_summary[col],
                    mode="lines+markers",
                    name=col,
                ),
                row=1,
                col=1,
            )

    for col in ["vitality_index_mean", "change_intensity_index_mean"]:
        if col in national_summary.columns:
            fig.add_trace(
                go.Scatter(
                    x=national_summary["year"],
                    y=national_summary[col],
                    mode="lines+markers",
                    name=col,
                ),
                row=1,
                col=2,
            )

    change_counts = panel.groupby(["year", "change_type"]).size().reset_index(name="count")
    for change_type, group in change_counts.groupby("change_type"):
        fig.add_trace(
            go.Bar(x=group["year"], y=group["count"], name=f"type:{change_type}"),
            row=2,
            col=1,
        )

    stage_share = (
        panel.groupby(["year", "row_stage"]).size()
        / panel.groupby("year").size()
    ).reset_index(name="share")
    for stage, group in stage_share.groupby("row_stage"):
        fig.add_trace(
            go.Scatter(x=group["year"], y=group["share"], mode="lines+markers", name=f"stage:{stage}"),
            row=2,
            col=2,
        )

    fig.update_layout(
        title="Time-series diagnostics and source-stage audit",
        barmode="stack",
        height=850,
        template="plotly_white",
    )
    fig.write_html(output_html, include_plotlyjs="cdn")
