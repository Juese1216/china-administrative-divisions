from __future__ import annotations

import numpy as np
import pandas as pd

from .io import STATIC_COLUMNS


def _available_static_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in STATIC_COLUMNS if col in df.columns]


def _status_col(metric: str) -> str:
    return f"status_{metric}"


def _source_col(metric: str) -> str:
    return f"source_{metric}"


def expand_annual_panel(
    df: pd.DataFrame,
    metrics: list[str],
    start_year: int,
    baseline_year: int,
) -> pd.DataFrame:
    """Create one row per unit-year and linearly interpolate internal history gaps.

    The function never extrapolates beyond each unit's available years. Interpolated
    values are flagged per metric and can be excluded from forecasting.
    """

    df = df.copy()
    df["unit_id"] = df["unit_id"].astype("string")
    df["year"] = df["year"].astype(int)
    static_cols = _available_static_columns(df)

    unit_static = (
        df.sort_values("year")
        .groupby("unit_id", as_index=False)[static_cols]
        .last()
    )

    units = unit_static["unit_id"].astype("string").tolist()
    years = list(range(start_year, baseline_year + 1))
    base = pd.MultiIndex.from_product([units, years], names=["unit_id", "year"]).to_frame(index=False)
    panel = base.merge(unit_static, on="unit_id", how="left", suffixes=("", "_static"))

    value_cols = ["unit_id", "year"] + [m for m in metrics if m in df.columns]
    for metric in metrics:
        for col in [_source_col(metric), _status_col(metric)]:
            if col in df.columns and col not in value_cols:
                value_cols.append(col)
    if "source" in df.columns:
        value_cols.append("source")

    panel = panel.merge(df[value_cols], on=["unit_id", "year"], how="left", suffixes=("", "_raw"))

    for metric in metrics:
        if metric not in panel.columns:
            panel[metric] = np.nan
        panel[metric] = pd.to_numeric(panel[metric], errors="coerce")

        source_col = _source_col(metric)
        status_col = _status_col(metric)
        if source_col not in panel.columns:
            panel[source_col] = panel["source"] if "source" in panel.columns else pd.NA
        if status_col not in panel.columns:
            panel[status_col] = pd.NA

        observed_mask = panel[metric].notna()
        panel.loc[observed_mask & panel[status_col].isna(), status_col] = "observed"

        interpolated = panel.groupby("unit_id")[metric].transform(
            lambda values: values.interpolate(method="linear", limit_area="inside")
        )
        newly_filled = panel[metric].isna() & interpolated.notna()
        panel[metric] = interpolated
        panel.loc[newly_filled, status_col] = "interpolated"
        panel.loc[newly_filled, source_col] = "linear_interpolation_between_observed_years"

    panel["row_stage"] = row_stage(panel, metrics)
    return panel.sort_values(["unit_id", "year"]).reset_index(drop=True)


def row_stage(df: pd.DataFrame, metrics: list[str]) -> pd.Series:
    stages = []
    for _, row in df.iterrows():
        metric_statuses = []
        for metric in metrics:
            status_col = _status_col(metric)
            if status_col in df.columns and pd.notna(row.get(metric)):
                metric_statuses.append(str(row.get(status_col) or "observed"))
        if not metric_statuses:
            stages.append("missing")
        elif any(status == "forecast" for status in metric_statuses):
            stages.append("forecast")
        elif any(status == "weak_history_forecast" for status in metric_statuses):
            stages.append("forecast")
        elif any(status == "nowcast" for status in metric_statuses):
            stages.append("nowcast")
        elif any(status == "interpolated" for status in metric_statuses):
            stages.append("interpolated")
        else:
            stages.append("observed")
    return pd.Series(stages, index=df.index)


def add_yearly_changes(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out = df.sort_values(["unit_id", "year"]).copy()
    for metric in metrics:
        out[f"{metric}_yoy_abs"] = out.groupby("unit_id")[metric].diff()
        previous = out.groupby("unit_id")[metric].shift(1)
        out[f"{metric}_yoy_pct"] = np.where(
            previous.abs() > 1e-12,
            out[f"{metric}_yoy_abs"] / previous,
            np.nan,
        )
        out[f"{metric}_log_change"] = np.log1p(out[metric].clip(lower=0)) - np.log1p(previous.clip(lower=0))
    return out


def _robust_0_100(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() < 2:
        return pd.Series(np.nan, index=series.index)
    lo, hi = values.quantile([0.01, 0.99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(50.0, index=series.index).where(values.notna(), np.nan)
    clipped = values.clip(lo, hi)
    return ((clipped - lo) / (hi - lo) * 100).where(values.notna(), np.nan)


def add_indices(df: pd.DataFrame, metrics: list[str], weights: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    score_cols: list[str] = []
    change_cols: list[str] = []

    for metric in metrics:
        score_col = f"{metric}_score"
        change_col = f"{metric}_change_score"
        out[score_col] = out.groupby("year", group_keys=False)[metric].apply(_robust_0_100)
        out[change_col] = out.groupby("year", group_keys=False)[f"{metric}_yoy_pct"].apply(
            lambda s: _robust_0_100(s.abs())
        )
        score_cols.append(score_col)
        change_cols.append(change_col)

    def weighted_average(row: pd.Series, cols: list[str]) -> float:
        numerator = 0.0
        denominator = 0.0
        for metric, col in zip(metrics, cols):
            value = row.get(col)
            if pd.notna(value):
                weight = float(weights.get(metric, 0.0))
                numerator += weight * float(value)
                denominator += weight
        return numerator / denominator if denominator else np.nan

    out["vitality_index"] = out.apply(lambda row: weighted_average(row, score_cols), axis=1)
    out["change_intensity_index"] = out.apply(lambda row: weighted_average(row, change_cols), axis=1)

    pop = out.get("population_yoy_pct", pd.Series(np.nan, index=out.index))
    light = out.get("nightlight_yoy_pct", pd.Series(np.nan, index=out.index))
    built = out.get("builtup_area_yoy_pct", pd.Series(np.nan, index=out.index))
    vitality = out["vitality_index"]
    change = out["change_intensity_index"]

    conditions = [
        (pop < -0.02) & (light < -0.02),
        (built > 0.03) & (light > 0.01),
        (vitality >= 70) & (change >= 50),
        (vitality <= 30) & (change >= 50),
        change < 20,
    ]
    labels = [
        "shrinking",
        "urbanizing",
        "fast_growth",
        "rapid_decline_or_transition",
        "stable",
    ]
    out["change_type"] = np.select(conditions, labels, default="mixed")
    return out


def national_summary(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    grouped = df.groupby("year")
    for year, group in grouped:
        row: dict[str, float | int | str] = {"year": int(year)}
        for metric in metrics:
            row[f"{metric}_sum"] = float(pd.to_numeric(group[metric], errors="coerce").sum(min_count=1))
            row[f"{metric}_mean"] = float(pd.to_numeric(group[metric], errors="coerce").mean())
        row["vitality_index_mean"] = float(group["vitality_index"].mean())
        row["change_intensity_index_mean"] = float(group["change_intensity_index"].mean())
        row["forecast_share"] = float((group["row_stage"] == "forecast").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("year")
