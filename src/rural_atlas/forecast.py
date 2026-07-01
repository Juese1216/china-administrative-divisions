from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, LinearRegression
import warnings

from .io import STATIC_COLUMNS


@dataclass
class ForecastSettings:
    forecast_start_year: int = 2027
    forecast_end_year: int = 2035
    damping: float = 0.90
    max_abs_log_growth_per_year: float = 0.20
    min_points_for_trend: int = 4
    prediction_interval_z: float = 1.96
    use_interpolated_for_forecast: bool = False


def _fit_regression(years: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    x = (years - years.min()).reshape(-1, 1)
    y = np.log1p(np.clip(values, a_min=0, a_max=None))

    if len(years) >= 4:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            try:
                model = HuberRegressor(epsilon=1.35, alpha=0.0001).fit(x, y)
                intercept = float(model.intercept_)
                slope = float(model.coef_[0])
            except Exception:
                model = LinearRegression().fit(x, y)
                intercept = float(model.intercept_)
                slope = float(model.coef_[0])
    else:
        model = LinearRegression().fit(x, y)
        intercept = float(model.intercept_)
        slope = float(model.coef_[0])

    residuals = y - (intercept + slope * x.ravel())
    sigma = float(np.nanstd(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    return intercept, slope, sigma


def _damped_horizon(horizon: int, damping: float) -> float:
    if horizon <= 0:
        return 0.0
    if abs(damping - 1.0) < 1e-12:
        return float(horizon)
    powers = [damping**step for step in range(1, horizon + 1)]
    return float(sum(powers))


def _forecast_one_series(
    series: pd.DataFrame,
    metric: str,
    settings: ForecastSettings,
) -> pd.DataFrame:
    value_col = metric
    status_col = f"status_{metric}"
    source_col = f"source_{metric}"
    working = series[["year", value_col] + ([status_col] if status_col in series.columns else [])].copy()
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
    working = working.dropna(subset=[value_col])
    if status_col in working.columns and not settings.use_interpolated_for_forecast:
        eligible_statuses = {"observed", "observed_or_modelled_epoch"}
        working = working[working[status_col].fillna("observed").isin(eligible_statuses)]

    working = working[working["year"] < settings.forecast_start_year]
    if working.empty:
        return pd.DataFrame()

    years = working["year"].astype(int).to_numpy()
    values = working[value_col].astype(float).to_numpy()
    latest_idx = int(np.argmax(years))
    anchor_year = int(years[latest_idx])
    anchor_value = float(values[latest_idx])
    anchor_log = float(np.log1p(max(anchor_value, 0.0)))

    weak_history = len(years) < settings.min_points_for_trend
    if len(years) >= 2:
        _, slope, sigma = _fit_regression(years, values)
    else:
        slope, sigma = 0.0, 0.0

    slope = float(np.clip(slope, -settings.max_abs_log_growth_per_year, settings.max_abs_log_growth_per_year))
    if weak_history:
        slope *= 0.25
        sigma = max(sigma, 0.10)

    rows = []
    for year in range(settings.forecast_start_year, settings.forecast_end_year + 1):
        horizon = year - anchor_year
        damped_h = _damped_horizon(horizon, settings.damping)
        yhat = anchor_log + slope * damped_h
        uncertainty = settings.prediction_interval_z * (sigma + 0.015 * max(horizon, 1))
        value = max(float(np.expm1(yhat)), 0.0)
        lower = max(float(np.expm1(yhat - uncertainty)), 0.0)
        upper = max(float(np.expm1(yhat + uncertainty)), 0.0)
        rows.append(
            {
                "year": year,
                metric: value,
                f"{metric}_lower95": lower,
                f"{metric}_upper95": upper,
                status_col: "weak_history_forecast" if weak_history else "forecast",
                source_col: "statistical_forecast_damped_huber_log_trend",
                f"forecast_model_{metric}": "damped_huber_log_trend",
                f"forecast_anchor_year_{metric}": anchor_year,
                f"forecast_training_points_{metric}": int(len(years)),
            }
        )
    return pd.DataFrame(rows)


def forecast_panel(
    annual_panel: pd.DataFrame,
    metrics: list[str],
    settings: ForecastSettings,
) -> pd.DataFrame:
    static_cols = [col for col in STATIC_COLUMNS if col in annual_panel.columns]
    forecast_rows = []

    for unit_id, group in annual_panel.groupby("unit_id", sort=False):
        group = group.sort_values("year")
        static = group[static_cols].dropna(how="all").tail(1)
        static_values = static.iloc[0].to_dict() if not static.empty else {"unit_id": unit_id}
        unit_forecast = pd.DataFrame({"year": range(settings.forecast_start_year, settings.forecast_end_year + 1)})
        unit_forecast["unit_id"] = str(unit_id)
        for key, value in static_values.items():
            if key not in ["unit_id", "year"]:
                unit_forecast[key] = value

        for metric in metrics:
            forecast = _forecast_one_series(group, metric, settings)
            if forecast.empty:
                continue
            unit_forecast = unit_forecast.merge(forecast, on="year", how="left")

        forecast_rows.append(unit_forecast)

    if not forecast_rows:
        return pd.DataFrame()

    out = pd.concat(forecast_rows, ignore_index=True)
    metric_cols = [metric for metric in metrics if metric in out.columns]
    out = out.dropna(subset=metric_cols, how="all")
    out["row_stage"] = "forecast"
    return out.sort_values(["unit_id", "year"]).reset_index(drop=True)
