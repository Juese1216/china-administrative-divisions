"""时序预测主流程：经济指标准备、年度统计和未来趋势预测。

这个文件已经把外部经济指标下载和预测建模整合到同一个顶层脚本中，
不再依赖额外子包。
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.parse
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

matplotlib.use("Agg")
import matplotlib.pyplot as plt

econ_DEFAULT_OUTPUT_DIR = Path("data/source/economic_indicators")
econ_DEFAULT_OUTPUT_FILE = econ_DEFAULT_OUTPUT_DIR / "china_macro_indicators.csv"
econ_INDICATORS = {
    "gdp_current_lcu": {
        "code": "NY.GDP.MKTP.CN",
        "name": "GDP（现价本币）",
        "url": "https://data.worldbank.org/indicator/NY.GDP.MKTP.CN?locations=CN",
    },
    "population_total": {
        "code": "SP.POP.TOTL",
        "name": "总人口",
        "url": "https://data.worldbank.org/indicator/SP.POP.TOTL?locations=CN",
    },
}


def econ_build_api_url(indicator_code: str, start_year: int, end_year: int) -> str:
    query = urllib.parse.urlencode(
        {"format": "json", "per_page": 200, "date": f"{start_year}:{end_year}"}
    )
    return (
        f"https://api.worldbank.org/v2/country/CHN/indicator/{indicator_code}?{query}"
    )


def econ_fetch_json(url: str, timeout: int, retries: int) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "administrative-division-course-project/1.0",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"下载失败：{url}；原因：{last_error}") from last_error


def econ_fetch_indicator(
    indicator_key: str, start_year: int, end_year: int, timeout: int, retries: int
) -> pd.DataFrame:
    config = econ_INDICATORS[indicator_key]
    api_url = econ_build_api_url(config["code"], start_year, end_year)
    payload = econ_fetch_json(api_url, timeout=timeout, retries=retries)
    if (
        not isinstance(payload, list)
        or len(payload) < 2
        or (not isinstance(payload[1], list))
    ):
        raise RuntimeError(f"World Bank API 返回格式异常：{config['code']}")
    rows = []
    for item in payload[1]:
        year = item.get("date")
        value = item.get("value")
        if year is None:
            continue
        rows.append({"year": int(year), indicator_key: value})
    return pd.DataFrame(rows)


def econ_build_indicator_table(
    start_year: int, end_year: int, timeout: int, retries: int
) -> pd.DataFrame:
    frame = pd.DataFrame({"year": range(start_year, end_year + 1)})
    for indicator_key in econ_INDICATORS:
        indicator_frame = econ_fetch_indicator(
            indicator_key, start_year, end_year, timeout, retries
        )
        frame = frame.merge(indicator_frame, on="year", how="left")
    frame = frame.sort_values("year").reset_index(drop=True)
    frame["gdp_per_capita_lcu"] = frame["gdp_current_lcu"] / frame["population_total"]
    return frame


def econ_write_metadata(output_file: Path, start_year: int, end_year: int) -> None:
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "country": "China",
        "start_year": start_year,
        "end_year": end_year,
        "source": "World Bank API",
        "indicators": econ_INDICATORS,
        "note": "GDP 和人口用于时序预测的外部回归因子。若某年 API 尚未发布数据，CSV 中会保留空值，由预测脚本自动估计。",
    }
    metadata_path = output_file.with_name("china_macro_indicators_metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


series_DEFAULT_CLASSIFICATION = Path(
    "data/processed/classification/rule_classification.csv"
)
series_DEFAULT_ECONOMIC_INDICATORS = Path(
    "data/source/economic_indicators/china_macro_indicators.csv"
)
series_DEFAULT_OUTPUT_DIR = Path("data/processed/time_series_forecast")
series_PREFERRED_RAW_INDICATORS = [
    "gdp_current_lcu",
    "population_total",
    "gdp_per_capita_lcu",
]
series_PREFERRED_FEATURES = [
    "log_gdp_current_lcu",
    "log_population_total",
    "log_gdp_per_capita_lcu",
    "gdp_current_lcu_growth_rate",
    "population_total_growth_rate",
]


def series_clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def series_configure_plot_style() -> None:
    font_names = {font.name for font in font_manager.fontManager.ttflist}
    for name in [
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
    ]:
        if name in font_names:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 180


def series_read_classification(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"缺少分类结果文件：{path}")
    frame = pd.read_csv(path)
    required = {"record_id", "year", "type_label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{path} 缺少字段：{', '.join(missing)}")
    frame = frame.copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["type_label"] = (
        frame["type_label"].map(series_clean_cell).replace("", "未分类")
    )
    frame = frame.dropna(subset=["year"])
    frame["year"] = frame["year"].astype(int)
    return frame


def series_read_economic_indicators(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "year" not in frame.columns:
        raise SystemExit(f"{path} 缺少字段：year")
    frame = frame.copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["year"])
    frame["year"] = frame["year"].astype(int)
    for column in frame.columns:
        if column != "year":
            try:
                frame[column] = pd.to_numeric(frame[column])
            except (TypeError, ValueError):
                pass
    return frame.sort_values("year").reset_index(drop=True)


def series_complete_year_index(records: pd.DataFrame) -> pd.Index:
    start_year = int(records["year"].min())
    end_year = int(records["year"].max())
    return pd.Index(range(start_year, end_year + 1), name="year")


def series_build_annual_tables(
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = series_complete_year_index(records)
    source_years = set(records["year"].tolist())
    annual_total = (
        records.groupby("year")["record_id"]
        .nunique()
        .reindex(years, fill_value=0)
        .rename("total_count")
        .reset_index()
    )
    annual_total["source_year_present"] = annual_total["year"].isin(source_years)
    annual_total["note"] = np.where(
        annual_total["source_year_present"],
        "原始资料中有该年份记录",
        "原始资料缺失，按 0 条补齐",
    )
    annual_type_wide = (
        records.pivot_table(
            index="year",
            columns="type_label",
            values="record_id",
            aggfunc="nunique",
            fill_value=0,
        )
        .reindex(years, fill_value=0)
        .sort_index(axis=1)
        .reset_index()
    )
    annual_type_wide.columns.name = None
    annual_type_long = annual_type_wide.melt(
        id_vars="year", var_name="type_label", value_name="type_count"
    ).sort_values(["year", "type_label"])
    return (annual_total, annual_type_wide, annual_type_long)


def series_numeric_indicator_columns(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    preferred = [
        column for column in series_PREFERRED_RAW_INDICATORS if column in frame.columns
    ]
    other_numeric = [
        column
        for column in frame.columns
        if column != "year"
        and column not in preferred
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return preferred + other_numeric


def series_growth_clip_for_column(column: str) -> tuple[float, float]:
    if "population" in column:
        return (-0.02, 0.02)
    if "gdp" in column:
        return (-0.05, 0.15)
    return (-0.2, 0.2)


def series_fill_indicator_values(
    years: pd.Series, values: pd.Series, column: str
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    known = numeric.notna()
    if known.sum() == 0:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    if known.sum() == 1:
        return numeric.ffill().bfill()
    filled = numeric.interpolate(method="linear", limit_area="inside").bfill()
    known_positions = np.flatnonzero(known.to_numpy())
    last_known_position = int(known_positions[-1])
    lower, upper = series_growth_clip_for_column(column)
    for position in range(last_known_position + 1, len(filled)):
        observed_until_now = filled.iloc[:position].dropna()
        recent_growth = observed_until_now.pct_change(fill_method=None).dropna().tail(5)
        growth = float(recent_growth.mean()) if not recent_growth.empty else 0.0
        growth = float(np.clip(growth, lower, upper))
        filled.iloc[position] = float(filled.iloc[position - 1]) * (1.0 + growth)
    return filled


def series_build_exogenous_features(
    economic_indicators: pd.DataFrame, start_year: int, forecast_end_year: int
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if economic_indicators.empty:
        return (
            pd.DataFrame(),
            [],
            {"exogenous_available": False, "reason": "未找到经济指标 CSV"},
        )
    years = pd.DataFrame({"year": range(start_year, forecast_end_year + 1)})
    raw_columns = series_numeric_indicator_columns(economic_indicators)
    if not raw_columns:
        return (
            pd.DataFrame(),
            [],
            {"exogenous_available": False, "reason": "经济指标 CSV 中没有数值字段"},
        )
    merged = years.merge(
        economic_indicators[["year", *raw_columns]], on="year", how="left"
    )
    actual_flags = merged[raw_columns].notna().all(axis=1)
    for column in raw_columns:
        merged[column] = series_fill_indicator_values(
            merged["year"], merged[column], column
        )
    if (
        "gdp_current_lcu" in merged.columns
        and "population_total" in merged.columns
        and ("gdp_per_capita_lcu" not in merged.columns)
    ):
        merged["gdp_per_capita_lcu"] = merged["gdp_current_lcu"] / merged[
            "population_total"
        ].replace(0, np.nan)
        raw_columns.append("gdp_per_capita_lcu")
    features = merged.copy()
    features["economic_data_status"] = np.where(actual_flags, "actual", "estimated")
    created_features: list[str] = []
    for column in raw_columns:
        values = pd.to_numeric(features[column], errors="coerce")
        if values.notna().all() and (values > 0).all():
            feature_name = f"log_{column}"
            features[feature_name] = np.log(values)
            created_features.append(feature_name)
        growth_name = f"{column}_growth_rate"
        features[growth_name] = (
            values.pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            * 100
        )
        created_features.append(growth_name)
    feature_columns = [
        column for column in series_PREFERRED_FEATURES if column in features.columns
    ]
    if len(feature_columns) < 2:
        feature_columns = [
            column for column in created_features if column in features.columns
        ]
    feature_columns = feature_columns[:6]
    status = {
        "exogenous_available": bool(feature_columns),
        "reason": (
            "已使用经济指标外部回归因子"
            if feature_columns
            else "经济指标不足以构造特征"
        ),
        "raw_indicator_columns": raw_columns,
        "feature_columns": feature_columns,
        "actual_indicator_years": int(actual_flags.sum()),
        "estimated_indicator_years": int((~actual_flags).sum()),
    }
    return (features, feature_columns, status)


def series_fit_holt_winters(
    series: pd.Series, horizon: int
) -> tuple[np.ndarray, np.ndarray, str]:
    values = series.astype(float).to_numpy()
    if len(values) < 4 or float(np.sum(values)) == 0.0:
        mean_value = float(np.mean(values)) if len(values) else 0.0
        return (
            np.full(len(values), mean_value),
            np.full(horizon, mean_value),
            "mean_fallback",
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                values,
                trend="add",
                damped_trend=True,
                seasonal=None,
                initialization_method="estimated",
            )
            fitted = model.fit(optimized=True, use_brute=False)
        return (
            np.asarray(fitted.fittedvalues),
            np.asarray(fitted.forecast(horizon)),
            "holt_winters",
        )
    except Exception:
        mean_value = float(np.mean(values))
        return (
            np.full(len(values), mean_value),
            np.full(horizon, mean_value),
            "mean_fallback",
        )


def series_fit_linear_regression(
    years: np.ndarray, series: pd.Series, future_years: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    model = LinearRegression()
    x_train = years.reshape(-1, 1)
    y_train = series.astype(float).to_numpy()
    model.fit(x_train, y_train)
    fitted = model.predict(x_train)
    forecast = model.predict(future_years.reshape(-1, 1))
    return (fitted, forecast)


def series_choose_poisson_alpha(x_train: np.ndarray, y_train: np.ndarray) -> float:
    if len(y_train) < 8:
        return 10.0
    split_index = max(4, len(y_train) - 3)
    x_fit = x_train[:split_index]
    y_fit = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]
    best_alpha = 10.0
    best_rmse = float("inf")
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("poisson", PoissonRegressor(alpha=alpha, max_iter=2000)),
            ]
        )
        try:
            model.fit(x_fit, y_fit)
            prediction = model.predict(x_valid)
        except Exception:
            continue
        rmse = float(np.sqrt(np.mean((prediction - y_valid) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
    return best_alpha


def series_fit_exogenous_regression(
    target_name: str,
    years: np.ndarray,
    series: pd.Series,
    future_years: np.ndarray,
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None, list[dict[str, Any]], str]:
    if feature_frame.empty or not feature_columns:
        return (None, None, [], "no_exogenous_features")
    history = feature_frame[feature_frame["year"].isin(years)].sort_values("year")
    future = feature_frame[feature_frame["year"].isin(future_years)].sort_values("year")
    if len(history) != len(years) or len(future) != len(future_years):
        return (None, None, [], "exogenous_years_incomplete")
    x_train = history[feature_columns].astype(float).to_numpy()
    x_future = future[feature_columns].astype(float).to_numpy()
    y_train = series.astype(float).to_numpy()
    if not np.isfinite(x_train).all() or not np.isfinite(x_future).all():
        return (None, None, [], "exogenous_features_contain_nan")
    if len(y_train) < max(6, len(feature_columns) + 3):
        return (None, None, [], "too_few_samples")
    alpha = series_choose_poisson_alpha(x_train, y_train)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("poisson", PoissonRegressor(alpha=alpha, max_iter=2000)),
        ]
    )
    model.fit(x_train, y_train)
    fitted = model.predict(x_train)
    forecast = model.predict(x_future)
    poisson = model.named_steps["poisson"]
    importance = []
    for feature, weight in zip(feature_columns, poisson.coef_, strict=False):
        importance.append(
            {
                "target": target_name,
                "feature": feature,
                "standardized_weight": round(float(weight), 6),
                "abs_weight": round(float(abs(weight)), 6),
                "alpha": float(alpha),
            }
        )
    return (fitted, forecast, importance, "exogenous_poisson")


def series_clamp(values: np.ndarray) -> np.ndarray:
    return np.maximum(values.astype(float), 0.0)


def series_metric_rows(
    target_name: str, actual: np.ndarray, predictions: list[tuple[str, np.ndarray, str]]
) -> list[dict[str, Any]]:
    rows = []
    for model_name, fitted, method_detail in predictions:
        error = actual - series_clamp(fitted)
        rows.append(
            {
                "target": target_name,
                "model": model_name,
                "method_detail": method_detail,
                "mae": round(float(np.mean(np.abs(error))), 4),
                "rmse": round(float(np.sqrt(np.mean(error**2))), 4),
            }
        )
    return rows


def series_forecast_series(
    target_name: str,
    years: np.ndarray,
    values: pd.Series,
    future_years: np.ndarray,
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    hw_fitted, hw_forecast, hw_method = series_fit_holt_winters(
        values, len(future_years)
    )
    lr_fitted, lr_forecast = series_fit_linear_regression(years, values, future_years)
    exog_fitted, exog_forecast, importance, exog_method = (
        series_fit_exogenous_regression(
            target_name, years, values, future_years, feature_frame, feature_columns
        )
    )
    hw_forecast = series_clamp(hw_forecast)
    lr_forecast = series_clamp(lr_forecast)
    prediction_rows = [
        ("holt_winters", hw_fitted, hw_method),
        ("linear_regression", lr_fitted, "ordinary_least_squares"),
    ]
    if exog_forecast is not None and exog_fitted is not None:
        exog_forecast = series_clamp(exog_forecast)
        chosen = exog_forecast
        chosen_model = "exogenous_poisson"
        exog_forecast_for_frame: np.ndarray | list[float] = exog_forecast
        prediction_rows.append(("exogenous_poisson", exog_fitted, exog_method))
    else:
        chosen = hw_forecast if hw_method == "holt_winters" else lr_forecast
        chosen_model = (
            "holt_winters" if hw_method == "holt_winters" else "linear_regression"
        )
        exog_forecast_for_frame = [np.nan] * len(future_years)
    forecast_frame = pd.DataFrame(
        {
            "target": target_name,
            "forecast_year": future_years,
            "horizon": np.arange(1, len(future_years) + 1),
            "holt_winters_forecast": np.round(hw_forecast, 4),
            "linear_regression_forecast": np.round(lr_forecast, 4),
            "exogenous_regression_forecast": np.round(exog_forecast_for_frame, 4),
            "chosen_forecast": np.round(chosen, 4),
            "chosen_forecast_rounded": np.rint(chosen).astype(int),
            "chosen_model": chosen_model,
        }
    )
    metrics = series_metric_rows(
        target_name, values.astype(float).to_numpy(), prediction_rows
    )
    return (forecast_frame, metrics, importance)


def series_build_forecasts(
    annual_total: pd.DataFrame,
    annual_type_wide: pd.DataFrame,
    forecast_years: int,
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = annual_total["year"].to_numpy(dtype=int)
    future_year_values = np.arange(years.max() + 1, years.max() + forecast_years + 1)
    total_forecast, metric_rows_total, importance_total = series_forecast_series(
        "全部变更",
        years,
        annual_total["total_count"],
        future_year_values,
        feature_frame,
        feature_columns,
    )
    type_forecasts = []
    metric_rows_all = metric_rows_total
    importance_all = importance_total
    for type_label in [
        column for column in annual_type_wide.columns if column != "year"
    ]:
        forecast_frame, rows, importance = series_forecast_series(
            type_label,
            years,
            annual_type_wide[type_label],
            future_year_values,
            feature_frame,
            feature_columns,
        )
        type_forecasts.append(forecast_frame)
        metric_rows_all.extend(rows)
        importance_all.extend(importance)
    type_forecast = (
        pd.concat(type_forecasts, ignore_index=True)
        if type_forecasts
        else pd.DataFrame()
    )
    metrics = pd.DataFrame(metric_rows_all)
    importance_frame = pd.DataFrame(importance_all)
    if not importance_frame.empty:
        importance_frame = importance_frame.sort_values(
            ["target", "abs_weight"], ascending=[True, False]
        )
    return (total_forecast, type_forecast, metrics, importance_frame)


def series_build_high_year_tables(
    annual_total: pd.DataFrame, total_forecast: pd.DataFrame, top_years: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    high_observed = (
        annual_total.sort_values(["total_count", "year"], ascending=[False, True])
        .head(top_years)
        .reset_index(drop=True)
    )
    high_observed.insert(0, "rank", range(1, len(high_observed) + 1))
    likely_high = (
        total_forecast.sort_values(
            ["chosen_forecast", "forecast_year"], ascending=[False, True]
        )
        .head(top_years)
        .reset_index(drop=True)
    )
    likely_high.insert(0, "rank", range(1, len(likely_high) + 1))
    likely_high["reason"] = "未来预测值较高，适合作为趋势图中的重点年份"
    return (high_observed, likely_high)


def series_plot_total_trend(
    annual_total: pd.DataFrame, total_forecast: pd.DataFrame, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        annual_total["year"],
        annual_total["total_count"],
        marker="o",
        linewidth=2,
        label="历史总数",
        color="#2458A6",
    )
    ax.plot(
        total_forecast["forecast_year"],
        total_forecast["holt_winters_forecast"],
        marker="o",
        linestyle="--",
        linewidth=2,
        label="Holt-Winters 预测",
        color="#D94841",
    )
    ax.plot(
        total_forecast["forecast_year"],
        total_forecast["linear_regression_forecast"],
        marker="s",
        linestyle=":",
        linewidth=2,
        label="线性回归预测",
        color="#2C7A4B",
    )
    if (
        "exogenous_regression_forecast" in total_forecast.columns
        and total_forecast["exogenous_regression_forecast"].notna().any()
    ):
        ax.plot(
            total_forecast["forecast_year"],
            total_forecast["exogenous_regression_forecast"],
            marker="^",
            linestyle="-.",
            linewidth=2,
            label="经济指标 Poisson 预测",
            color="#8A5A00",
        )
    ax.axvline(
        int(annual_total["year"].max()), color="#777777", linestyle="--", linewidth=1
    )
    ax.set_title("行政区划变更年度总数趋势预测")
    ax.set_xlabel("年份")
    ax.set_ylabel("记录数")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def series_plot_type_trends(
    annual_type_wide: pd.DataFrame, type_forecast: pd.DataFrame, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    type_columns = [column for column in annual_type_wide.columns if column != "year"]
    colors = ["#2458A6", "#D94841", "#2C7A4B", "#8A5A00", "#6F4AA8", "#008C8C"]
    for index, type_label in enumerate(type_columns):
        color = colors[index % len(colors)]
        ax.plot(
            annual_type_wide["year"],
            annual_type_wide[type_label],
            marker="o",
            linewidth=1.8,
            label=f"{type_label} 历史",
            color=color,
        )
        forecast_part = type_forecast[type_forecast["target"].eq(type_label)]
        if not forecast_part.empty:
            ax.plot(
                forecast_part["forecast_year"],
                forecast_part["chosen_forecast"],
                linestyle="--",
                linewidth=1.6,
                color=color,
                alpha=0.8,
            )
    ax.axvline(
        int(annual_type_wide["year"].max()),
        color="#777777",
        linestyle="--",
        linewidth=1,
    )
    ax.set_title("各类变更年度趋势和未来外推")
    ax.set_xlabel("年份")
    ax.set_ylabel("记录数")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def series_build_visualization_json(
    annual_total: pd.DataFrame,
    annual_type_long: pd.DataFrame,
    total_forecast: pd.DataFrame,
    type_forecast: pd.DataFrame,
    likely_high: pd.DataFrame,
    economic_features: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "annual_total": annual_total.to_dict(orient="records"),
        "annual_type_counts": annual_type_long.to_dict(orient="records"),
        "forecast_total": total_forecast.to_dict(orient="records"),
        "forecast_by_type": type_forecast.to_dict(orient="records"),
        "likely_high_years": likely_high.to_dict(orient="records"),
        "economic_indicator_features": (
            economic_features.to_dict(orient="records")
            if not economic_features.empty
            else []
        ),
    }


def series_write_outputs(
    output_dir: Path,
    outputs: dict[str, pd.DataFrame],
    visualization: dict[str, Any],
    debug_output: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
    if debug_output:
        (output_dir / "forecast_data.json").write_text(
            json.dumps(visualization, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成年度变更统计和未来趋势预测。")
    parser.add_argument(
        "--classification", type=Path, default=series_DEFAULT_CLASSIFICATION
    )
    parser.add_argument(
        "--economic-indicators", type=Path, default=series_DEFAULT_ECONOMIC_INDICATORS
    )
    parser.add_argument("--output-dir", type=Path, default=series_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--forecast-years", type=int, default=5)
    parser.add_argument("--top-years", type=int, default=5)
    parser.add_argument(
        "--disable-exogenous",
        action="store_true",
        help="只运行单变量预测，不使用经济指标。",
    )
    parser.add_argument(
        "--skip-economic-download",
        action="store_true",
        help="不重新下载 World Bank GDP/人口指标，直接使用本地 CSV。",
    )
    parser.add_argument("--economic-start-year", type=int, default=1999)
    parser.add_argument("--economic-end-year", type=int, default=2025)
    parser.add_argument("--economic-timeout", type=int, default=90)
    parser.add_argument("--economic-retries", type=int, default=3)
    parser.add_argument(
        "--debug-output",
        action="store_true",
        help="Write auxiliary feature/high-year JSON and CSV files. Default keeps only report/Web essentials.",
    )
    return parser.parse_args()


def prepare_economic_indicators(args: argparse.Namespace) -> None:
    if args.skip_economic_download or args.disable_exogenous:
        return
    if args.economic_start_year > args.economic_end_year:
        raise SystemExit("--economic-start-year 不能大于 --economic-end-year")
    frame = econ_build_indicator_table(
        args.economic_start_year,
        args.economic_end_year,
        args.economic_timeout,
        args.economic_retries,
    )
    args.economic_indicators.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.economic_indicators, index=False, encoding="utf-8-sig")
    econ_write_metadata(
        args.economic_indicators, args.economic_start_year, args.economic_end_year
    )
    print(f"经济指标输出文件：{args.economic_indicators}")


def run_forecast(args: argparse.Namespace) -> None:
    if args.forecast_years <= 0:
        raise SystemExit("--forecast-years 必须大于 0")
    series_configure_plot_style()
    records = series_read_classification(args.classification)
    annual_total, annual_type_wide, annual_type_long = series_build_annual_tables(
        records
    )
    forecast_end_year = int(annual_total["year"].max()) + args.forecast_years
    economic_indicators = (
        pd.DataFrame()
        if args.disable_exogenous
        else series_read_economic_indicators(args.economic_indicators)
    )
    economic_features, feature_columns, exogenous_status = (
        series_build_exogenous_features(
            economic_indicators, int(annual_total["year"].min()), forecast_end_year
        )
    )
    total_forecast, type_forecast, metrics, feature_importance = series_build_forecasts(
        annual_total,
        annual_type_wide,
        args.forecast_years,
        economic_features,
        feature_columns,
    )
    high_observed, likely_high = series_build_high_year_tables(
        annual_total, total_forecast, args.top_years
    )
    overview = pd.DataFrame(
        [
            {
                "record_count": int(records["record_id"].nunique()),
                "start_year": int(annual_total["year"].min()),
                "end_year": int(annual_total["year"].max()),
                "historical_year_count": int(len(annual_total)),
                "forecast_year_count": int(args.forecast_years),
                "type_count": int(records["type_label"].nunique()),
                "exogenous_model_used": bool(feature_columns),
                "exogenous_feature_columns": " / ".join(feature_columns),
                "economic_indicator_path": str(args.economic_indicators),
                "economic_indicator_status": exogenous_status.get("reason", ""),
                "historical_peak_year": int(high_observed.iloc[0]["year"]),
                "historical_peak_count": int(high_observed.iloc[0]["total_count"]),
                "future_peak_year": int(likely_high.iloc[0]["forecast_year"]),
                "future_peak_forecast": float(likely_high.iloc[0]["chosen_forecast"]),
            }
        ]
    )
    outputs = {
        "annual_total.csv": annual_total,
        "annual_type_counts_long.csv": annual_type_long,
        "forecast_total.csv": total_forecast,
        "forecast_by_type.csv": type_forecast,
        "model_metrics.csv": metrics,
        "forecast_overview.csv": overview,
    }
    if args.debug_output:
        outputs.update(
            {
                "annual_type_counts.csv": annual_type_wide,
                "economic_indicator_features.csv": economic_features,
                "exogenous_feature_importance.csv": feature_importance,
                "high_observed_years.csv": high_observed,
                "likely_high_years.csv": likely_high,
            }
        )
    visualization = series_build_visualization_json(
        annual_total,
        annual_type_long,
        total_forecast,
        type_forecast,
        likely_high,
        economic_features,
    )
    series_write_outputs(args.output_dir, outputs, visualization, args.debug_output)
    series_plot_total_trend(
        annual_total, total_forecast, args.output_dir / "trend_forecast_total.png"
    )
    series_plot_type_trends(
        annual_type_wide,
        type_forecast,
        args.output_dir / "type_trends_and_forecast.png",
    )
    print(f"时序预测输出目录：{args.output_dir}")
    print(overview.to_string(index=False))
    print("\n未来总量预测：")
    print(total_forecast.to_string(index=False))
    print("\n可能的高变更年份：")
    print(
        likely_high[
            [
                "rank",
                "forecast_year",
                "chosen_forecast_rounded",
                "chosen_model",
                "reason",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    args = parse_args()
    prepare_economic_indicators(args)
    run_forecast(args)


if __name__ == "__main__":
    main()
