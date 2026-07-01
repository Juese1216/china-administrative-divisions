from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.subplots import make_subplots
from sklearn.linear_model import HuberRegressor, LinearRegression
from shapely.geometry import Point, shape
from shapely.validation import make_valid


COUNT_COLS = ["township_street_count", "street_count", "town_count", "township_count", "other_township_level_count"]
TAIWAN_ADCODE = 710000
TAIWAN_NAME = "台湾省"
NO_DATA_FILL_COLOR = "#cbd5e1"
NO_DATA_LINE_COLOR = "#475467"
CHINA_MAP_LON_RANGE = [72, 136]
CHINA_MAP_LAT_RANGE = [16, 55]


def _read_datav_centers(boundary_geojson: str | Path) -> pd.DataFrame:
    data = json.loads(Path(boundary_geojson).read_text(encoding="utf-8"))
    rows = []
    for feature in data["features"]:
        props = feature["properties"]
        center = props.get("centroid") or props.get("center")
        if not center:
            continue
        rows.append(
            {
                "adcode": int(props["adcode"]),
                "unit_id": str(props["adcode"]),
                "province": props["name"],
                "lon": float(center[0]),
                "lat": float(center[1]),
            }
        )
    return pd.DataFrame(rows)


def _classify_township_type(name: str) -> str:
    if name.endswith("街道办事处") or name.endswith("街道"):
        return "street"
    if name.endswith("镇"):
        return "town"
    if name.endswith("乡") or name.endswith("民族乡") or name.endswith("苏木") or name.endswith("民族苏木"):
        return "township"
    return "other"


def build_township_street_counts(
    streets_dir: str | Path,
    boundary_geojson: str | Path,
    output_csv: str | Path,
    nowcast_end_year: int = 2026,
    yearbook_province_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Build province-year township/street quantity panel from yearly street CSV files."""

    streets_dir = Path(streets_dir)
    centers = _read_datav_centers(boundary_geojson)
    province_names = centers.assign(provinceCode=lambda d: d["adcode"].astype(str).str[:2]).set_index("provinceCode")[
        "province"
    ]
    center_by_code = centers.assign(provinceCode=lambda d: d["adcode"].astype(str).str[:2]).set_index("provinceCode")
    center_by_province = centers.set_index("province")

    rows = []
    if yearbook_province_csv and Path(yearbook_province_csv).exists():
        yearbook = pd.read_csv(yearbook_province_csv, dtype={"year": int, "province": str})
        yearbook["unit_id"] = yearbook["province"].map(center_by_province["unit_id"])
        yearbook["adcode"] = yearbook["province"].map(center_by_province["adcode"])
        yearbook["lon"] = yearbook["province"].map(center_by_province["lon"])
        yearbook["lat"] = yearbook["province"].map(center_by_province["lat"])
        rows.append(yearbook)

    for path in sorted(streets_dir.glob("CN_streets_*.csv")):
        year = int(path.stem.rsplit("_", 1)[-1])
        df = pd.read_csv(path, dtype=str)
        df["township_type"] = df["name"].fillna("").map(_classify_township_type)
        summary = (
            df.pivot_table(
                index="provinceCode",
                columns="township_type",
                values="code",
                aggfunc="count",
                fill_value=0,
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
        for col in ["street", "town", "township", "other"]:
            if col not in summary.columns:
                summary[col] = 0
        summary["township_street_count"] = summary[["street", "town", "township", "other"]].sum(axis=1)
        summary["street_count"] = summary["street"]
        summary["town_count"] = summary["town"]
        summary["township_count"] = summary["township"]
        summary["other_township_level_count"] = summary["other"]
        summary["year"] = year
        summary["unit_id"] = summary["provinceCode"].astype(str).str.zfill(2) + "0000"
        summary["adcode"] = summary["unit_id"].astype(int)
        summary["province"] = summary["provinceCode"].map(province_names)
        summary["lon"] = summary["provinceCode"].map(center_by_code["lon"])
        summary["lat"] = summary["provinceCode"].map(center_by_code["lat"])
        summary["source_township_street_count"] = (
            "gaohr_cn_streets_csv_from_nbs_statistical_division_codes"
        )
        summary["status_township_street_count"] = "observed"
        rows.append(summary)

    observed = pd.concat(rows, ignore_index=True, sort=False)
    observed = observed.dropna(subset=["unit_id", "adcode"]).copy()
    observed["year"] = observed["year"].astype(int)
    for col in COUNT_COLS:
        observed[col] = pd.to_numeric(observed[col], errors="coerce").fillna(0).astype(int)

    interpolated_rows = []
    for unit_id, group in observed.groupby("unit_id"):
        group = group.sort_values("year")
        static = group[["unit_id", "adcode", "province", "lon", "lat"]].iloc[-1].to_dict()
        year_range = range(int(group["year"].min()), int(group["year"].max()) + 1)
        indexed = group.set_index("year")
        for year in year_range:
            if year in indexed.index:
                continue
            row = {**static, "year": year}
            for col in COUNT_COLS:
                series = indexed[col].astype(float).reindex(year_range).interpolate(method="linear")
                row[col] = int(round(series.loc[year]))
            row["source_township_street_count"] = "linear_interpolation_between_nbs_yearbook_and_statistical_division_counts"
            row["status_township_street_count"] = "interpolated"
            interpolated_rows.append(row)
    if interpolated_rows:
        observed = pd.concat([observed, pd.DataFrame(interpolated_rows)], ignore_index=True, sort=False)

    latest_year = int(observed["year"].max())
    latest = observed[observed["year"].eq(latest_year)].copy()
    nowcasts = []
    for year in range(latest_year + 1, nowcast_end_year + 1):
        carried = latest.copy()
        carried["year"] = year
        carried["source_township_street_count"] = f"carried_forward_from_{latest_year}_statistical_division_codes"
        carried["status_township_street_count"] = "nowcast"
        nowcasts.append(carried)
    out = pd.concat([observed, *nowcasts], ignore_index=True, sort=False)

    keep = [
        "unit_id",
        "adcode",
        "province",
        "lon",
        "lat",
        "year",
        "township_street_count",
        "street_count",
        "town_count",
        "township_count",
        "other_township_level_count",
        "source_township_street_count",
        "status_township_street_count",
    ]
    out = out[keep].sort_values(["unit_id", "year"]).reset_index(drop=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return out


def build_population_external_regressors(
    population_panel_csv: str | Path,
    boundary_geojson: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """Aggregate GHSL population grid points to provinces for external regression."""

    population_panel_csv = Path(population_panel_csv)
    if not population_panel_csv.exists():
        return pd.DataFrame()

    geojson = json.loads(Path(boundary_geojson).read_text(encoding="utf-8"))
    province_features = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        try:
            adcode = int(props.get("adcode"))
        except (TypeError, ValueError):
            continue
        if adcode > 650000 or not props.get("name"):
            continue
        province_features.append(
            {
                "adcode": adcode,
                "unit_id": str(adcode),
                "province": props["name"],
                "geometry": make_valid(shape(feature["geometry"])),
            }
        )

    panel = pd.read_csv(
        population_panel_csv,
        usecols=["unit_id", "lon", "lat", "year", "population", "source_population", "status_population"],
    )
    panel = panel.dropna(subset=["unit_id", "lon", "lat", "year", "population"]).copy()
    locations = panel[["unit_id", "lon", "lat"]].drop_duplicates("unit_id")

    assignments = []
    for row in locations.itertuples(index=False):
        point = Point(float(row.lon), float(row.lat))
        matched = None
        for province in province_features:
            if province["geometry"].covers(point):
                matched = province
                break
        if matched:
            assignments.append(
                {
                    "unit_id": row.unit_id,
                    "province_unit_id": matched["unit_id"],
                    "adcode": matched["adcode"],
                    "province": matched["province"],
                }
            )

    if not assignments:
        return pd.DataFrame()

    assigned = panel.merge(pd.DataFrame(assignments), on="unit_id", how="inner")
    source_summary = (
        assigned.groupby(["province_unit_id", "adcode", "province", "year"], as_index=False)
        .agg(
            population_external=("population", "sum"),
            source_population_external=("source_population", lambda s: ";".join(sorted(set(s.dropna().astype(str)))[:3])),
            status_population_external=("status_population", lambda s: ";".join(sorted(set(s.dropna().astype(str)))[:3])),
        )
        .rename(columns={"province_unit_id": "unit_id"})
    )
    source_summary["year"] = source_summary["year"].astype(int)
    source_summary = source_summary.sort_values(["unit_id", "year"]).reset_index(drop=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    source_summary.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return source_summary


def _forecast_count_series(years: np.ndarray, values: np.ndarray, target_years: list[int]) -> list[int]:
    if len(years) < 2:
        return [int(round(values[-1])) for _ in target_years]
    x = (years - years.min()).reshape(-1, 1)
    y = np.asarray(values, dtype=float)
    try:
        model = HuberRegressor(epsilon=1.35, alpha=0.0001).fit(x, y)
        intercept = float(model.intercept_)
        slope = float(model.coef_[0])
    except Exception:
        model = LinearRegression().fit(x, y)
        intercept = float(model.intercept_)
        slope = float(model.coef_[0])

    latest_year = int(years.max())
    latest_value = float(values[np.argmax(years)])
    forecasts = []
    for year in target_years:
        horizon = year - latest_year
        damped_horizon = sum(0.88**i for i in range(1, horizon + 1))
        fitted_latest = intercept + slope * (latest_year - years.min())
        value = latest_value + (slope * damped_horizon) + 0.25 * (fitted_latest - latest_value)
        forecasts.append(max(0, int(round(value))))
    return forecasts


def _damped_sum(horizon: int, damping: float = 0.88) -> float:
    return float(sum(damping**i for i in range(1, horizon + 1))) if horizon > 0 else 0.0


def _external_feature_cols(external: pd.DataFrame | None) -> list[str]:
    if external is None or external.empty:
        return []
    candidates = ["population_external", "gdp_external", "gdp_per_capita_external"]
    return [col for col in candidates if col in external.columns and pd.to_numeric(external[col], errors="coerce").notna().any()]


def _forecast_count_series_with_external(
    years: np.ndarray,
    values: np.ndarray,
    target_years: list[int],
    external: pd.DataFrame | None,
) -> tuple[list[int], str]:
    base = _forecast_count_series(years, values, target_years)
    feature_cols = _external_feature_cols(external)
    if not feature_cols:
        return base, "damped_huber_trend"

    aligned = pd.DataFrame({"year": years.astype(int), "value": values.astype(float)})
    ext = external[["year", *feature_cols]].copy()
    ext["year"] = ext["year"].astype(int)
    for col in feature_cols:
        ext[col] = pd.to_numeric(ext[col], errors="coerce")
    aligned = aligned.merge(ext, on="year", how="inner").dropna(subset=feature_cols + ["value"])
    if len(aligned) < max(6, len(feature_cols) + 3):
        return base, "damped_huber_trend"

    latest_year = int(years.max())
    latest_value = float(values[np.argmax(years)])
    target_ext = ext[ext["year"].isin([latest_year, *target_years])].dropna(subset=feature_cols)
    if set([latest_year, *target_years]) - set(target_ext["year"].astype(int)):
        return base, "damped_huber_trend"

    def design(df: pd.DataFrame) -> np.ndarray:
        columns = [((df["year"].astype(float) - aligned["year"].min()) / 10.0).to_numpy()]
        for col in feature_cols:
            columns.append(np.log1p(np.clip(df[col].astype(float).to_numpy(), a_min=0, a_max=None)))
        return np.column_stack(columns)

    x_train = design(aligned)
    y_train = np.log1p(np.clip(aligned["value"].astype(float).to_numpy(), a_min=0, a_max=None))
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    x_train = (x_train - mean) / std
    try:
        model = HuberRegressor(epsilon=1.35, alpha=0.001).fit(x_train, y_train)
    except Exception:
        model = LinearRegression().fit(x_train, y_train)

    target_ext = target_ext.set_index("year").sort_index()
    latest_features = (design(target_ext.loc[[latest_year]].reset_index()) - mean) / std
    latest_fit = float(model.predict(latest_features)[0])
    anchor_log = float(np.log1p(max(latest_value, 0.0)))

    adjusted = []
    for base_value, year in zip(base, target_years):
        features = (design(target_ext.loc[[year]].reset_index()) - mean) / std
        external_fit = float(model.predict(features)[0])
        horizon = max(year - latest_year, 1)
        damped_ratio = _damped_sum(horizon) / horizon
        external_value = max(float(np.expm1(anchor_log + (external_fit - latest_fit) * damped_ratio)), 0.0)
        clipped_external = float(np.clip(external_value, base_value * 0.75, base_value * 1.25))
        adjusted.append(max(0, int(round(0.65 * base_value + 0.35 * clipped_external))))
    return adjusted, "hybrid_damped_huber_trend_external_" + "_".join(feature_cols)


def add_township_forecast(
    panel: pd.DataFrame,
    output_csv: str | Path,
    forecast_start_year: int = 2027,
    forecast_end_year: int = 2035,
    external_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    target_years = list(range(forecast_start_year, forecast_end_year + 1))
    forecasts = []
    observed = panel[panel["status_township_street_count"].eq("observed")]
    static_cols = ["unit_id", "adcode", "province", "lon", "lat"]
    for unit_id, group in observed.groupby("unit_id"):
        static = panel[panel["unit_id"].eq(unit_id)][static_cols].tail(1).iloc[0].to_dict()
        years = group["year"].astype(int).to_numpy()
        unit_external = None
        if external_panel is not None and not external_panel.empty:
            unit_external = external_panel[external_panel["unit_id"].astype(str).eq(str(unit_id))].copy()
        forecast_values = {}
        model_labels = {}
        for col in COUNT_COLS:
            values = group[col].astype(float).to_numpy()
            forecast_values[col], model_labels[col] = _forecast_count_series_with_external(
                years,
                values,
                target_years,
                unit_external,
            )
        for idx, year in enumerate(target_years):
            row = {**static, "year": year}
            for col in COUNT_COLS:
                row[col] = forecast_values[col][idx]
            if any(label.startswith("hybrid_") for label in model_labels.values()):
                row["source_township_street_count"] = (
                    "statistical_forecast_from_2009_2023_counts_with_external_regressors"
                )
            else:
                row["source_township_street_count"] = "statistical_forecast_from_2009_2023_nbs_yearbook_and_division_counts"
            row["status_township_street_count"] = "forecast"
            row["forecast_model_township_street_count"] = model_labels["township_street_count"]
            forecasts.append(row)
    out = pd.concat([panel, pd.DataFrame(forecasts)], ignore_index=True, sort=False)
    if external_panel is not None and not external_panel.empty:
        merge_cols = [
            col
            for col in [
                "unit_id",
                "year",
                "population_external",
                "source_population_external",
                "status_population_external",
                "gdp_external",
                "source_gdp_external",
                "status_gdp_external",
            ]
            if col in external_panel.columns
        ]
        if len(merge_cols) > 2:
            out = out.merge(external_panel[merge_cols], on=["unit_id", "year"], how="left")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return out


def build_natural_village_series(output_csv: str | Path) -> pd.DataFrame:
    """Build national natural-village quantity series from agricultural census points.

    Official public points used here:
    - 2016 year-end: 3.17 million natural villages.
    - 2016 count is 3.8% lower than 2006, so 2006 is derived from the same official statement.
    """

    observed_2016 = 3_170_000.0
    observed_2006 = observed_2016 / (1 - 0.038)
    annual_log_rate = math.log(observed_2016 / observed_2006) / 10
    rows = []
    for year in range(2006, 2036):
        if year == 2006:
            value = observed_2006
            status = "derived_observed"
            source = "nbs_third_agricultural_census_interpretation_2016_down_3_8pct_from_2006"
        elif year == 2016:
            value = observed_2016
            status = "observed"
            source = "nbs_third_agricultural_census_interpretation_2016_3_17_million"
        elif year < 2016:
            value = observed_2006 * math.exp(annual_log_rate * (year - 2006))
            status = "interpolated"
            source = "log_interpolation_between_agricultural_census_points"
        elif year <= 2026:
            value = observed_2016 * math.exp(annual_log_rate * (year - 2016))
            status = "nowcast"
            source = "two_point_agricultural_census_trend_nowcast"
        else:
            damped = sum(0.90**i for i in range(1, year - 2026 + 1))
            base_2026 = observed_2016 * math.exp(annual_log_rate * (2026 - 2016))
            value = base_2026 * math.exp(annual_log_rate * damped)
            status = "forecast"
            source = "two_point_agricultural_census_trend_forecast"
        rows.append(
            {
                "scope": "全国",
                "year": year,
                "natural_village_count": int(round(value)),
                "status_natural_village_count": status,
                "source_natural_village_count": source,
            }
        )
    df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return df


OFFICIAL_TOWNSHIP_NATIONAL_ROWS = [
    (2009, 40858, 19322, 14848, 6686, "nbs_yearbook_machine_table"),
    (2010, 40906, 19410, 14571, 6923, "nbs_yearbook_machine_table"),
    (2011, 40466, 19683, 13587, 7194, "nbs_yearbook_machine_table"),
    (2012, 40446, 19881, 13281, 7282, "nbs_yearbook_machine_xls"),
    (2013, 40497, 20117, 12812, 7566, "nbs_yearbook_machine_xls"),
    (2014, 40381, 20401, 12282, 7696, "nbs_yearbook_image_manual_check"),
    (2015, 39789, 20515, 11315, 7957, "nbs_yearbook_image_manual_check"),
    (2016, 39862, 20883, 10872, 8105, "nbs_yearbook_image_manual_check"),
    (2017, 39888, 21116, 10529, 8241, "nbs_yearbook_image_manual_check"),
    (2018, 39945, 21297, 10253, 8393, "nbs_yearbook_image_manual_check"),
    (2019, 38755, 21013, 9221, 8519, "nbs_yearbook_image_manual_check"),
    (2020, 38741, 21157, 8809, 8773, "nbs_yearbook_machine_xls"),
    (2021, 38558, 21322, 8309, 8925, "nbs_yearbook_image_manual_check"),
    (2022, 38602, 21389, 8227, 8984, "nbs_yearbook_image_manual_check"),
    (2023, 38658, 21421, 8190, 9045, "nbs_yearbook_image_manual_check"),
    (2024, 38712, 21464, 8128, 9118, "nbs_yearbook_image_manual_check"),
]


def build_official_township_national_series(
    output_csv: str | Path,
    nowcast_end_year: int = 2026,
    forecast_start_year: int = 2027,
    forecast_end_year: int = 2035,
) -> pd.DataFrame:
    rows = []
    for year, total, town, township, street, source in OFFICIAL_TOWNSHIP_NATIONAL_ROWS:
        rows.append(
            {
                "scope": "全国",
                "year": year,
                "township_street_count": total,
                "town_count": town,
                "township_count": township,
                "street_count": street,
                "other_township_level_count": total - town - township - street,
                "status_township_street_count": "observed",
                "source_township_street_count": source,
            }
        )

    observed = pd.DataFrame(rows).sort_values("year")
    latest = observed.iloc[-1].copy()
    nowcasts = []
    for year in range(int(latest["year"]) + 1, nowcast_end_year + 1):
        carried = latest.copy()
        carried["year"] = year
        carried["status_township_street_count"] = "nowcast"
        carried["source_township_street_count"] = f"carried_forward_from_nbs_yearbook_{int(latest['year'])}"
        nowcasts.append(carried)

    target_years = list(range(forecast_start_year, forecast_end_year + 1))
    forecast_rows = []
    years = observed["year"].astype(int).to_numpy()
    for idx, year in enumerate(target_years):
        row = {"scope": "全国", "year": year}
        for col in ["township_street_count", "town_count", "township_count", "street_count", "other_township_level_count"]:
            values = observed[col].astype(float).to_numpy()
            row[col] = _forecast_count_series(years, values, target_years)[idx]
        row["status_township_street_count"] = "forecast"
        row["source_township_street_count"] = "statistical_forecast_from_nbs_yearbook_2009_2024"
        forecast_rows.append(row)

    out = pd.concat([observed, pd.DataFrame(nowcasts), pd.DataFrame(forecast_rows)], ignore_index=True, sort=False)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return out


def _ring_signed_area(ring: list) -> float:
    area = 0.0
    for idx, point in enumerate(ring):
        prev = ring[idx - 1]
        area += prev[0] * point[1] - point[0] * prev[1]
    return area / 2.0


def _rewind_polygon_for_plotly(rings: list) -> list:
    if not rings:
        return rings
    rewound = []
    outer = list(rings[0])
    if _ring_signed_area(outer) > 0:
        outer = list(reversed(outer))
    rewound.append(outer)
    for ring in rings[1:]:
        hole = list(ring)
        if _ring_signed_area(hole) < 0:
            hole = list(reversed(hole))
        rewound.append(hole)
    return rewound


def _rewind_geometry_for_plotly(geometry: dict) -> dict:
    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    cleaned = dict(geometry)
    if geom_type == "Polygon":
        cleaned["coordinates"] = _rewind_polygon_for_plotly(coordinates)
    elif geom_type == "MultiPolygon":
        cleaned["coordinates"] = [_rewind_polygon_for_plotly(polygon) for polygon in coordinates]
    return cleaned


def _township_source_label(source: str) -> str:
    if not isinstance(source, str) or not source:
        return "未标注来源"
    if source.startswith("nbs_yearbook_province_admin_"):
        year = source.rsplit("_", 1)[-1]
        return f"国家统计局《中国统计年鉴》1-1 全国行政区划省级表（{year}年底）"
    if source == "gaohr_cn_streets_csv_from_nbs_statistical_division_codes":
        return "GaoHR 整理的国家统计局统计用区划代码乡级 CSV（2017-2023）"
    if source == "linear_interpolation_between_nbs_yearbook_and_statistical_division_counts":
        return "线性插值：连接官方年鉴省级表与统计用区划代码统计值"
    if source.startswith("carried_forward_from_"):
        return "现势估计：沿用最新统计用区划代码省级统计值"
    if source == "statistical_forecast_from_2009_2023_counts_with_external_regressors":
        return "统计预测：历史数量趋势 + 外部回归因子"
    if source.startswith("statistical_forecast_from_"):
        return "统计预测：基于历史省级数量序列的阻尼 Huber 趋势"
    return source


def _clean_geojson_for_locations(geojson: dict, valid_adcodes: set[int]) -> dict:
    """Keep only matched province features so the fixed projection has stable bounds."""

    features = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        try:
            adcode = int(props.get("adcode"))
        except (TypeError, ValueError):
            continue
        if adcode not in valid_adcodes or not props.get("name"):
            continue
        cleaned = dict(feature)
        cleaned_props = dict(props)
        cleaned_props["adcode"] = adcode
        cleaned["properties"] = cleaned_props
        cleaned["geometry"] = _rewind_geometry_for_plotly(feature.get("geometry", {}))
        features.append(cleaned)
    return {"type": "FeatureCollection", "features": features}


def _geojson_has_adcode(geojson: dict, adcode: int) -> bool:
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        try:
            if int(props.get("adcode")) == adcode and props.get("name"):
                return True
        except (TypeError, ValueError):
            continue
    return False


def build_quantity_atlas(
    township_panel: pd.DataFrame,
    natural_panel: pd.DataFrame,
    official_township_panel: pd.DataFrame,
    boundary_geojson: str | Path,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geojson = json.loads(Path(boundary_geojson).read_text(encoding="utf-8"))
    plot_df = township_panel.copy()
    plot_df["year"] = plot_df["year"].astype(str)
    plot_df["阶段"] = plot_df["status_township_street_count"].map(
        {"observed": "真实观测", "interpolated": "插值", "nowcast": "现势估计", "forecast": "预测"}
    ).fillna(plot_df["status_township_street_count"])
    plot_df["数据源"] = plot_df["source_township_street_count"].map(_township_source_label)
    hover_data = ["乡镇/街道总数", "街道数", "镇数", "乡数", "其他乡级单元数", "阶段", "数据源"]
    if "population_external" in plot_df.columns:
        plot_df["外部人口因子"] = pd.to_numeric(plot_df["population_external"], errors="coerce").round(0)
        hover_data.insert(-2, "外部人口因子")
    plot_df = plot_df.rename(
        columns={
            "province": "省份",
            "township_street_count": "乡镇/街道总数",
            "street_count": "街道数",
            "town_count": "镇数",
            "township_count": "乡数",
            "other_township_level_count": "其他乡级单元数",
        }
    )
    valid_adcodes = set(pd.to_numeric(plot_df["adcode"], errors="coerce").dropna().astype(int))
    include_taiwan_no_data = TAIWAN_ADCODE not in valid_adcodes and _geojson_has_adcode(geojson, TAIWAN_ADCODE)
    map_adcodes = set(valid_adcodes)
    if include_taiwan_no_data:
        map_adcodes.add(TAIWAN_ADCODE)
    map_geojson = _clean_geojson_for_locations(geojson, map_adcodes)
    color_min = 0
    color_max = int(math.ceil(float(plot_df["乡镇/街道总数"].max()) / 500.0) * 500)
    color_tick_step = 1000 if color_max >= 3000 else 500
    strong_color_scale = [
        [0.00, "#f7fbff"],
        [0.16, "#60a5fa"],
        [0.34, "#14b8a6"],
        [0.54, "#facc15"],
        [0.74, "#f97316"],
        [1.00, "#7f1d1d"],
    ]

    map_fig = px.choropleth(
        plot_df,
        geojson=map_geojson,
        locations="adcode",
        featureidkey="properties.adcode",
        color="乡镇/街道总数",
        animation_frame="year",
        hover_name="省份",
        hover_data=hover_data,
        color_continuous_scale=strong_color_scale,
        range_color=(color_min, color_max),
        projection="mercator",
    )
    if include_taiwan_no_data:
        taiwan_trace = go.Choropleth(
            geojson=map_geojson,
            locations=[TAIWAN_ADCODE],
            featureidkey="properties.adcode",
            z=[0],
            zmin=0,
            zmax=1,
            colorscale=[[0, NO_DATA_FILL_COLOR], [1, NO_DATA_FILL_COLOR]],
            marker_line_color=NO_DATA_LINE_COLOR,
            marker_line_width=0.8,
            showscale=False,
            showlegend=True,
            name="数据暂缺",
            hovertemplate=f"{TAIWAN_NAME}<br>乡镇/街道数量：数据暂缺<br>说明：底图展示，未纳入数量统计<extra></extra>",
        )
        map_fig.add_trace(taiwan_trace)
        for frame in map_fig.frames:
            frame.data = tuple(list(frame.data) + [go.Choropleth(taiwan_trace.to_plotly_json())])
            frame.traces = tuple(range(len(frame.data)))
    map_fig.update_geos(
        fitbounds=False,
        lonaxis_range=CHINA_MAP_LON_RANGE,
        lataxis_range=CHINA_MAP_LAT_RANGE,
        visible=False,
        bgcolor="rgba(0,0,0,0)",
        showframe=False,
        showcountries=False,
        showcoastlines=False,
        showland=False,
        showocean=False,
        lataxis_showgrid=False,
        lonaxis_showgrid=False,
    )
    for trace in map_fig.data:
        trace.update(zmin=color_min, zmax=color_max)
    for frame in map_fig.frames:
        for trace in frame.data:
            trace.update(zmin=color_min, zmax=color_max)
    map_fig.update_layout(
        title="全国乡镇/街道数量变化图谱（2009-2026基准，2027-2035预测）",
        font={"family": "Microsoft YaHei, SimHei, Arial", "size": 14},
        autosize=True,
        margin={"r": 24, "t": 64, "l": 24, "b": 28},
        height=920,
        paper_bgcolor="white",
        plot_bgcolor="white",
        coloraxis={
            "cmin": color_min,
            "cmax": color_max,
            "cauto": False,
            "colorbar": {
                "title": "数量（个）",
                "tickmode": "array",
                "tickvals": list(range(color_min, color_max + 1, color_tick_step)),
            },
        },
    )
    if map_fig.layout.sliders:
        map_fig.layout.sliders[0].currentvalue.prefix = "年份："
        map_fig.layout.sliders[0].currentvalue.font.size = 14
    if map_fig.layout.updatemenus:
        buttons = map_fig.layout.updatemenus[0].buttons
        if len(buttons) >= 1:
            buttons[0].label = "播放"
        if len(buttons) >= 2:
            buttons[1].label = "暂停"

    national = official_township_panel.copy()
    national["阶段"] = national["status_township_street_count"].map(
        {"observed": "真实观测", "nowcast": "现势估计", "forecast": "预测"}
    ).fillna("未知")

    dash = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["全国乡镇/街道数量结构", "全国自然村数量变化（农业普查口径）"],
        vertical_spacing=0.13,
    )
    dash.add_trace(go.Scatter(x=national["year"], y=national["township_street_count"], name="乡镇/街道总数", mode="lines+markers"), row=1, col=1)
    dash.add_trace(go.Scatter(x=national["year"], y=national["street_count"], name="街道数", mode="lines"), row=1, col=1)
    dash.add_trace(go.Scatter(x=national["year"], y=national["town_count"], name="镇数", mode="lines"), row=1, col=1)
    dash.add_trace(go.Scatter(x=national["year"], y=national["township_count"], name="乡数", mode="lines"), row=1, col=1)

    natural = natural_panel.copy()
    natural["阶段"] = natural["status_natural_village_count"].map(
        {
            "observed": "真实观测",
            "derived_observed": "由官方比例推算",
            "interpolated": "插值",
            "nowcast": "现势估计",
            "forecast": "预测",
        }
    )
    dash.add_trace(
        go.Scatter(
            x=natural["year"],
            y=natural["natural_village_count"],
            name="自然村数量",
            mode="lines+markers",
            customdata=natural[["阶段"]],
            hovertemplate="年份：%{x}<br>自然村数量：%{y:,} 个<br>阶段：%{customdata[0]}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    dash.add_vrect(x0=2026.5, x1=2035.5, fillcolor="LightSalmon", opacity=0.16, line_width=0, annotation_text="预测期", annotation_position="top left")
    dash.update_layout(
        title="数量趋势诊断：乡镇/街道与自然村",
        font={"family": "Microsoft YaHei, SimHei, Arial", "size": 14},
        template="plotly_white",
        autosize=True,
        height=930,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.08},
    )
    dash.update_yaxes(title_text="数量（个）", row=1, col=1)
    dash.update_yaxes(
        title_text="数量（个）",
        range=[2_900_000, 3_300_000],
        dtick=100_000,
        row=2,
        col=1,
    )

    map_path = output_dir / "乡镇街道数量地图.html"
    dashboard_path = output_dir / "数量趋势仪表盘.html"
    atlas_path = output_dir / "数量变化图谱.html"
    year_start = int(pd.to_numeric(plot_df["year"], errors="coerce").min())
    year_end = int(pd.to_numeric(plot_df["year"], errors="coerce").max())
    forecast_start = int(pd.to_numeric(plot_df.loc[plot_df["阶段"].eq("预测"), "year"], errors="coerce").min())
    province_count = int(plot_df["省份"].nunique())
    latest_national = national.sort_values("year").iloc[-1]
    latest_natural = natural.sort_values("year").iloc[-1]
    preview_note = f"""
    <section class="hero">
      <div>
        <h1>全国乡镇/街道与自然村数量变化图谱</h1>
        <p class="lead">省级地图展示乡镇/街道数量，趋势图展示全国乡级行政区划结构和自然村数量变化。预测期从 {forecast_start} 年开始。</p>
      </div>
    </section>
    <section class="stats">
      <div><span>{province_count}</span><small>统计省级单元</small></div>
      <div><span>{year_start}-{year_end}</span><small>覆盖年份</small></div>
      <div><span>{int(latest_national["township_street_count"]):,}</span><small>{int(latest_national["year"])} 乡镇/街道</small></div>
      <div><span>{int(latest_natural["natural_village_count"]):,}</span><small>{int(latest_natural["year"])} 自然村</small></div>
    </section>
    <section class="method-strip">
      <span>2009-2013 官方省级年鉴</span>
      <span>2014-2016 线性插值</span>
      <span>2017-2023 统计用区划代码</span>
      <span>2024-2026 现势估计</span>
      <span>2027-2035 统计预测</span>
    </section>
    """
    source_note = """
    <details class="source-panel">
      <summary>数据源与口径说明</summary>
      <p>原始来源链接：
        <a href="https://www.stats.gov.cn/sj/ndsj/" target="_blank">国家统计局《中国统计年鉴》</a>；
        <a href="https://gaohr.win/site/blogs/2020/2020-08-10-china-villages.html" target="_blank">GaoHR 历年统计用区划代码整理</a>；
        <a href="https://www.stats.gov.cn/sj/tjgb/nypcgb/qgnypcgb/202302/t20230206_1902101.html" target="_blank">国家统计局第三次全国农业普查公报/解读</a>；
        <a href="https://app.www.gov.cn/govdata/gov/201712/15/416657/article.html" target="_blank">国务院数据转载自然村减少比例</a>；
        <a href="https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php" target="_blank">GHSL R2023A 人口栅格</a>；
        <a href="https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json" target="_blank">DataV 省界 GeoJSON</a>。
      </p>
      <table>
        <thead><tr>
          <th>指标/时期</th>
          <th>数据源</th>
          <th>说明</th>
        </tr></thead>
        <tbody>
          <tr><td>省级乡镇/街道 2009-2013</td><td>国家统计局《中国统计年鉴》1-1 全国行政区划</td><td>官方省级表，逐省读取乡镇级区划数、镇、乡级、街道办事处。</td></tr>
          <tr><td>省级乡镇/街道 2014-2016</td><td>线性插值</td><td>连接 2013 官方省级表与 2017 统计用区划代码统计值，仅用于逐年动画连续显示。</td></tr>
          <tr><td>省级乡镇/街道 2017-2023</td><td>GaoHR 整理的国家统计局统计用区划代码乡级 CSV</td><td>按乡级代码逐条统计到省，包含镇、乡、街道及其他乡级单元。</td></tr>
          <tr><td>省级乡镇/街道 2024-2026</td><td>现势估计</td><td>因 2026 年完整年度区划数据尚未发布，省级地图沿用最新可核验 2023 乡级代码统计值。</td></tr>
          <tr><td>省级乡镇/街道 2027-2035</td><td>统计预测</td><td>使用历史省级数量序列进行阻尼 Huber 趋势预测；人口外部因子可用时，加入人口回归修正并进行限幅融合。</td></tr>
          <tr><td>外部回归因子：人口</td><td>GHSL R2023A 30 弧秒人口栅格</td><td>按省界把 GHSL 网格人口汇总到省；用于 2027-2035 乡镇/街道数量预测的外部人口回归修正。GDP 接口已预留，未取得可靠省级 GDP CSV 时不启用。</td></tr>
          <tr><td>全国乡级行政区划 2009-2024</td><td>国家统计局《中国统计年鉴》全国行政区划表</td><td>用于趋势图中的全国乡镇/街道总数、镇数、乡数、街道数。</td></tr>
          <tr><td>自然村数量</td><td>国家统计局第三次全国农业普查解读、国务院数据转载</td><td>2016 年末自然村 317 万个；2006 年按“比 2006 年减少 3.8%”推算；其余年份按该两点趋势插值/估计/预测。</td></tr>
          <tr><td>台湾省边界</td><td>阿里云 DataV/高德边界服务 100000_full GeoJSON</td><td>底图保留台湾省边界；因缺少与本表同口径的乡镇/街道年度数量序列，地图标注为“数据暂缺”，不参与统计色阶和数量汇总。</td></tr>
          <tr><td>省界底图</td><td>阿里云 DataV/高德边界服务 100000_full GeoJSON</td><td>用于省级面图显示，不参与数量计算。</td></tr>
        </tbody>
      </table>
    </details>
    """
    plotly_config = {"responsive": True, "displaylogo": False}
    map_html = to_html(map_fig, include_plotlyjs="cdn", full_html=False, config=plotly_config)
    dash_html = to_html(dash, include_plotlyjs=False, full_html=False, config=plotly_config)
    map_fig.write_html(map_path, include_plotlyjs="cdn", config=plotly_config)
    dash.write_html(dashboard_path, include_plotlyjs="cdn", config=plotly_config)
    atlas_path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>数量变化图谱</title>
  <style>
    :root {{
      --bg: #ffffff;
      --panel: #ffffff;
      --text: #172033;
      --muted: #687386;
      --line: #d9e0ea;
      --blue: #2f6fed;
      --blue-dark: #1f4fb3;
      --teal: #0f9f8f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", "Segoe UI", sans-serif;
      line-height: 1.55;
      letter-spacing: 0;
    }}
    a {{ color: var(--blue); text-decoration: none; }}
    a:hover {{ color: var(--blue-dark); text-decoration: underline; }}
    .page {{
      width: 100%;
      max-width: none;
      margin: 0;
      padding: 30px 34px 48px;
      background: #fff;
    }}
    .hero {{
      padding: 0 0 18px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0;
      font-size: clamp(28px, 4vw, 44px);
      line-height: 1.16;
    }}
    .lead {{
      max-width: 860px;
      margin: 14px 0 0;
      color: var(--muted);
      font-size: 17px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      padding: 16px 0 2px;
    }}
    .stats div {{
      min-height: 78px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .stats span {{
      display: block;
      font-size: 26px;
      font-weight: 780;
      line-height: 1.1;
    }}
    .stats small {{
      display: block;
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .method-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      padding: 18px 0 8px;
    }}
    .method-strip span {{
      padding: 7px 10px;
      border: 1px solid #cfe0ff;
      border-radius: 999px;
      background: #ffffff;
      color: #315170;
      font-size: 13px;
      font-weight: 650;
    }}
    .viz-panel {{
      margin-top: 18px;
      padding: 18px 16px 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: visible;
      box-shadow: 0 1px 2px rgba(15, 29, 51, 0.04);
    }}
    .viz-panel + .viz-panel {{ margin-top: 22px; }}
    .viz-panel .plotly-graph-div,
    .viz-panel .plot-container,
    .viz-panel .svg-container {{
      width: 100% !important;
      max-width: none !important;
    }}
    .viz-panel .svg-container,
    .viz-panel .main-svg {{
      overflow: visible !important;
    }}
    .map-panel {{
      min-height: 980px;
    }}
    .trend-panel {{
      min-height: 990px;
    }}
    .source-panel {{
      margin-top: 24px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 16px 18px;
    }}
    .source-panel summary {{
      cursor: pointer;
      font-size: 18px;
      font-weight: 760;
    }}
    .source-panel p {{ color: var(--muted); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid #edf1f6;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: #33445c;
      background: #f8fafc;
      font-weight: 760;
    }}
    @media (max-width: 760px) {{
      .page {{ padding: 22px 16px 40px; }}
      .stats {{ grid-template-columns: 1fr 1fr; }}
      .viz-panel {{
        overflow-x: auto;
        overflow-y: visible;
      }}
      .map-panel > div,
      .trend-panel > div,
      .viz-panel .plotly-graph-div {{
        min-width: 760px;
      }}
      .map-panel {{ min-height: 780px; }}
      .trend-panel {{ min-height: 900px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    {preview_note}
    <section class="viz-panel map-panel">{map_html}</section>
    <section class="viz-panel trend-panel">{dash_html}</section>
    {source_note}
  </main>
  <script>
    window.resizePlotlyCharts = function resizePlotlyCharts() {{
      if (!window.Plotly) return;
      document.querySelectorAll('.js-plotly-plot').forEach(plot => {{
        window.Plotly.Plots.resize(plot);
      }});
    }};
    window.addEventListener('load', () => {{
      window.resizePlotlyCharts();
      window.setTimeout(window.resizePlotlyCharts, 150);
    }});
    window.addEventListener('resize', window.resizePlotlyCharts);
  </script>
</body>
</html>""",
        encoding="utf-8",
    )
