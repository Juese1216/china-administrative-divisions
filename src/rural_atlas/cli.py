from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .atlas import build_dashboard, build_map
from .admin import build_modood_2023_units
from .download import fetch_modood_2023
from .ghsl import (
    aggregate_ghsl_population_tif,
    download_ghsl_population,
    extract_first_tif,
    make_population_nowcast,
)
from .forecast import ForecastSettings, forecast_panel
from .io import (
    metric_names_from_config,
    metric_weights_from_config,
    read_panel,
    read_toml,
    write_csv,
    write_json,
    write_template,
)
from .metrics import add_indices, add_yearly_changes, expand_annual_panel, national_summary, row_stage
from .quantity import (
    add_township_forecast,
    build_official_township_national_series,
    build_natural_village_series,
    build_population_external_regressors,
    build_quantity_atlas,
    build_township_street_counts,
)
from .validate import validate_panel
from .zonal import aggregate_raster_to_units


DEFAULT_CONFIG = Path("config/rural_time_series_project.toml")


def _load_config(path: str | Path) -> dict[str, Any]:
    return read_toml(path)


def _settings_from_config(config: dict[str, Any], args: argparse.Namespace) -> ForecastSettings:
    forecast_cfg = config.get("forecast", {})
    project_cfg = config.get("project", {})
    return ForecastSettings(
        forecast_start_year=args.forecast_start_year or int(project_cfg.get("forecast_start_year", 2027)),
        forecast_end_year=args.forecast_end_year or int(project_cfg.get("forecast_end_year", 2035)),
        damping=float(forecast_cfg.get("damping", 0.90)),
        max_abs_log_growth_per_year=float(forecast_cfg.get("max_abs_log_growth_per_year", 0.20)),
        min_points_for_trend=int(forecast_cfg.get("min_points_for_trend", 4)),
        prediction_interval_z=float(forecast_cfg.get("prediction_interval_z", 1.96)),
        use_interpolated_for_forecast=bool(
            args.use_interpolated_for_forecast or forecast_cfg.get("use_interpolated_for_forecast", False)
        ),
    )


def _print_report(report) -> None:
    if report.errors:
        print("ERRORS:")
        for item in report.errors:
            print(f"  - {item}")
    if report.warnings:
        print("WARNINGS:")
        for item in report.warnings:
            print(f"  - {item}")
    if report.ok:
        print("Validation passed.")


def command_make_template(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    metrics = args.metrics or metric_names_from_config(config)
    write_template(args.output, metrics)
    print(f"Wrote template: {args.output}")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    project = config.get("project", {})
    metrics = args.metrics or metric_names_from_config(config)
    df = read_panel(args.input)
    report = validate_panel(
        df,
        metrics,
        observed_start_year=int(project.get("observed_start_year", 1999)),
        baseline_year=int(project.get("baseline_year", 2026)),
        forecast_start_year=int(project.get("forecast_start_year", 2027)),
        require_sources=not args.allow_missing_source,
    )
    _print_report(report)
    return 0 if report.ok else 2


def command_run(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    project = config.get("project", {})
    atlas_cfg = config.get("atlas", {})
    metrics = args.metrics or metric_names_from_config(config)
    weights = metric_weights_from_config(config, metrics)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    observed_start_year = args.start_year or int(project.get("observed_start_year", 1999))
    baseline_year = args.baseline_year or int(project.get("baseline_year", 2026))
    forecast_start_year = args.forecast_start_year or int(project.get("forecast_start_year", 2027))

    if forecast_start_year <= baseline_year:
        raise ValueError("forecast_start_year must be greater than baseline_year.")

    raw = read_panel(args.input)
    report = validate_panel(
        raw,
        metrics,
        observed_start_year=observed_start_year,
        baseline_year=baseline_year,
        forecast_start_year=forecast_start_year,
        require_sources=not args.allow_missing_source,
    )
    _print_report(report)
    if report.errors:
        return 2

    annual = expand_annual_panel(raw, metrics, observed_start_year, baseline_year)
    annual = add_yearly_changes(annual, metrics)
    annual = add_indices(annual, metrics, weights)

    settings = _settings_from_config(config, args)
    forecast = forecast_panel(annual, metrics, settings)
    if forecast.empty:
        raise ValueError("No forecast rows were produced. Check metric history and unit_id coverage.")
    combined = pd.concat([annual, forecast], ignore_index=True, sort=False)
    combined["row_stage"] = row_stage(combined, metrics)
    combined = add_yearly_changes(combined, metrics)
    combined = add_indices(combined, metrics, weights)
    summary = national_summary(combined, metrics)

    annual_name = f"panel_annual_{observed_start_year}_{baseline_year}.csv"
    forecast_name = f"panel_forecast_{settings.forecast_start_year}_{settings.forecast_end_year}.csv"
    full_name = f"panel_full_{observed_start_year}_{settings.forecast_end_year}.csv"
    write_csv(annual, output_dir / annual_name)
    write_csv(forecast, output_dir / forecast_name)
    write_csv(combined, output_dir / full_name)
    write_csv(summary, output_dir / "national_summary.csv")

    build_map(
        combined,
        output_dir / "atlas.html",
        color_column=args.color_column or atlas_cfg.get("default_color_column", "change_intensity_index"),
        size_column=args.size_column or atlas_cfg.get("default_size_column", "vitality_index"),
        geojson_path=args.geojson,
        geojson_id_property=args.geojson_id_property,
        map_style=atlas_cfg.get("map_style", "carto-positron"),
        center_lat=float(atlas_cfg.get("center_lat", 35.0)),
        center_lon=float(atlas_cfg.get("center_lon", 104.0)),
        zoom=float(atlas_cfg.get("zoom", 3.2)),
    )
    build_dashboard(combined, summary, output_dir / "dashboard.html", metrics)

    metadata = {
        "input": str(args.input),
        "geojson": str(args.geojson) if args.geojson else None,
        "metrics": metrics,
        "weights": weights,
        "observed_start_year": observed_start_year,
        "baseline_year": baseline_year,
        "forecast_start_year": settings.forecast_start_year,
        "forecast_end_year": settings.forecast_end_year,
        "forecast_method": "damped Huber log-linear trend with prediction intervals",
        "outputs": [
            annual_name,
            forecast_name,
            full_name,
            "national_summary.csv",
            "atlas.html",
            "dashboard.html",
        ],
    }
    write_json(metadata, output_dir / "run_metadata.json")
    print(f"Wrote outputs to {output_dir.resolve()}")
    return 0


def command_zonal_raster(args: argparse.Namespace) -> int:
    aggregate_raster_to_units(
        units_geojson=args.units_geojson,
        raster_path=args.raster,
        output_csv=args.output,
        metric=args.metric,
        year=args.year,
        source_id=args.source_id,
        unit_id_column=args.unit_id_column,
        stats=tuple(args.stats),
    )
    print(f"Wrote raster aggregation: {args.output}")
    return 0


def command_build_admin(args: argparse.Namespace) -> int:
    units = build_modood_2023_units(args.raw_dir, args.output)
    print(f"Wrote {len(units):,} administrative units: {args.output}")
    return 0


def command_fetch_admin_2023(args: argparse.Namespace) -> int:
    files = fetch_modood_2023(args.output_dir, overwrite=args.overwrite)
    for path in files:
        print(f"{path} ({path.stat().st_size:,} bytes)")
    return 0


def command_fetch_ghsl_population(args: argparse.Namespace) -> int:
    for year in args.years:
        zip_path = download_ghsl_population(year, args.output_dir, overwrite=args.overwrite)
        print(f"{zip_path} ({zip_path.stat().st_size:,} bytes)")
        if args.extract:
            tif_path = extract_first_tif(zip_path, Path(args.output_dir) / "extracted")
            print(f"  extracted {tif_path} ({tif_path.stat().st_size:,} bytes)")
    return 0


def command_aggregate_ghsl_population(args: argparse.Namespace) -> int:
    tif_path = args.tif
    if args.zip:
        tif_path = extract_first_tif(args.zip, Path(args.output).parent / "extracted")
    df = aggregate_ghsl_population_tif(
        tif_path=tif_path,
        boundary_geojson=args.boundary_geojson,
        year=args.year,
        output_csv=args.output,
        resolution_degrees=args.resolution,
    )
    print(f"Wrote {len(df):,} grid rows: {args.output}")
    return 0


def command_nowcast_population(args: argparse.Namespace) -> int:
    panel = pd.read_csv(args.input, dtype={"unit_id": "string"})
    out = make_population_nowcast(panel, args.source_year, args.target_year, args.output)
    print(f"Wrote {len(out):,} nowcast rows: {args.output}")
    return 0


def command_quantity_atlas(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    external_panel = pd.DataFrame()
    if args.population_external_panel and Path(args.population_external_panel).exists():
        external_panel = build_population_external_regressors(
            population_panel_csv=args.population_external_panel,
            boundary_geojson=args.boundary_geojson,
            output_csv=output_dir / "外部回归因子_人口_2000_2035.csv",
        )
    if args.economic_external_csv and Path(args.economic_external_csv).exists():
        economic = pd.read_csv(args.economic_external_csv)
        if "unit_id" not in economic.columns and "adcode" in economic.columns:
            economic["unit_id"] = economic["adcode"].astype(int).astype(str)
        economic["unit_id"] = economic["unit_id"].astype(str)
        economic["year"] = economic["year"].astype(int)
        keep = [
            col
            for col in [
                "unit_id",
                "year",
                "gdp_external",
                "gdp_per_capita_external",
                "source_gdp_external",
                "status_gdp_external",
            ]
            if col in economic.columns
        ]
        economic = economic[keep].drop_duplicates(["unit_id", "year"])
        external_panel = economic if external_panel.empty else external_panel.merge(economic, on=["unit_id", "year"], how="outer")
    if not external_panel.empty:
        external_panel.to_csv(output_dir / "外部回归因子_合并_2000_2035.csv", index=False, encoding="utf-8-sig")
    township_base = build_township_street_counts(
        streets_dir=args.streets_dir,
        boundary_geojson=args.boundary_geojson,
        output_csv=output_dir / "乡镇街道数量_2009_2026.csv",
        nowcast_end_year=args.baseline_year,
        yearbook_province_csv=args.yearbook_province_csv,
    )
    township_full = add_township_forecast(
        township_base,
        output_csv=output_dir / "乡镇街道数量_2009_2035.csv",
        forecast_start_year=args.forecast_start_year,
        forecast_end_year=args.forecast_end_year,
        external_panel=external_panel,
    )
    natural = build_natural_village_series(output_dir / "自然村数量_2006_2035.csv")
    official_township = build_official_township_national_series(
        output_dir / "官方年鉴_乡级行政区划数量_2009_2035.csv",
        nowcast_end_year=args.baseline_year,
        forecast_start_year=args.forecast_start_year,
        forecast_end_year=args.forecast_end_year,
    )
    build_quantity_atlas(township_full, natural, official_township, args.boundary_geojson, output_dir)
    print(f"已生成数量图谱：{(output_dir / '数量变化图谱.html').resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a real-data China rural time-series atlas.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=DEFAULT_CONFIG, help="Project TOML config.")
    common.add_argument("--metrics", nargs="*", help="Metric columns to use. Defaults to config metrics.")

    p_template = sub.add_parser("make-template", parents=[common], help="Create an empty observed panel CSV template.")
    p_template.add_argument("--output", default="data/source/rural_time_series/raw/panel_observed_template.csv")
    p_template.set_defaults(func=command_make_template)

    p_validate = sub.add_parser("validate", parents=[common], help="Validate observed data and source labels.")
    p_validate.add_argument("--input", required=True)
    p_validate.add_argument("--allow-missing-source", action="store_true")
    p_validate.set_defaults(func=command_validate)

    p_run = sub.add_parser("run", parents=[common], help="Build annual comparison, forecast, map, and dashboard.")
    p_run.add_argument("--input", required=True, help="Observed annual panel CSV.")
    p_run.add_argument("--geojson", help="Optional unit polygons GeoJSON.")
    p_run.add_argument("--geojson-id-property", default="unit_id")
    p_run.add_argument("--output-dir", default="data/processed/rural_time_series/general_atlas")
    p_run.add_argument("--start-year", type=int)
    p_run.add_argument("--baseline-year", type=int)
    p_run.add_argument("--forecast-start-year", type=int)
    p_run.add_argument("--forecast-end-year", type=int)
    p_run.add_argument("--color-column")
    p_run.add_argument("--size-column")
    p_run.add_argument("--allow-missing-source", action="store_true")
    p_run.add_argument("--use-interpolated-for-forecast", action="store_true")
    p_run.set_defaults(func=command_run)

    p_zonal = sub.add_parser(
        "zonal-raster",
        parents=[common],
        help="Aggregate a real raster source to unit polygons.",
    )
    p_zonal.add_argument("--units-geojson", required=True)
    p_zonal.add_argument("--raster", required=True)
    p_zonal.add_argument("--output", required=True)
    p_zonal.add_argument("--metric", required=True)
    p_zonal.add_argument("--year", type=int, required=True)
    p_zonal.add_argument("--source-id", required=True)
    p_zonal.add_argument("--unit-id-column", default="unit_id")
    p_zonal.add_argument("--stats", nargs="+", default=["mean"])
    p_zonal.set_defaults(func=command_zonal_raster)

    p_admin = sub.add_parser(
        "build-admin",
        parents=[common],
        help="Build a normalized 2023 administrative hierarchy from downloaded CSV files.",
    )
    p_admin.add_argument("--raw-dir", default="data/source/rural_time_series/raw")
    p_admin.add_argument("--output", default="data/processed/rural_time_series/admin_units_2023.csv")
    p_admin.set_defaults(func=command_build_admin)

    p_fetch_admin = sub.add_parser(
        "fetch-admin-2023",
        parents=[common],
        help="Download the 2023 five-level administrative CSV files from the modood mirror.",
    )
    p_fetch_admin.add_argument("--output-dir", default="data/source/rural_time_series/raw")
    p_fetch_admin.add_argument("--overwrite", action="store_true")
    p_fetch_admin.set_defaults(func=command_fetch_admin_2023)

    p_fetch_ghsl = sub.add_parser(
        "fetch-ghsl-population",
        parents=[common],
        help="Download GHSL R2023A 30 arc-second population zip files.",
    )
    p_fetch_ghsl.add_argument("--years", nargs="+", type=int, required=True)
    p_fetch_ghsl.add_argument("--output-dir", default="data/source/rural_time_series/raw/ghsl")
    p_fetch_ghsl.add_argument("--overwrite", action="store_true")
    p_fetch_ghsl.add_argument("--extract", action="store_true")
    p_fetch_ghsl.set_defaults(func=command_fetch_ghsl_population)

    p_agg_ghsl = sub.add_parser(
        "aggregate-ghsl-population",
        parents=[common],
        help="Aggregate a GHSL population GeoTIFF to China coarse grid points.",
    )
    p_agg_ghsl.add_argument("--tif")
    p_agg_ghsl.add_argument("--zip")
    p_agg_ghsl.add_argument("--boundary-geojson", required=True)
    p_agg_ghsl.add_argument("--year", type=int, required=True)
    p_agg_ghsl.add_argument("--resolution", type=float, default=0.5)
    p_agg_ghsl.add_argument("--output", required=True)
    p_agg_ghsl.set_defaults(func=command_aggregate_ghsl_population)

    p_nowcast = sub.add_parser(
        "nowcast-population",
        parents=[common],
        help="Create a labelled baseline nowcast by carrying latest GHSL population cells forward.",
    )
    p_nowcast.add_argument("--input", required=True)
    p_nowcast.add_argument("--source-year", type=int, required=True)
    p_nowcast.add_argument("--target-year", type=int, required=True)
    p_nowcast.add_argument("--output", required=True)
    p_nowcast.set_defaults(func=command_nowcast_population)

    p_quantity = sub.add_parser(
        "quantity-atlas",
        parents=[common],
        help="生成中文数量变化图谱：乡镇/街道数量 + 自然村数量趋势。",
    )
    p_quantity.add_argument("--streets-dir", default="data/source/rural_time_series/raw/gaohr/extracted")
    p_quantity.add_argument("--boundary-geojson", default="data/source/rural_time_series/raw/china_provinces_datav_100000_full.json")
    p_quantity.add_argument(
        "--yearbook-province-csv",
        default="data/source/rural_time_series/raw/stats_yearbook/official_province_admin_2009_2013.csv",
    )
    p_quantity.add_argument(
        "--population-external-panel",
        default="data/processed/rural_time_series/ghsl_population_atlas/panel_full_2000_2035.csv",
        help="可选：GHSL 人口面板，用于构建省级人口外部回归因子。",
    )
    p_quantity.add_argument(
        "--economic-external-csv",
        help="可选：省级 GDP 等经济外部因子 CSV，至少包含 unit_id/adcode、year、gdp_external。",
    )
    p_quantity.add_argument("--output-dir", default="data/processed/rural_time_series/quantity_atlas")
    p_quantity.add_argument("--baseline-year", type=int, default=2026)
    p_quantity.add_argument("--forecast-start-year", type=int, default=2027)
    p_quantity.add_argument("--forecast-end-year", type=int, default=2035)
    p_quantity.set_defaults(func=command_quantity_atlas)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
