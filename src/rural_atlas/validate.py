from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def validate_panel(
    df: pd.DataFrame,
    metrics: list[str],
    observed_start_year: int,
    baseline_year: int,
    forecast_start_year: int,
    require_sources: bool = True,
) -> ValidationReport:
    report = ValidationReport()

    for col in ["unit_id", "year"]:
        if col not in df.columns:
            report.add_error(f"Missing required column: {col}")

    if report.errors:
        return report

    if df.empty:
        report.add_error("Input panel is empty. Fill it with real observed rows before running the atlas.")
        return report

    duplicate_count = int(df.duplicated(["unit_id", "year"]).sum())
    if duplicate_count:
        report.add_error(f"Found {duplicate_count} duplicated unit_id-year rows.")

    if df["year"].min() > observed_start_year:
        report.add_warning(
            f"Earliest input year is {int(df['year'].min())}; requested history starts at {observed_start_year}."
        )
    if df["year"].max() < baseline_year:
        report.add_warning(
            f"Latest input year is {int(df['year'].max())}; 2026 baseline will be unavailable unless you add data."
        )
    if (df["year"] >= forecast_start_year).any():
        report.add_error(
            f"Input panel contains year >= {forecast_start_year}. Put model outputs in outputs/, not raw observations."
        )

    missing_metrics = [metric for metric in metrics if metric not in df.columns]
    if missing_metrics:
        report.add_error(f"Missing metric columns: {', '.join(missing_metrics)}")

    for metric in metrics:
        if metric not in df.columns:
            continue
        converted = pd.to_numeric(df[metric], errors="coerce")
        bad_numeric = int(df[metric].notna().sum() - converted.notna().sum())
        if bad_numeric:
            report.add_error(f"Metric {metric} has {bad_numeric} non-numeric values.")
        if (converted.dropna() < 0).any():
            report.add_warning(f"Metric {metric} has negative values; confirm the metric definition.")

        if require_sources:
            source_col = f"source_{metric}"
            has_shared_source = "source" in df.columns
            if source_col not in df.columns and not has_shared_source:
                report.add_error(
                    f"Metric {metric} needs a source column: {source_col} or shared column source."
                )
            else:
                source_series = df[source_col] if source_col in df.columns else df["source"]
                missing_source = int((converted.notna() & source_series.isna()).sum())
                blank_source = int((converted.notna() & (source_series.astype("string").str.strip() == "")).sum())
                if missing_source + blank_source:
                    report.add_error(
                        f"Metric {metric} has {missing_source + blank_source} values without source labels."
                    )

    counts = df.groupby("unit_id")["year"].nunique()
    low_history = int((counts < 4).sum())
    if low_history:
        report.add_warning(
            f"{low_history} units have fewer than 4 observed years; their forecasts will use weak-history fallback."
        )

    return report
