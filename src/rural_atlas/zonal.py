from __future__ import annotations

from pathlib import Path

import pandas as pd


def aggregate_raster_to_units(
    units_geojson: str | Path,
    raster_path: str | Path,
    output_csv: str | Path,
    metric: str,
    year: int,
    source_id: str,
    unit_id_column: str = "unit_id",
    stats: tuple[str, ...] = ("mean",),
) -> None:
    """Aggregate one raster to unit polygons.

    This function intentionally lives behind optional dependencies because
    rasterio/geopandas are heavy on Windows. Install with:
    pip install -e .[geo]
    """

    try:
        import geopandas as gpd
        from rasterstats import zonal_stats
    except ImportError as exc:
        raise RuntimeError(
            "Raster aggregation needs optional geospatial dependencies. "
            "Install them with: pip install -e .[geo]"
        ) from exc

    units = gpd.read_file(units_geojson)
    if unit_id_column not in units.columns:
        raise ValueError(f"GeoJSON is missing unit id column: {unit_id_column}")

    stats_rows = zonal_stats(
        vectors=units,
        raster=str(raster_path),
        stats=list(stats),
        geojson_out=False,
        nodata=None,
    )
    stat_name = "mean" if "mean" in stats else stats[0]
    values = [row.get(stat_name) for row in stats_rows]

    out = pd.DataFrame(
        {
            "unit_id": units[unit_id_column].astype("string"),
            "year": int(year),
            metric: values,
            f"source_{metric}": source_id,
            f"status_{metric}": "observed",
        }
    )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

