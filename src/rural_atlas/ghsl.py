from __future__ import annotations

import json
import math
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import tifffile
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.validation import make_valid


GHSL_POP_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_POP_GLOBE_R2023A/GHS_POP_E{year}_GLOBE_R2023A_4326_30ss/"
    "V1-0/GHS_POP_E{year}_GLOBE_R2023A_4326_30ss_V1_0.zip"
)


def download_ghsl_population(year: int, output_dir: str | Path, overwrite: bool = False) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"GHS_POP_E{year}_GLOBE_R2023A_4326_30ss_V1_0.zip"
    if target.exists() and target.stat().st_size > 100_000_000 and not overwrite:
        return target

    url = GHSL_POP_URL.format(year=year)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as response, target.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return target


def extract_first_tif(zip_path: str | Path, output_dir: str | Path) -> Path:
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as zf:
        tif_names = [name for name in zf.namelist() if name.lower().endswith((".tif", ".tiff"))]
        if not tif_names:
            raise ValueError(f"No GeoTIFF found in {zip_path}")
        name = tif_names[0]
        target = output_dir / Path(name).name
        if not target.exists():
            zf.extract(name, output_dir)
            extracted = output_dir / name
            if extracted != target:
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted.replace(target)
        return target


def _geometry_union(geojson_path: str | Path):
    data = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    geoms = [make_valid(shape(feature["geometry"])) for feature in data["features"]]
    return unary_union(geoms)


def _tiff_geo(page: tifffile.TiffPage) -> tuple[float, float, float, float]:
    scale = page.tags["ModelPixelScaleTag"].value
    tie = page.tags["ModelTiepointTag"].value
    x0 = float(tie[3])
    y0 = float(tie[4])
    dx = float(scale[0])
    dy = float(scale[1])
    return x0, y0, dx, dy


def _pixel_window(bounds: tuple[float, float, float, float], x0: float, y0: float, dx: float, dy: float):
    minx, miny, maxx, maxy = bounds
    col0 = max(0, int(math.floor((minx - x0) / dx)) - 1)
    col1 = int(math.ceil((maxx - x0) / dx)) + 1
    row0 = max(0, int(math.floor((y0 - maxy) / dy)) - 1)
    row1 = int(math.ceil((y0 - miny) / dy)) + 1
    return row0, row1, col0, col1


def aggregate_ghsl_population_tif(
    tif_path: str | Path,
    boundary_geojson: str | Path,
    year: int,
    output_csv: str | Path,
    resolution_degrees: float = 0.5,
) -> pd.DataFrame:
    """Aggregate GHSL 30 arc-second population to a coarse China grid."""

    china = _geometry_union(boundary_geojson)
    prepared = prep(china)
    bounds = china.bounds

    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]
        x0, y0, dx, dy = _tiff_geo(page)
        row0, row1, col0, col1 = _pixel_window(bounds, x0, y0, dx, dy)
        row1 = min(row1, page.imagelength)
        col1 = min(col1, page.imagewidth)

        lon_min = math.floor(bounds[0] / resolution_degrees) * resolution_degrees
        lon_max = math.ceil(bounds[2] / resolution_degrees) * resolution_degrees
        lat_min = math.floor(bounds[1] / resolution_degrees) * resolution_degrees
        lat_max = math.ceil(bounds[3] / resolution_degrees) * resolution_degrees
        n_lon = int(round((lon_max - lon_min) / resolution_degrees))
        n_lat = int(round((lat_max - lat_min) / resolution_degrees))
        sums = np.zeros((n_lat, n_lon), dtype="float64")

        tile_h = int(page.tilelength)
        tile_w = int(page.tilewidth)
        tiles_across = int(math.ceil(page.imagewidth / tile_w))
        tile_row0 = row0 // tile_h
        tile_row1 = (row1 - 1) // tile_h
        tile_col0 = col0 // tile_w
        tile_col1 = (col1 - 1) // tile_w

        fh = tif.filehandle
        for tr in range(tile_row0, tile_row1 + 1):
            for tc in range(tile_col0, tile_col1 + 1):
                tile_index = tr * tiles_across + tc
                offset = page.dataoffsets[tile_index]
                bytecount = page.databytecounts[tile_index]
                fh.seek(offset)
                encoded = fh.read(bytecount)
                decoded, _, _ = page.decode(encoded, tile_index, jpegtables=page.jpegtables)
                tile = np.squeeze(decoded)

                global_r0 = tr * tile_h
                global_c0 = tc * tile_w
                r_start = max(row0, global_r0)
                r_end = min(row1, global_r0 + tile.shape[0])
                c_start = max(col0, global_c0)
                c_end = min(col1, global_c0 + tile.shape[1])
                if r_start >= r_end or c_start >= c_end:
                    continue

                sub = tile[r_start - global_r0 : r_end - global_r0, c_start - global_c0 : c_end - global_c0]
                if not np.isfinite(sub).any() or float(np.nanmax(sub)) <= 0:
                    continue

                rows = np.arange(r_start, r_end)
                cols = np.arange(c_start, c_end)
                lats = y0 - (rows + 0.5) * dy
                lons = x0 + (cols + 0.5) * dx
                lat_bins = np.floor((lats - lat_min) / resolution_degrees).astype(int)
                lon_bins = np.floor((lons - lon_min) / resolution_degrees).astype(int)
                valid_lat = (lat_bins >= 0) & (lat_bins < n_lat)
                valid_lon = (lon_bins >= 0) & (lon_bins < n_lon)
                if not valid_lat.any() or not valid_lon.any():
                    continue
                sub2 = sub[np.ix_(valid_lat, valid_lon)]
                lat_bins2 = lat_bins[valid_lat]
                lon_bins2 = lon_bins[valid_lon]
                rr = np.repeat(lat_bins2[:, None], len(lon_bins2), axis=1)
                cc = np.repeat(lon_bins2[None, :], len(lat_bins2), axis=0)
                vals = np.nan_to_num(sub2, nan=0.0, posinf=0.0, neginf=0.0)
                np.add.at(sums, (rr.ravel(), cc.ravel()), vals.ravel())

    records = []
    from shapely.geometry import Point

    for lat_idx in range(n_lat):
        lat = lat_min + (lat_idx + 0.5) * resolution_degrees
        for lon_idx in range(n_lon):
            value = sums[lat_idx, lon_idx]
            if value <= 0:
                continue
            lon = lon_min + (lon_idx + 0.5) * resolution_degrees
            if not prepared.contains(Point(lon, lat)):
                continue
            records.append(
                {
                    "unit_id": f"ghsl_{resolution_degrees:g}_{lat_idx}_{lon_idx}",
                    "unit_name": f"GHSL grid {lon:.2f},{lat:.2f}",
                    "level": f"ghsl_grid_{resolution_degrees:g}deg",
                    "province": pd.NA,
                    "city": pd.NA,
                    "county": pd.NA,
                    "township": pd.NA,
                    "village": pd.NA,
                    "lon": lon,
                    "lat": lat,
                    "year": int(year),
                    "population": float(value),
                    "source_population": f"ghsl_population_r2023a_{year}_30ss",
                    "status_population": "observed_or_modelled_epoch",
                }
            )

    df = pd.DataFrame(records)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return df


def make_population_nowcast(
    panel: pd.DataFrame,
    source_year: int,
    target_year: int,
    output_csv: str | Path,
) -> pd.DataFrame:
    """Create a clearly labelled baseline nowcast by carrying latest available GHSL cells."""

    latest = panel[panel["year"].eq(source_year)].copy()
    if latest.empty:
        raise ValueError(f"No rows found for source_year={source_year}")
    latest["year"] = int(target_year)
    latest["source_population"] = f"carried_forward_from_ghsl_population_{source_year}"
    latest["status_population"] = "nowcast"
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    latest.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return latest
