from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_modood_2023_units(raw_dir: str | Path, output_csv: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    provinces = pd.read_csv(raw_dir / "modood_2023_provinces.csv", dtype=str)
    cities = pd.read_csv(raw_dir / "modood_2023_cities.csv", dtype=str)
    areas = pd.read_csv(raw_dir / "modood_2023_areas.csv", dtype=str)
    streets = pd.read_csv(raw_dir / "modood_2023_streets.csv", dtype=str)
    villages = pd.read_csv(raw_dir / "modood_2023_villages.csv", dtype=str)

    province_names = provinces.set_index("code")["name"]
    city_names = cities.set_index("code")["name"]
    area_names = areas.set_index("code")["name"]
    street_names = streets.set_index("code")["name"]

    rows: list[pd.DataFrame] = []

    province_units = provinces.rename(columns={"code": "unit_id", "name": "unit_name"}).copy()
    province_units["level"] = "province"
    province_units["parent_id"] = pd.NA
    province_units["province"] = province_units["unit_name"]
    rows.append(province_units)

    city_units = cities.rename(columns={"code": "unit_id", "name": "unit_name"}).copy()
    city_units["level"] = "city"
    city_units["parent_id"] = city_units["provinceCode"]
    city_units["province"] = city_units["provinceCode"].map(province_names)
    city_units["city"] = city_units["unit_name"]
    rows.append(city_units)

    area_units = areas.rename(columns={"code": "unit_id", "name": "unit_name"}).copy()
    area_units["level"] = "county"
    area_units["parent_id"] = area_units["cityCode"]
    area_units["province"] = area_units["provinceCode"].map(province_names)
    area_units["city"] = area_units["cityCode"].map(city_names)
    area_units["county"] = area_units["unit_name"]
    rows.append(area_units)

    street_units = streets.rename(columns={"code": "unit_id", "name": "unit_name"}).copy()
    street_units["level"] = "township"
    street_units["parent_id"] = street_units["areaCode"]
    street_units["province"] = street_units["provinceCode"].map(province_names)
    street_units["city"] = street_units["cityCode"].map(city_names)
    street_units["county"] = street_units["areaCode"].map(area_names)
    street_units["township"] = street_units["unit_name"]
    rows.append(street_units)

    village_units = villages.rename(columns={"code": "unit_id", "name": "unit_name"}).copy()
    village_units["level"] = "village_committee"
    village_units["parent_id"] = village_units["streetCode"]
    village_units["province"] = village_units["provinceCode"].map(province_names)
    village_units["city"] = village_units["cityCode"].map(city_names)
    village_units["county"] = village_units["areaCode"].map(area_names)
    village_units["township"] = village_units["streetCode"].map(street_names)
    village_units["village"] = village_units["unit_name"]
    rows.append(village_units)

    keep = [
        "unit_id",
        "unit_name",
        "level",
        "parent_id",
        "province",
        "city",
        "county",
        "township",
        "village",
        "provinceCode",
        "cityCode",
        "areaCode",
        "streetCode",
    ]
    units = pd.concat(rows, ignore_index=True, sort=False)
    for col in keep:
        if col not in units.columns:
            units[col] = pd.NA
    units = units[keep]
    units["source_admin"] = "modood_administrative_divisions_of_china_2023_from_nbs"
    units["source_admin_url"] = "https://github.com/modood/Administrative-divisions-of-China"

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    units.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return units

