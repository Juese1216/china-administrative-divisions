from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


STATIC_COLUMNS = [
    "unit_id",
    "unit_name",
    "level",
    "province",
    "city",
    "county",
    "township",
    "village",
    "lon",
    "lat",
]


def read_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        return tomllib.load(f)


def metric_names_from_config(config: dict[str, Any]) -> list[str]:
    return list(config.get("metrics", {}).keys())


def metric_weights_from_config(config: dict[str, Any], metrics: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for metric in metrics:
        weights[metric] = float(config.get("metrics", {}).get(metric, {}).get("weight", 1.0))
    total = sum(weights.values())
    if total <= 0:
        return {m: 1.0 / len(metrics) for m in metrics}
    return {m: w / total for m, w in weights.items()}


def read_panel(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input panel not found: {path}")
    df = pd.read_csv(path, dtype={"unit_id": "string"})
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    if "unit_id" in df.columns:
        df["unit_id"] = df["unit_id"].astype("string")
    return df


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_template(path: str | Path, metrics: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = STATIC_COLUMNS + ["year"]
    for metric in metrics:
        columns.extend([metric, f"source_{metric}", f"status_{metric}"])
    columns.extend(["source_note"])
    pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8-sig")
