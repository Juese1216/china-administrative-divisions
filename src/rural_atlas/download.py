from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


MOODOD_2023_FILES = {
    "provinces": "https://raw.githubusercontent.com/modood/Administrative-divisions-of-China/master/dist/provinces.csv",
    "cities": "https://raw.githubusercontent.com/modood/Administrative-divisions-of-China/master/dist/cities.csv",
    "areas": "https://raw.githubusercontent.com/modood/Administrative-divisions-of-China/master/dist/areas.csv",
    "streets": "https://raw.githubusercontent.com/modood/Administrative-divisions-of-China/master/dist/streets.csv",
    "villages": "https://raw.githubusercontent.com/modood/Administrative-divisions-of-China/master/dist/villages.csv",
}


def fetch_modood_2023(output_dir: str | Path, overwrite: bool = False) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for name, url in MOODOD_2023_FILES.items():
        target = output_dir / f"modood_2023_{name}.csv"
        if target.exists() and not overwrite:
            downloaded.append(target)
            continue
        urlretrieve(url, target)
        downloaded.append(target)
    return downloaded

