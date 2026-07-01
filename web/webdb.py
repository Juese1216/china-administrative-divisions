"""构建 Web 仪表盘使用的 SQLite 数据库。

本脚本读取 `data/processed/` 下已经生成的 CSV 结果，把 NER、RE、分类、
知识图谱和两版时序预测数据写入 SQLite，并建立常用查询索引。

运行示例：

    conda run --no-capture-output -n nlpEnv python web/webdb.py
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = Path("data/app/dashboard.sqlite")

TABLE_SOURCES = {
    "records": Path("data/processed/relation_extraction/records.csv"),
    "dynamic_triples": Path("data/processed/relation_extraction/dynamic_triples.csv"),
    "static_admin_nodes": Path(
        "data/processed/relation_extraction/static_admin_nodes.csv"
    ),
    "static_affiliation_triples": Path(
        "data/processed/relation_extraction/static_affiliation_triples.csv"
    ),
    "ner_entities": Path("data/processed/ner_rule_uie/entities.csv"),
    "class_frequency": Path("data/processed/classification/class_frequency.csv"),
    "rule_classification": Path(
        "data/processed/classification/rule_classification.csv"
    ),
    "ml_confusion_matrix": Path(
        "data/processed/classification/ml_confusion_matrix.csv"
    ),
    "ml_evaluation": Path("data/processed/classification/ml_evaluation.csv"),
    "semantic_ml_confusion_matrix": Path(
        "data/processed/classification/semantic_ml_confusion_matrix.csv"
    ),
    "semantic_ml_evaluation": Path(
        "data/processed/classification/semantic_ml_evaluation.csv"
    ),
    "annual_total": Path("data/processed/time_series_forecast/annual_total.csv"),
    "annual_type_counts_long": Path(
        "data/processed/time_series_forecast/annual_type_counts_long.csv"
    ),
    "forecast_total": Path("data/processed/time_series_forecast/forecast_total.csv"),
    "forecast_by_type": Path(
        "data/processed/time_series_forecast/forecast_by_type.csv"
    ),
    "forecast_metrics": Path("data/processed/time_series_forecast/model_metrics.csv"),
    "forecast_overview": Path(
        "data/processed/time_series_forecast/forecast_overview.csv"
    ),
    "rural_township_province": Path(
        "data/processed/rural_time_series/quantity_atlas/乡镇街道数量_2009_2035.csv"
    ),
    "rural_township_national": Path(
        "data/processed/rural_time_series/quantity_atlas/官方年鉴_乡级行政区划数量_2009_2035.csv"
    ),
    "rural_natural_village": Path(
        "data/processed/rural_time_series/quantity_atlas/自然村数量_2006_2035.csv"
    ),
}


def project_path(path: Path) -> Path:
    """把项目内相对路径转换成绝对路径。"""
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建 Flask 仪表盘使用的 SQLite 数据库。"
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    return parser.parse_args()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    resolved_path = project_path(path)
    if not resolved_path.exists():
        print(f"跳过缺失文件：{path}")
        return pd.DataFrame()
    return pd.read_csv(resolved_path)


def write_table(
    connection: sqlite3.Connection, table_name: str, csv_path: Path
) -> None:
    frame = read_csv_if_exists(csv_path)
    if frame.empty:
        pd.DataFrame().to_sql(table_name, connection, if_exists="replace", index=False)
        return
    frame.to_sql(table_name, connection, if_exists="replace", index=False)
    print(f"{table_name}: {len(frame)} rows")


def create_index(
    connection: sqlite3.Connection, table: str, columns: list[str], name: str
) -> None:
    existing = {
        row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if not set(columns).issubset(existing):
        return
    column_sql = ", ".join(columns)
    connection.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({column_sql})")


def create_indexes(connection: sqlite3.Connection) -> None:
    index_specs = [
        ("records", ["year"], "idx_records_year"),
        ("records", ["type_label"], "idx_records_type_label"),
        ("records", ["record_id"], "idx_records_record_id"),
        ("dynamic_triples", ["year"], "idx_dynamic_year"),
        ("dynamic_triples", ["predicate"], "idx_dynamic_predicate"),
        ("dynamic_triples", ["type_label"], "idx_dynamic_type"),
        ("dynamic_triples", ["subject"], "idx_dynamic_subject"),
        ("dynamic_triples", ["object"], "idx_dynamic_object"),
        ("ner_entities", ["year"], "idx_entities_year"),
        ("ner_entities", ["entity_type"], "idx_entities_type"),
        ("ner_entities", ["entity_text"], "idx_entities_text"),
        ("ner_entities", ["admin_code"], "idx_entities_admin_code"),
        ("static_admin_nodes", ["admin_code"], "idx_admin_code"),
        ("static_admin_nodes", ["admin_name"], "idx_admin_name"),
        ("static_admin_nodes", ["admin_level"], "idx_admin_level"),
        ("static_affiliation_triples", ["predicate"], "idx_static_predicate"),
        ("static_affiliation_triples", ["subject_name"], "idx_static_subject_name"),
        ("static_affiliation_triples", ["object_name"], "idx_static_object_name"),
        ("static_affiliation_triples", ["subject_code"], "idx_static_subject_code"),
        ("rule_classification", ["year"], "idx_rule_year"),
        ("rule_classification", ["type_label"], "idx_rule_type"),
        ("annual_total", ["year"], "idx_annual_total_year"),
        ("annual_type_counts_long", ["year"], "idx_annual_type_year"),
        ("annual_type_counts_long", ["type_label"], "idx_annual_type_label"),
        ("forecast_total", ["forecast_year"], "idx_forecast_total_year"),
        ("forecast_by_type", ["forecast_year"], "idx_forecast_type_year"),
        ("forecast_by_type", ["target"], "idx_forecast_type_target"),
        ("rural_township_province", ["province"], "idx_rural_province"),
        ("rural_township_province", ["year"], "idx_rural_province_year"),
        ("rural_township_national", ["year"], "idx_rural_national_year"),
        ("rural_natural_village", ["year"], "idx_rural_natural_year"),
    ]
    for table, columns, name in index_specs:
        create_index(connection, table, columns, name)


def create_metadata(connection: sqlite3.Connection, db_path: Path) -> None:
    rows = []
    for table_name, csv_path in TABLE_SOURCES.items():
        source_path = project_path(csv_path)
        source_stat = source_path.stat() if source_path.exists() else None
        row_count = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[
            0
        ]
        rows.append(
            {
                "table_name": table_name,
                "source_path": str(csv_path),
                "source_exists": int(source_path.exists()),
                "source_mtime": source_stat.st_mtime if source_stat else None,
                "source_size": source_stat.st_size if source_stat else None,
                "row_count": int(row_count),
            }
        )
    pd.DataFrame(rows).to_sql(
        "web_data_sources", connection, if_exists="replace", index=False
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)


def build_database(db_path: Path = DEFAULT_DB) -> None:
    db_path = project_path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as connection:
        for table_name, csv_path in TABLE_SOURCES.items():
            write_table(connection, table_name, csv_path)
        create_indexes(connection)
        create_metadata(connection, db_path)
        connection.commit()
    print(f"SQLite 数据库已生成：{db_path}")


def main() -> None:
    args = parse_args()
    build_database(args.db)


if __name__ == "__main__":
    main()
