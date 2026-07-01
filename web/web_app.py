"""Web 可视化系统。

本文件负责 Flask 页面和 JSON API。网页只读取当前项目生成的
`data/app/dashboard.sqlite`，该库由 `data/processed/` 下的新一轮
NER、RE、分类、知识图谱和时序预测 CSV 构建。

运行示例：

    conda run --no-capture-output -n nlpEnv python web/web_app.py
"""

from __future__ import annotations

import math
import os
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_from_directory

try:
    from webdb import DEFAULT_DB, TABLE_SOURCES, build_database
except ModuleNotFoundError:
    from web.webdb import DEFAULT_DB, TABLE_SOURCES, build_database


BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "web"
DB_PATH = BASE_DIR / DEFAULT_DB
RURAL_OUTPUT_DIR = BASE_DIR / "data/processed/rural_time_series"

LEVEL_NAMES = {
    "country": "国家",
    "province": "省级",
    "prefecture": "地市级",
    "city": "地市级",
    "county": "县区级",
    "township": "乡镇街道级",
    "town": "乡镇街道级",
}

LEVEL_ALIASES = {
    "province": "province",
    "prefecture": "prefecture",
    "city": "prefecture",
    "county": "county",
    "town": "township",
    "township": "township",
}

ENTITY_TYPE_NAMES = {
    "ADMIN_AREA": "区划名称",
    "GOV_ORG": "政府机关",
    "ADDRESS": "驻地地址",
}

GRAPH_RELATION_COLORS = {
    "下辖": "#8b98a9",
    "变更为": "#dc4c4c",
    "划归": "#f28e2b",
    "隶属于": "#4e79a7",
    "政府驻地": "#7b61d1",
    "驻地迁至": "#14a3a0",
    "直辖于": "#59a14f",
    "代管于": "#9c755f",
    "设立": "#2f6fed",
    "撤销": "#b07aa1",
    "合并为": "#edc948",
    "范围约束": "#6b7280",
    "发生调整": "#9467bd",
}

RELATION_TYPE_NAMES = {
    "REVOKE_ADMIN": "撤销建制",
    "ESTABLISH_ADMIN": "设立建制",
    "RENAME_ADMIN": "名称变更",
    "MERGE_ADMIN": "合并建制",
    "TRANSFER_ADMIN": "区域划转",
    "JURISDICTION_ADMIN": "管辖隶属",
    "DIRECT_ADMIN": "省级直辖",
    "ENTRUST_ADMIN": "委托代管",
    "GOV_RESIDENCE": "政府驻地",
    "RESIDENCE_TRANSFER": "驻地迁移",
    "AREA_INHERITANCE": "行政区域承继",
    "ADJUSTMENT_EVENT": "行政区划调整事件",
    "SCOPE_CONSTRAINT": "范围包含排除",
}

RELATION_LABEL_ORDER = list(RELATION_TYPE_NAMES)
RELATION_LABEL_TO_ID = {
    label: relation_id for relation_id, label in RELATION_TYPE_NAMES.items()
}

SOURCE_RELATION_TYPE_IDS = {
    "TEXT_UNIT_TRANSFER": ["TRANSFER_ADMIN"],
    "TEXT_UNIT_JURISDICTION": ["JURISDICTION_ADMIN"],
    "TEXT_RESIDENCE_TRANSFER": ["RESIDENCE_TRANSFER"],
    "TEXT_GOV_RESIDENCE": ["GOV_RESIDENCE"],
}

SEMANTIC_CLASS_GROUPS = {
    "建制变更": {"ESTABLISH_ADMIN", "REVOKE_ADMIN", "MERGE_ADMIN"},
    "名称变更": {"RENAME_ADMIN"},
    "管辖隶属调整": {"JURISDICTION_ADMIN", "DIRECT_ADMIN", "ENTRUST_ADMIN"},
    "区域划转": {"TRANSFER_ADMIN", "AREA_INHERITANCE", "SCOPE_CONSTRAINT"},
    "驻地变更": {"GOV_RESIDENCE", "RESIDENCE_TRANSFER"},
    "其他综合调整": {"ADJUSTMENT_EVENT"},
}

SEMANTIC_CLASS_ORDER = [
    "建制变更",
    "名称变更",
    "管辖隶属调整",
    "区域划转",
    "驻地变更",
    "其他综合调整",
]

SEMANTIC_CLASS_BY_TYPE_ID = {
    type_id: group_name
    for group_name, type_ids in SEMANTIC_CLASS_GROUPS.items()
    for type_id in type_ids
}

SEMANTIC_CLASS_BY_TYPE_LABEL = {
    RELATION_TYPE_NAMES[type_id]: group_name
    for type_id, group_name in SEMANTIC_CLASS_BY_TYPE_ID.items()
}

SEMANTIC_CLASS_TYPE_LABELS = {
    group_name: {
        RELATION_TYPE_NAMES[type_id]
        for type_id in type_ids
        if type_id in RELATION_TYPE_NAMES
    }
    for group_name, type_ids in SEMANTIC_CLASS_GROUPS.items()
}

CLASSIFICATION_VIEW_OPTIONS = [
    {"id": "detail", "label": "13类细粒度"},
    {"id": "semantic", "label": "语义聚合"},
]

APP_VERSION_OPTIONS = [
    {"id": "detail", "label": "13类细粒度版"},
    {"id": "semantic", "label": "语义聚合版"},
]

FORECAST_MODEL_OPTIONS = [
    {
        "id": "chosen",
        "label": "自动采用",
        "column": "chosen_forecast",
        "metric_model": None,
    },
    {
        "id": "exogenous_poisson",
        "label": "外部因子 Poisson",
        "column": "exogenous_regression_forecast",
        "metric_model": "exogenous_poisson",
    },
    {
        "id": "holt_winters",
        "label": "Holt-Winters",
        "column": "holt_winters_forecast",
        "metric_model": "holt_winters",
    },
    {
        "id": "linear_regression",
        "label": "线性回归",
        "column": "linear_regression_forecast",
        "metric_model": "linear_regression",
    },
]
FORECAST_MODEL_BY_ID = {item["id"]: item for item in FORECAST_MODEL_OPTIONS}


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(WEB_DIR / "templates"),
        static_folder=str(WEB_DIR / "static"),
        static_url_path="/static",
    )
    app.config["JSON_AS_ASCII"] = False

    if database_needs_rebuild(DB_PATH):
        build_database(DB_PATH)

    @app.context_processor
    def inject_app_version() -> dict[str, Any]:
        return {
            "app_version": request_version(),
            "app_version_options": APP_VERSION_OPTIONS,
        }

    @app.get("/")
    def dashboard_page() -> str:
        return render_template("dashboard.html", active="dashboard", title="首页")

    @app.get("/areas")
    def areas_page() -> str:
        return render_template("areas.html", active="areas", title="行政区划浏览")

    @app.get("/area/<code>")
    def area_detail_page(code: str) -> str:
        area = area_by_code(code)
        return render_template(
            "area_detail.html", active="areas", title="区划详情", area=area, code=code
        )

    @app.get("/graph")
    def graph_page() -> str:
        keyword = request.args.get("keyword", "")
        return render_template(
            "graph.html",
            active="graph",
            title="知识图谱",
            keyword=keyword,
        )

    @app.get("/ner")
    def ner_page() -> str:
        return render_template("ner.html", active="ner", title="NER 实体识别")

    @app.get("/relations")
    def relations_page() -> str:
        return render_template("relations.html", active="relations", title="关系抽取")

    @app.get("/classification")
    def classification_page() -> str:
        return render_template(
            "classification.html", active="classification", title="变更分类"
        )

    @app.get("/forecast")
    def forecast_page() -> str:
        return render_template("forecast.html", active="forecast", title="时序预测")

    @app.get("/forecast-v2")
    def forecast_v2_page() -> str:
        return render_template(
            "forecast_v2.html", active="forecast_v2", title="自然村预测"
        )

    @app.get("/artifacts/rural/<path:filename>")
    def rural_artifact(filename: str):
        return send_from_directory(RURAL_OUTPUT_DIR, filename)

    @app.get("/api/overview")
    def api_overview():
        version = request_version()
        with connect_db() as db:
            counts = {
                "区划编码": scalar(db, "SELECT COUNT(*) FROM static_admin_nodes"),
                "变更记录": scalar(db, "SELECT COUNT(*) FROM records"),
                "NER实体": scalar(db, "SELECT COUNT(*) FROM ner_entities"),
                "关系三元组": scalar(db, "SELECT COUNT(*) FROM dynamic_triples"),
                "关系类型": (
                    len(SEMANTIC_CLASS_ORDER)
                    if version == "semantic"
                    else scalar(db, "SELECT COUNT(*) FROM class_frequency")
                ),
                "分类记录": scalar(
                    db, "SELECT COUNT(DISTINCT record_id) FROM rule_classification"
                ),
                "静态隶属边": scalar(
                    db, "SELECT COUNT(*) FROM static_affiliation_triples"
                ),
                "预测年份": scalar(db, "SELECT COUNT(*) FROM forecast_total"),
            }
            level_distribution = {
                LEVEL_NAMES.get(item["admin_level"], item["admin_level"]): item["value"]
                for item in rows(
                    db,
                    """
                    SELECT admin_level, COUNT(*) AS value
                    FROM static_admin_nodes
                    GROUP BY admin_level
                    ORDER BY value DESC
                    """,
                )
            }
            type_counts = rows(
                db,
                "SELECT type_label, record_count AS value FROM class_frequency ORDER BY record_count DESC",
            )
            classification_distribution = (
                semantic_distribution_from_type_counts(type_counts)
                if version == "semantic"
                else detail_distribution_from_type_counts(type_counts)
            )
            annual_total = rows(
                db,
                """
                SELECT year, total_count AS total_changes
                FROM annual_total
                ORDER BY year
                """,
            )
        return jsonify(
            {
                "version": version,
                "version_options": APP_VERSION_OPTIONS,
                "counts": counts,
                "level_distribution": level_distribution,
                "classification_distribution": classification_distribution,
                "annual_total": annual_total,
            }
        )

    @app.get("/api/data-sources")
    def api_data_sources():
        with connect_db() as db:
            source_rows = rows(
                db,
                """
                SELECT table_name,
                       source_path,
                       source_exists,
                       source_mtime,
                       source_size,
                       row_count
                FROM web_data_sources
                ORDER BY table_name
                """,
            )
        return jsonify(source_rows)

    @app.get("/api/areas/provinces")
    def api_provinces():
        with connect_db() as db:
            data = rows(
                db,
                """
                SELECT admin_code, admin_name, admin_level, full_path
                FROM static_admin_nodes
                WHERE admin_level = 'province'
                ORDER BY admin_code
                """,
            )
        return jsonify([admin_row_to_area(item) for item in data])

    @app.get("/api/areas/children")
    def api_area_children():
        code = request.args.get("code", "")
        level_set = parse_level_set(request.args.get("levels", ""))
        return jsonify(children_of(code, level_set))

    @app.get("/api/area/<code>")
    def api_area_detail(code: str):
        area = area_by_code(code)
        if not area:
            return jsonify({"error": "area not found"}), 404
        changes = related_changes_for_area(code, limit=0)
        relations = related_relations_for_area(code, limit=0)
        entities = related_entities_for_area(code, limit=0)
        version = request_version()
        classifications = classifications_for_records(
            [item["event_id"] for item in changes],
            version=version,
        )
        children = children_of(code, None)
        return jsonify(
            {
                "version": version,
                "area": area,
                "children": children,
                "summary": {
                    "children_count": len(children),
                    "change_count": len(changes),
                    "relation_count": len(relations),
                    "entity_count": len(entities),
                },
                "changes": changes,
                "relations": relations,
                "entities": entities,
                "classifications": classifications,
                "graph": graph_for_keyword(area["name"], limit=100, version=version),
            }
        )

    @app.get("/api/entities")
    def api_entities():
        entity_type = request.args.get("type", "").strip()
        keyword = request.args.get("keyword", "").strip()
        limit = bounded_int(request.args.get("limit"), 0, 0, 100000)
        filters = []
        if entity_type:
            filters.append(("entity_type = ?", entity_type))
        if keyword:
            filters.append(
                (
                    "(entity_text LIKE ? OR sentence_id LIKE ? OR sentence LIKE ?)",
                    [like(keyword), like(keyword), like(keyword)],
                )
            )
        where, params = build_filters(filters)
        with connect_db() as db:
            record_rows = rows(
                db,
                f"""
                SELECT sentence_id AS event_id,
                       year,
                       entity_text,
                       entity_type,
                       start_pos,
                       end_pos,
                       admin_code AS normalized_code,
                       confidence,
                       method,
                       sentence
                FROM ner_entities
                {where}
                ORDER BY year DESC, sentence_id, start_pos
                """,
                params,
            )
            type_distribution = {
                ENTITY_TYPE_NAMES.get(item["entity_type"], item["entity_type"]): item[
                    "value"
                ]
                for item in rows(
                    db,
                    f"""
                    SELECT entity_type, COUNT(*) AS value
                    FROM ner_entities
                    {where}
                    GROUP BY entity_type
                    ORDER BY value DESC
                    """,
                    params,
                )
            }
            top_entities = {
                item["name"]: item["value"]
                for item in rows(
                    db,
                    f"""
                    SELECT COALESCE(NULLIF(normalized_name, ''), entity_text) AS name,
                           COUNT(*) AS value
                    FROM ner_entities
                    {where}
                    GROUP BY name
                    ORDER BY value DESC, name
                    LIMIT 20
                    """,
                    params,
                )
            }
        return jsonify(
            {
                "type_distribution": type_distribution,
                "top_entities": top_entities,
                "records": apply_limit(record_rows, limit),
            }
        )

    @app.get("/api/relations")
    def api_relations():
        view = request_version()
        relation = request.args.get("relation", "").strip()
        relation_type = normalize_type_filter(request.args.get("type", "").strip(), view)
        predicate = request.args.get("predicate", "").strip()
        if relation:
            normalized_relation = normalize_type_filter(relation, view)
            if normalized_relation:
                relation_type = normalized_relation
            else:
                predicate = relation
        keyword = request.args.get("keyword", "").strip()
        year = request.args.get("year", "").strip()
        limit = bounded_int(request.args.get("limit"), 0, 0, 100000)
        filters = []
        if predicate:
            filters.append(("predicate = ?", predicate))
        if year.isdigit():
            filters.append(("year = ?", int(year)))
        if keyword:
            filters.append(
                (
                    "(subject LIKE ? OR object LIKE ? OR evidence_text LIKE ?)",
                    [like(keyword), like(keyword), like(keyword)],
                )
            )
        where, params = build_filters(filters)
        with connect_db() as db:
            base_rows = dynamic_relation_rows(db, where, params)
            record_rows = filter_relation_rows_by_type(base_rows, relation_type, view)
            distribution = (
                {relation_type: len(record_rows)}
                if relation_type
                else relation_distribution_by_view(record_rows, view)
            )
        return jsonify(
            {
                "view": view,
                "view_options": CLASSIFICATION_VIEW_OPTIONS,
                "distribution": distribution,
                "records": apply_limit(record_rows, limit),
            }
        )

    @app.get("/api/classifications")
    def api_classifications():
        view = request_version()
        change_type = normalize_type_filter(request.args.get("type", "").strip(), view)
        year = request.args.get("year", "").strip()
        limit = bounded_int(request.args.get("limit"), 0, 0, 100000)
        filters = []
        if change_type:
            type_labels = type_labels_for_filter(change_type, view)
            placeholders = ",".join("?" for _ in type_labels)
            filters.append((f"type_label IN ({placeholders})", type_labels))
        if year.isdigit():
            filters.append(("year = ?", int(year)))
        where, params = build_filters(filters)
        with connect_db() as db:
            record_rows = rows(
                db,
                f"""
                SELECT record_id AS event_id,
                       year,
                       type_label AS original_change_type,
                       classification_confidence AS confidence,
                       '关系类型弱监督 + 多标签文本分类器' AS method,
                       classification_rule AS evidence,
                       record_text
                FROM rule_classification
                {where}
                ORDER BY year DESC, record_id DESC
                """,
                params,
            )
            for item in record_rows:
                if view == "semantic":
                    item["change_type"] = semantic_class_for_type_label(
                        item.get("original_change_type")
                    )
                else:
                    item["change_type"] = item.get("original_change_type")
            type_counts = rows(
                db,
                f"""
                SELECT type_label, COUNT(DISTINCT record_id) AS value
                FROM rule_classification
                {where}
                GROUP BY type_label
                ORDER BY value DESC
                """,
                params,
            )
            distribution = (
                semantic_distribution_from_type_counts(type_counts)
                if view == "semantic"
                else detail_distribution_from_type_counts(type_counts)
            )
            if view == "semantic" and table_exists(db, "semantic_ml_evaluation"):
                evaluation = rows(db, "SELECT * FROM semantic_ml_evaluation LIMIT 1")
                matrix = rows(db, "SELECT * FROM semantic_ml_confusion_matrix")
            else:
                evaluation = rows(db, "SELECT * FROM ml_evaluation LIMIT 1")
                matrix = rows(db, "SELECT * FROM ml_confusion_matrix")
                if view == "semantic":
                    matrix = semantic_confusion_matrix(matrix)
        return jsonify(
            {
                "view": view,
                "view_options": CLASSIFICATION_VIEW_OPTIONS,
                "distribution": distribution,
                "records": apply_limit(record_rows, limit),
                "evaluation": evaluation[0] if evaluation else {},
                "confusion_matrix": matrix,
            }
        )

    @app.get("/api/forecast")
    def api_forecast():
        target = request.args.get("target", "total_changes").strip() or "total_changes"
        model = request.args.get("model", "chosen").strip() or "chosen"
        if model not in FORECAST_MODEL_BY_ID:
            model = "chosen"
        resolved_target = "全部变更" if target == "total_changes" else target
        with connect_db() as db:
            targets = ["total_changes"] + [
                item["type_label"]
                for item in rows(
                    db,
                    "SELECT DISTINCT type_label FROM annual_type_counts_long ORDER BY type_label",
                )
            ]
            if resolved_target == "全部变更":
                history = rows(
                    db,
                    """
                    SELECT year, total_count AS actual
                    FROM annual_total
                    ORDER BY year
                    """,
                )
                forecast = rows(
                    db, "SELECT * FROM forecast_total ORDER BY forecast_year"
                )
            else:
                history = rows(
                    db,
                    """
                    SELECT year, type_count AS actual
                    FROM annual_type_counts_long
                    WHERE type_label = ?
                    ORDER BY year
                    """,
                    [resolved_target],
                )
                forecast = rows(
                    db,
                    "SELECT * FROM forecast_by_type WHERE target = ? ORDER BY forecast_year",
                    [resolved_target],
                )
            metrics = rows(
                db, "SELECT * FROM forecast_metrics WHERE target = ?", [resolved_target]
            )
            overview = rows(db, "SELECT * FROM forecast_overview LIMIT 1")
            annual = rows(
                db,
                "SELECT year, total_count AS total_changes FROM annual_total ORDER BY year",
            )
        series = forecast_series(target, history, forecast, metrics, model)
        return jsonify(
            {
                "targets": targets,
                "target": target,
                "model": model,
                "model_options": FORECAST_MODEL_OPTIONS,
                "series": series,
                "annual": annual,
                "metrics": metrics,
                "overview": overview[0] if overview else {},
            }
        )

    @app.get("/api/forecast/year/<int:year>")
    def api_forecast_year(year: int):
        with connect_db() as db:
            events = rows(
                db,
                """
                SELECT record_id AS event_id,
                       year,
                       type_label AS title,
                       record_text AS content
                FROM records
                WHERE year = ?
                ORDER BY record_id
                """,
                [year],
            )
            classes = rows(
                db,
                """
                SELECT record_id AS event_id,
                       year,
                       type_label AS change_type,
                       classification_confidence AS confidence,
                       classification_rule AS evidence
                FROM rule_classification
                WHERE year = ?
                ORDER BY record_id
                """,
                [year],
            )
        return jsonify({"year": year, "events": events, "classifications": classes})

    @app.get("/api/forecast/rural")
    def api_forecast_rural():
        province = request.args.get("province", "北京市").strip() or "北京市"
        with connect_db() as db:
            provinces = [
                item["province"]
                for item in rows(
                    db,
                    """
                    SELECT DISTINCT province
                    FROM rural_township_province
                    WHERE province IS NOT NULL AND province != ''
                    ORDER BY province
                    """,
                )
            ]
            national = rows(db, "SELECT * FROM rural_township_national ORDER BY year")
            natural = rows(db, "SELECT * FROM rural_natural_village ORDER BY year")
            province_series = rows(
                db,
                """
                SELECT *
                FROM rural_township_province
                WHERE province = ?
                ORDER BY year
                """,
                [province],
            )
        artifact_candidates = {
            "quantity_atlas": "/artifacts/rural/quantity_atlas/数量变化图谱.html",
            "township_map": "/artifacts/rural/quantity_atlas/乡镇街道数量地图.html",
            "trend_dashboard": "/artifacts/rural/quantity_atlas/数量趋势仪表盘.html",
            "population_atlas": "/artifacts/rural/ghsl_population_atlas/atlas.html",
        }
        artifacts = {
            name: url
            for name, url in artifact_candidates.items()
            if (RURAL_OUTPUT_DIR / url.removeprefix("/artifacts/rural/")).exists()
        }
        return jsonify(
            {
                "province": province,
                "provinces": provinces,
                "national": national,
                "natural": natural,
                "province_series": province_series,
                "artifacts": artifacts,
            }
        )

    @app.get("/api/graph/search")
    def api_graph_search():
        version = request_version()
        keyword = request.args.get("keyword", "通辽市").strip() or "通辽市"
        return jsonify(graph_for_keyword(keyword, limit=140, version=version))

    @app.get("/api/graph/overview")
    def api_graph_overview():
        graph = graph_overview()
        graph["version"] = request_version()
        return jsonify(graph)

    @app.get("/api/graph/node")
    def api_graph_node():
        name = request.args.get("name", "").strip()
        code = resolve_area_code_by_name(name)
        area = area_by_code(code) if code else None
        return jsonify(
            {
                "name": name,
                "area": area,
                "relations": related_relations_for_area(name, limit=50),
            }
        )

    return app


def database_needs_rebuild(db_path: Path) -> bool:
    """判断 Web SQLite 是否需要按新版 CSV 重新生成。"""
    if not db_path.exists():
        return True

    db_mtime = db_path.stat().st_mtime
    for source_path in TABLE_SOURCES.values():
        resolved_path = (
            BASE_DIR / source_path if not source_path.is_absolute() else source_path
        )
        if resolved_path.exists() and resolved_path.stat().st_mtime > db_mtime:
            return True

    connection = None
    try:
        connection = sqlite3.connect(db_path)
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        expected_tables = set(TABLE_SOURCES) | {"web_data_sources"}
        if not expected_tables.issubset(table_names):
            return True
        source_rows = connection.execute(
            "SELECT table_name, source_path FROM web_data_sources"
        ).fetchall()
    except sqlite3.Error:
        return True
    finally:
        if connection is not None:
            connection.close()

    expected_sources = {name: str(path) for name, path in TABLE_SOURCES.items()}
    actual_sources = {name: source for name, source in source_rows}
    if actual_sources != expected_sources:
        return True
    return False


def connect_db() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def scalar(db: sqlite3.Connection, sql: str, params: list[Any] | None = None) -> Any:
    row = db.execute(sql, params or []).fetchone()
    return row[0] if row else None


def rows(
    db: sqlite3.Connection, sql: str, params: list[Any] | None = None
) -> list[dict[str, Any]]:
    return [clean_row(dict(row)) for row in db.execute(sql, params or []).fetchall()]


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in row.items():
        if isinstance(value, float) and math.isnan(value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


def bounded_int(value: str | None, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value) if value is not None else default
    except ValueError:
        number = default
    return max(minimum, min(maximum, number))


def like(value: str) -> str:
    return f"%{value}%"


def build_filters(filters: list[tuple[str, Any]]) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for clause, value in filters:
        if value in ("", None, []):
            continue
        clauses.append(clause)
        if isinstance(value, list):
            params.extend(value)
        else:
            params.append(value)
    return ("WHERE " + " AND ".join(clauses), params) if clauses else ("", [])


def apply_limit(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return items[:limit] if limit > 0 else items


def sorted_relation_type_ids(type_ids: list[str]) -> list[str]:
    order = {relation_id: index for index, relation_id in enumerate(RELATION_LABEL_ORDER)}
    known = [item for item in type_ids if item in RELATION_TYPE_NAMES]
    return sorted(set(known), key=lambda item: order[item])


def relation_type_ids_from_source(
    source_relation_type: Any, fallback_type_label: Any = ""
) -> list[str]:
    source = str(source_relation_type or "").strip()
    if source in SOURCE_RELATION_TYPE_IDS:
        return SOURCE_RELATION_TYPE_IDS[source]
    source_ids = [item.strip() for item in source.split("+") if item.strip()]
    if source_ids:
        known_source_ids = sorted_relation_type_ids(source_ids)
        if known_source_ids:
            return known_source_ids
    labels = [item.strip() for item in str(fallback_type_label or "").split("/") if item.strip()]
    fallback_ids = sorted_relation_type_ids(
        [RELATION_LABEL_TO_ID[item] for item in labels if item in RELATION_LABEL_TO_ID]
    )
    return fallback_ids or ["ADJUSTMENT_EVENT"]


def semantic_classes_from_type_ids(type_ids: list[str]) -> list[str]:
    classes = {
        SEMANTIC_CLASS_BY_TYPE_ID.get(type_id, "其他综合调整")
        for type_id in type_ids
    }
    return [name for name in SEMANTIC_CLASS_ORDER if name in classes]


def semantic_classes_from_type_label_text(type_label: Any) -> list[str]:
    labels = [item.strip() for item in str(type_label or "").split("/") if item.strip()]
    classes = {
        SEMANTIC_CLASS_BY_TYPE_LABEL.get(label, "其他综合调整")
        for label in labels
    }
    return [name for name in SEMANTIC_CLASS_ORDER if name in classes]


def semantic_class_for_type_label(type_label: Any) -> str:
    classes = semantic_classes_from_type_label_text(type_label)
    return classes[0] if classes else "其他综合调整"


def semantic_classes_for_dynamic_relation(
    item: dict[str, Any], type_ids: list[str] | None = None
) -> list[str]:
    resolved_type_ids = type_ids or relation_type_ids_from_source(
        item.get("source_relation_type"), item.get("type_label")
    )
    return semantic_classes_from_type_ids(resolved_type_ids)


def normalize_semantic_type_filter(value: Any) -> str:
    candidate = str(value or "").strip()
    if not candidate:
        return ""
    if candidate in SEMANTIC_CLASS_GROUPS:
        return candidate
    if candidate in RELATION_LABEL_TO_ID or "/" in candidate:
        return semantic_class_for_type_label(candidate)
    return candidate


def semantic_type_labels_for_filter(value: Any) -> list[str]:
    semantic_type = normalize_semantic_type_filter(value)
    if semantic_type in SEMANTIC_CLASS_TYPE_LABELS:
        return sorted(SEMANTIC_CLASS_TYPE_LABELS[semantic_type])
    candidate = str(value or "").strip()
    return [candidate] if candidate in RELATION_LABEL_TO_ID else []


def semantic_distribution_from_type_counts(
    type_counts: list[dict[str, Any]],
    value_key: str = "value",
    include_all: bool = True,
) -> dict[str, int]:
    counts = {name: 0 for name in SEMANTIC_CLASS_ORDER} if include_all else {}
    for item in type_counts:
        count = int(item.get(value_key) or 0)
        for class_name in semantic_classes_from_type_label_text(item.get("type_label")):
            counts[class_name] = counts.get(class_name, 0) + count
    return {
        name: counts.get(name, 0)
        for name in sorted(
            counts,
            key=lambda item: (-counts[item], SEMANTIC_CLASS_ORDER.index(item)),
        )
    }


def classification_view(value: Any) -> str:
    view = str(value or "detail").strip().lower()
    return "semantic" if view in {"semantic", "sem", "aggregate", "aggregation"} else "detail"


def request_version(default: str = "detail") -> str:
    return classification_view(
        request.args.get("version") or request.args.get("view") or default
    )


def table_exists(db: sqlite3.Connection, table_name: str) -> bool:
    return bool(
        db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            [table_name],
        ).fetchone()
    )


def normalize_detail_type_filter(value: Any) -> str:
    candidate = str(value or "").strip()
    if not candidate:
        return ""
    if candidate in RELATION_LABEL_TO_ID:
        return candidate
    if candidate in RELATION_TYPE_NAMES:
        return RELATION_TYPE_NAMES[candidate]
    return candidate if candidate in RELATION_LABEL_TO_ID else ""


def normalize_type_filter(value: Any, view: str) -> str:
    if classification_view(view) == "detail":
        return normalize_detail_type_filter(value)
    return normalize_semantic_type_filter(value)


def type_labels_for_filter(value: Any, view: str) -> list[str]:
    if classification_view(view) == "detail":
        detail_type = normalize_detail_type_filter(value)
        return [detail_type] if detail_type else []
    return semantic_type_labels_for_filter(value)


def detail_distribution_from_type_counts(
    type_counts: list[dict[str, Any]], value_key: str = "value"
) -> dict[str, int]:
    counts = {label: 0 for label in RELATION_TYPE_NAMES.values()}
    for item in type_counts:
        label = str(item.get("type_label") or "").strip()
        if not label:
            continue
        counts[label] = counts.get(label, 0) + int(item.get(value_key) or 0)
    order = {label: index for index, label in enumerate(RELATION_TYPE_NAMES.values())}
    return {
        label: count
        for label, count in sorted(
            counts.items(),
            key=lambda pair: (-pair[1], order.get(pair[0], 999)),
        )
    }


def semantic_confusion_matrix(matrix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in matrix_rows:
        class_name = semantic_class_for_type_label(row.get("actual_type"))
        target = grouped.setdefault(
            class_name,
            {
                "actual_type": class_name,
                "type_id": class_name,
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 0,
                "true_negative": 0,
            },
        )
        for key in ("true_positive", "false_positive", "false_negative", "true_negative"):
            target[key] += int(row.get(key) or 0)
    result = []
    for class_name in SEMANTIC_CLASS_ORDER:
        if class_name not in grouped:
            continue
        item = grouped[class_name]
        tp = item["true_positive"]
        fp = item["false_positive"]
        fn = item["false_negative"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        item["precision"] = round(precision, 4)
        item["recall"] = round(recall, 4)
        item["f1"] = round(f1, 4)
        result.append(item)
    return result


ADMIN_ENDPOINT_SUFFIXES = (
    "特别行政区",
    "自治区",
    "自治州",
    "自治县",
    "自治旗",
    "地区",
    "新区",
    "矿区",
    "林区",
    "特区",
    "街道",
    "省",
    "市",
    "县",
    "区",
    "旗",
    "盟",
    "镇",
    "乡",
    "村",
)

ADMIN_CONTEXT_SUFFIXES = ("特别行政区", "自治区", "自治州", "地区", "省", "市", "盟")

ADMIN_ENDPOINT_PREFIXES = (
    "新设立的",
    "新设立",
    "新设",
    "新的",
    "县级",
    "地级",
    "省级",
    "原",
    "以原",
    "撤销",
    "设立",
    "将",
    "把",
    "由",
)

RELATION_ENDPOINT_MARKERS = (
    "行政区域划为",
    "的行政区域为",
    "行政区域为",
    "更名为",
    "设立县级",
    "设立地级",
    "设立新的",
    "设立",
)

INVALID_ENDPOINT_MARKERS = ("行政区划调整", "区划调整")


def strip_dynamic_endpoint_phrases(value: Any) -> str:
    text = str(value or "").strip(" ，。；;、：:/")
    if not text or text.startswith("无旧区划") or text.startswith("无新区划"):
        return text
    if any(marker in text for marker in INVALID_ENDPOINT_MARKERS):
        return ""
    if "人民政府驻" in text:
        text = text.rsplit("人民政府驻", 1)[-1]
    for prefix in ADMIN_ENDPOINT_PREFIXES:
        if text.startswith(prefix) and len(text) > len(prefix) + 1:
            text = text[len(prefix) :]
    if "辖原" in text:
        text = text.split("辖原", 1)[1]
    for marker in RELATION_ENDPOINT_MARKERS:
        if marker in text:
            text = text.split(marker, 1)[1]
    for prefix in ADMIN_ENDPOINT_PREFIXES:
        if text.startswith(prefix) and len(text) > len(prefix) + 1:
            text = text[len(prefix) :]
    for suffix in ("的行政区域", "行政区域", "管辖区域", "管辖"):
        if text.endswith(suffix) and len(text) > len(suffix):
            text = text[: -len(suffix)]
    return text.strip(" ，。；;、：:/")


def is_admin_endpoint_text(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if text.startswith("无旧区划") or text.startswith("无新区划"):
        return True
    if any(marker in text for marker in INVALID_ENDPOINT_MARKERS):
        return False
    return text.endswith(ADMIN_ENDPOINT_SUFFIXES)


def clean_historical_endpoint_candidate(candidate: str, fallback: str) -> str:
    text = str(candidate or "").strip(" ，。；;、：:/")
    for separator in ("和", "及", "与", "、"):
        if separator in text and text.endswith(fallback):
            text = text.rsplit(separator, 1)[-1]
    stripped = strip_dynamic_endpoint_phrases(text)
    if stripped or any(marker in text for marker in INVALID_ENDPOINT_MARKERS):
        text = stripped
    for prefix in ADMIN_ENDPOINT_PREFIXES:
        if text.startswith(prefix) and len(text) > len(prefix) + 1:
            text = text[len(prefix) :]
    search_end = max(0, len(text) - len(fallback))
    cut_at = 0
    for suffix in ADMIN_CONTEXT_SUFFIXES:
        position = text.rfind(suffix, 0, search_end)
        if position >= 0:
            cut_at = max(cut_at, position + len(suffix))
    if cut_at:
        text = text[cut_at:]
    stripped = strip_dynamic_endpoint_phrases(text)
    if stripped or any(marker in text for marker in INVALID_ENDPOINT_MARKERS):
        text = stripped
    for prefix in ADMIN_ENDPOINT_PREFIXES:
        if text.startswith(prefix) and len(text) > len(prefix) + 1:
            text = text[len(prefix) :]
    return text.strip(" ，。；;、：:/")


def restore_historical_endpoint_name(name: Any, evidence: Any) -> str:
    original = str(name or "").strip()
    if not original:
        return ""
    normalized = original[1:] if original.startswith("级") and len(original) > 2 else original
    evidence_text = str(evidence or "")
    if not evidence_text:
        return normalized
    pattern = re.compile(rf"[\u4e00-\u9fa5]{{0,14}}{re.escape(normalized)}")
    candidates: list[str] = []
    for match in pattern.finditer(evidence_text):
        candidate = clean_historical_endpoint_candidate(match.group(0), normalized)
        if (
            candidate
            and candidate != normalized
            and candidate.endswith(normalized)
            and candidate.endswith(ADMIN_ENDPOINT_SUFFIXES)
            and 2 < len(candidate) <= 14
        ):
            candidates.append(candidate)
    if not candidates:
        return normalized
    return sorted(set(candidates), key=lambda value: (len(value), value))[0]


def normalize_dynamic_endpoint_text(name: Any) -> str:
    text = strip_dynamic_endpoint_phrases(name)
    return text if is_admin_endpoint_text(text) else ""


def enrich_dynamic_relation_row(item: dict[str, Any]) -> dict[str, Any]:
    evidence = item.get("evidence")
    item["subject"] = restore_historical_endpoint_name(
        normalize_dynamic_endpoint_text(item.get("subject")), evidence
    )
    item["object"] = restore_historical_endpoint_name(
        normalize_dynamic_endpoint_text(item.get("object")), evidence
    )
    type_ids = relation_type_ids_from_source(
        item.get("source_relation_type"), item.get("type_label")
    )
    type_labels = [RELATION_TYPE_NAMES[type_id] for type_id in type_ids]
    semantic_labels = semantic_classes_for_dynamic_relation(item, type_ids)
    item["relation_type_ids"] = " / ".join(type_ids)
    item["relation_type_labels"] = " / ".join(type_labels)
    item["relation_type"] = item["relation_type_labels"]
    item["primary_relation_type"] = type_labels[0] if type_labels else ""
    item["semantic_relation_type_labels"] = " / ".join(semantic_labels)
    item["semantic_relation_type"] = item["semantic_relation_type_labels"]
    item["semantic_primary_relation_type"] = semantic_labels[0] if semantic_labels else ""
    return item


def filter_relation_rows_by_type(
    relation_rows: list[dict[str, Any]], relation_type: str, view: str = "detail"
) -> list[dict[str, Any]]:
    normalized_type = normalize_type_filter(relation_type, view)
    if not normalized_type:
        return relation_rows
    field_name = (
        "relation_type_labels"
        if classification_view(view) == "detail"
        else "semantic_relation_type_labels"
    )
    return [
        item
        for item in relation_rows
        if normalized_type
        in [
            label.strip()
            for label in str(item.get(field_name) or "").split("/")
            if label.strip()
        ]
    ]


def semantic_relation_type_distribution(relation_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in SEMANTIC_CLASS_ORDER}
    for item in relation_rows:
        labels = {
            label.strip()
            for label in str(item.get("semantic_relation_type_labels") or "").split("/")
            if label.strip()
        } or {"其他综合调整"}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
    return {
        label: count
        for label, count in sorted(
            counts.items(),
            key=lambda pair: (-pair[1], SEMANTIC_CLASS_ORDER.index(pair[0])),
        )
    }


def detail_relation_type_distribution(relation_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {label: 0 for label in RELATION_TYPE_NAMES.values()}
    for item in relation_rows:
        labels = {
            label.strip()
            for label in str(item.get("relation_type_labels") or "").split("/")
            if label.strip()
        }
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
    order = {label: index for index, label in enumerate(RELATION_TYPE_NAMES.values())}
    return {
        label: count
        for label, count in sorted(
            counts.items(),
            key=lambda pair: (-pair[1], order.get(pair[0], 999)),
        )
    }


def relation_distribution_by_view(
    relation_rows: list[dict[str, Any]], view: str
) -> dict[str, int]:
    if classification_view(view) == "detail":
        return detail_relation_type_distribution(relation_rows)
    return semantic_relation_type_distribution(relation_rows)


def graph_relation_color(relation: str, edge_type: str) -> str:
    return GRAPH_RELATION_COLORS.get(relation, "#60758f")


def normalize_area_text(value: Any) -> str:
    text = str(value or "").strip()
    for token in ["/", " ", "\t", "\n", "\r", "中国"]:
        text = text.replace(token, "")
    return text


def display_level(level: str | None) -> str:
    return LEVEL_NAMES.get(str(level or ""), str(level or ""))


def page_level(level: str | None) -> str:
    if level == "prefecture":
        return "city"
    if level == "township":
        return "town"
    return str(level or "")


def admin_row_to_area(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "code": str(row.get("admin_code") or ""),
        "name": row.get("admin_name"),
        "level": page_level(row.get("admin_level")),
        "level_name": display_level(row.get("admin_level")),
        "parent_code": row.get("parent_code"),
        "full_path": row.get("full_path"),
        "source_table": row.get("source_table"),
    }


def parse_level_set(levels: str) -> set[str] | None:
    if not levels:
        return None
    parsed = {
        LEVEL_ALIASES.get(item.strip(), item.strip())
        for item in levels.split(",")
        if item.strip()
    }
    return parsed or None


@lru_cache(maxsize=1)
def admin_lookup_rows() -> tuple[dict[str, Any], ...]:
    with connect_db() as db:
        return tuple(
            rows(
                db,
                """
                SELECT admin_code, admin_name, admin_level, full_path, source_table
                FROM static_admin_nodes
                WHERE admin_level != 'country'
                ORDER BY admin_level, admin_code
                """,
            )
        )


@lru_cache(maxsize=1)
def static_area_index() -> dict[str, Any]:
    with connect_db() as db:
        node_rows = rows(
            db,
            """
            SELECT admin_code, admin_name, admin_level, full_path, source_table
            FROM static_admin_nodes
            ORDER BY admin_level, admin_code
            """,
        )
        edge_rows = rows(
            db,
            """
            SELECT child.admin_code AS child_code,
                   parent.admin_code AS parent_code
            FROM static_affiliation_triples AS edge
            JOIN static_admin_nodes AS child ON child.node_id = edge.subject_node_id
            JOIN static_admin_nodes AS parent ON parent.node_id = edge.object_node_id
            """,
        )
    by_code = {str(item["admin_code"]): item for item in node_rows}
    parent_by_code = {
        str(item["child_code"]): str(item["parent_code"]) for item in edge_rows
    }
    children_by_parent: dict[str, list[str]] = {}
    for item in edge_rows:
        children_by_parent.setdefault(str(item["parent_code"]), []).append(
            str(item["child_code"])
        )
    return {
        "by_code": by_code,
        "parent_by_code": parent_by_code,
        "children_by_parent": children_by_parent,
    }


def static_area_row(code: str | None) -> dict[str, Any] | None:
    if not code:
        return None
    return static_area_index()["by_code"].get(str(code))


def static_parent_code(code: str | None) -> str | None:
    if not code:
        return None
    parent = static_area_index()["parent_by_code"].get(str(code))
    return parent if parent and parent != "CN" else parent


def static_path_rows(code: str | None, root_code: str | None = None) -> list[dict[str, Any]]:
    if not code:
        return []
    index = static_area_index()
    by_code = index["by_code"]
    parent_by_code = index["parent_by_code"]
    chain: list[dict[str, Any]] = []
    current = str(code)
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        row = by_code.get(current)
        if row:
            chain.append(row)
        if current == "CN":
            break
        current = parent_by_code.get(current, "")
    chain.reverse()
    if root_code and str(code) != str(root_code):
        root_text = str(root_code)
        for index_value, row in enumerate(chain):
            if str(row.get("admin_code")) == root_text:
                return chain[index_value:]
    return [row for row in chain if str(row.get("admin_code")) != "CN"]


@lru_cache(maxsize=20000)
def resolve_area_code_by_name(name: str | None) -> str | None:
    query = str(name or "").strip()
    if not query:
        return None
    query_text = normalize_area_text(query)
    exact = [item for item in admin_lookup_rows() if item.get("admin_name") == query]
    if exact:
        exact.sort(
            key=lambda item: (len(str(item["admin_code"])), str(item["admin_code"]))
        )
        return str(exact[0]["admin_code"])

    candidates: list[tuple[int, str]] = []
    for item in admin_lookup_rows():
        row_path = normalize_area_text(item.get("full_path"))
        score = 0
        if row_path.endswith(query_text):
            score = 400 + len(row_path)
        elif query_text and query_text in row_path:
            score = 300 + len(row_path)
        if score:
            candidates.append((score, str(item["admin_code"])))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def area_by_code(code: str | None) -> dict[str, Any] | None:
    if not code:
        return None
    with connect_db() as db:
        result = rows(
            db,
            """
            SELECT admin_code, admin_name, admin_level, full_path, source_table
            FROM static_admin_nodes
            WHERE admin_code = ?
            LIMIT 1
            """,
            [str(code)],
        )
    return admin_row_to_area(result[0]) if result else None


def children_of(code: str, levels: set[str] | None = None) -> list[dict[str, Any]]:
    area = area_by_code(code)
    if not area:
        return []
    filters = [
        "parent.node_id = child_edge.object_node_id",
        "child.node_id = child_edge.subject_node_id",
        "parent.admin_code = ?",
    ]
    params: list[Any] = [str(code)]
    if levels:
        filters.append("child.admin_level IN (" + ",".join("?" for _ in levels) + ")")
        params.extend(sorted(levels))
    with connect_db() as db:
        data = rows(
            db,
            f"""
            SELECT child.admin_code,
                   child.admin_name,
                   child.admin_level,
                   child.full_path,
                   child.source_table,
                   parent.admin_code AS parent_code
            FROM static_affiliation_triples AS child_edge
            JOIN static_admin_nodes AS parent
            JOIN static_admin_nodes AS child
            WHERE {' AND '.join(filters)}
            ORDER BY child.admin_level, child.admin_code
            """,
            params,
        )
    return [admin_row_to_area(item) for item in data]


def area_keyword(code_or_keyword: str) -> str:
    area = area_by_code(code_or_keyword)
    if area:
        return str(area["name"])
    resolved = resolve_area_code_by_name(code_or_keyword)
    area = area_by_code(resolved) if resolved else None
    return str(area["name"]) if area else str(code_or_keyword)


def related_changes_for_area(code: str, limit: int = 200) -> list[dict[str, Any]]:
    area = area_by_code(code)
    if not area:
        return []
    keyword = str(area["name"])
    with connect_db() as db:
        data = rows(
            db,
            """
            SELECT record_id AS event_id,
                   year,
                   type_label AS title,
                   record_text AS content
            FROM records
            WHERE record_text LIKE ?
               OR before_entities LIKE ?
               OR after_entities LIKE ?
            ORDER BY year DESC, record_id DESC
            """,
            [like(keyword), like(keyword), like(keyword)],
        )
    return apply_limit(data, limit)


def related_relations_for_area(
    code_or_keyword: str, limit: int = 200
) -> list[dict[str, Any]]:
    keyword = area_keyword(code_or_keyword)
    with connect_db() as db:
        data = dynamic_relation_rows(
            db,
            """
            WHERE subject LIKE ?
               OR object LIKE ?
               OR evidence_text LIKE ?
            """,
            [like(keyword), like(keyword), like(keyword)],
        )
    return apply_limit(data, limit)


def dynamic_relation_rows(
    db: sqlite3.Connection, where: str, params: list[Any]
) -> list[dict[str, Any]]:
    data = rows(
        db,
        f"""
        SELECT dynamic_triple_id,
               record_id AS event_id,
               year,
               subject,
               predicate AS relation,
               object,
               type_label,
               source_relation_type,
               evidence_text AS evidence,
               1.0 AS confidence
        FROM dynamic_triples
        {where}
        ORDER BY year DESC, dynamic_triple_id DESC
        """,
        params,
    )
    enriched = [enrich_dynamic_relation_row(item) for item in data]
    return [
        item
        for item in enriched
        if str(item.get("subject") or "").strip()
        and str(item.get("object") or "").strip()
    ]


def related_entities_for_area(code: str, limit: int = 200) -> list[dict[str, Any]]:
    area = area_by_code(code)
    if not area:
        return []
    keyword = str(area["name"])
    with connect_db() as db:
        data = rows(
            db,
            """
            SELECT sentence_id AS event_id,
                   year,
                   entity_text,
                   entity_type,
                   admin_code AS normalized_code,
                   confidence,
                   method,
                   start_pos,
                   end_pos,
                   sentence
            FROM ner_entities
            WHERE admin_code = ?
               OR entity_text LIKE ?
               OR sentence LIKE ?
            ORDER BY year DESC, sentence_id, start_pos
            """,
            [str(code), like(keyword), like(keyword)],
        )
    return apply_limit(data, limit)


def classifications_for_records(
    record_ids: list[str], version: str = "detail"
) -> list[dict[str, Any]]:
    if not record_ids:
        return []
    placeholders = ",".join("?" for _ in record_ids)
    with connect_db() as db:
        record_rows = rows(
            db,
            f"""
            SELECT record_id AS event_id,
                   year,
                   type_label AS original_change_type,
                   classification_confidence AS confidence,
                   '规则分类 + 弱监督文本分类器' AS method,
                   classification_rule AS evidence
            FROM rule_classification
            WHERE record_id IN ({placeholders})
            ORDER BY year DESC, record_id DESC
            """,
            record_ids,
        )
    view = classification_view(version)
    for item in record_rows:
        item["change_type"] = (
            semantic_class_for_type_label(item.get("original_change_type"))
            if view == "semantic"
            else item.get("original_change_type")
        )
    return record_rows


def graph_for_keyword(
    keyword: str, limit: int = 100, version: str = "detail"
) -> dict[str, Any]:
    view = classification_view(version)
    nodes: dict[str, dict[str, Any]] = {}
    links: list[dict[str, Any]] = []
    link_keys: set[tuple[Any, ...]] = set()
    dynamic_edges: list[dict[str, Any]] = []

    code = resolve_area_code_by_name(keyword)
    area = area_by_code(code) if code else None
    root_code = str(area["code"]) if area else None

    def static_symbol_size(level: str | None, is_root: bool = False) -> int:
        if is_root:
            return 58
        return {
            "province": 46,
            "prefecture": 38,
            "county": 32,
            "township": 26,
        }.get(str(level or ""), 30)

    def add_static_node(code_value: str, size: int | None = None) -> str | None:
        row = static_area_row(code_value)
        if not row:
            return None
        node_id = str(row["admin_code"])
        is_root = root_code is not None and node_id == root_code
        category = "查询区划" if is_root else "静态区划"
        node_size = size or static_symbol_size(row.get("admin_level"), is_root)
        if node_id in nodes:
            nodes[node_id]["symbolSize"] = max(
                nodes[node_id].get("symbolSize", node_size), node_size
            )
            if is_root:
                nodes[node_id]["category"] = "查询区划"
            return node_id
        nodes[node_id] = {
            "id": node_id,
            "name": row["admin_name"],
            "symbolSize": node_size,
            "category": category,
            "area_code": node_id,
            "admin_level": row.get("admin_level"),
            "value": row.get("full_path") or row["admin_name"],
        }
        return node_id

    def add_text_node(name: str, category: str = "历史动态实体", size: int = 30) -> str | None:
        if not name:
            return None
        node_id = f"TEXT::{name}"
        if node_id in nodes:
            nodes[node_id]["symbolSize"] = max(
                nodes[node_id].get("symbolSize", size), size
            )
            return node_id
        nodes[node_id] = {
            "id": node_id,
            "name": name,
            "symbolSize": size,
            "category": category,
            "area_code": None,
            "value": name,
        }
        return node_id

    def add_link(
        source_id: str | None,
        target_id: str | None,
        relation: str,
        edge_type: str,
        source_name: str | None = None,
        target_name: str | None = None,
        event_id: str | None = None,
        year: Any = None,
        evidence: Any = None,
        relation_type: str | None = None,
    ) -> None:
        if not source_id or not target_id or source_id == target_id:
            return
        key = (edge_type, source_id, target_id, relation, event_id or "")
        if key in link_keys:
            return
        link_keys.add(key)
        color = graph_relation_color(relation, edge_type)
        style = {"color": color, "width": 1.35, "curveness": 0.04}
        if edge_type == "dynamic":
            style = {"color": color, "width": 2.1, "curveness": 0.13}
        elif edge_type == "context":
            style = {"color": color, "width": 1.0, "type": "dashed", "curveness": 0.05}
        links.append(
            {
                "source": source_id,
                "target": target_id,
                "source_name": source_name or nodes.get(source_id, {}).get("name", source_id),
                "target_name": target_name or nodes.get(target_id, {}).get("name", target_id),
                "name": relation,
                "relation": relation,
                "relation_type": relation_type or "",
                "event_id": event_id,
                "year": year,
                "evidence": evidence,
                "edge_type": edge_type,
                "color": color,
                "lineStyle": style,
            }
        )

    def add_static_path(code_value: str | None) -> str | None:
        path_rows = static_path_rows(code_value, root_code)
        previous_id: str | None = None
        current_id: str | None = None
        for row in path_rows:
            current_id = add_static_node(str(row["admin_code"]))
            if previous_id and current_id:
                add_link(previous_id, current_id, "下辖", "static")
            previous_id = current_id
        return current_id or (add_static_node(str(code_value)) if code_value else None)

    def code_inside_root(code_value: str | None) -> bool:
        if not root_code or not code_value:
            return True
        return any(
            str(row.get("admin_code")) == root_code
            for row in static_path_rows(str(code_value))
        )

    def add_dynamic_endpoint(name: str, is_focus: bool = False) -> str | None:
        if not name:
            return None
        if name.startswith("无旧区划") or name.startswith("无新区划"):
            return add_text_node(name, "变更缺省端点", 24)
        resolved_code = resolve_area_code_by_name(name)
        if resolved_code and code_inside_root(resolved_code):
            node_id = add_static_path(resolved_code)
            if node_id and is_focus:
                nodes[node_id]["symbolSize"] = max(nodes[node_id]["symbolSize"], 42)
            return node_id
        return add_text_node(name, "历史动态实体", 34 if is_focus else 30)

    def connected_component_ids(start_id: str) -> set[str]:
        adjacency: dict[str, set[str]] = {}
        for link in links:
            source = str(link.get("source") or "")
            target = str(link.get("target") or "")
            if not source or not target:
                continue
            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)
        visited = {start_id}
        stack = [start_id]
        while stack:
            current = stack.pop()
            for next_id in adjacency.get(current, set()):
                if next_id not in visited:
                    visited.add(next_id)
                    stack.append(next_id)
        return visited

    def static_context_parent_code(evidence: Any, endpoint_names: set[str]) -> str | None:
        if not root_code:
            return None
        evidence_text = str(evidence or "")
        candidates: list[tuple[int, str]] = []
        level_scores = {"prefecture": 400, "county": 200}
        for code_value, row in static_area_index()["by_code"].items():
            level = str(row.get("admin_level") or "")
            if level not in level_scores:
                continue
            name = str(row.get("admin_name") or "")
            if not name or name in endpoint_names or name not in evidence_text:
                continue
            if not code_inside_root(str(code_value)):
                continue
            candidates.append((level_scores[level] + len(name), str(code_value)))
        if not candidates:
            return root_code
        candidates.sort(reverse=True)
        return candidates[0][1]

    def preferred_component_anchor(component_ids: set[str]) -> str | None:
        text_ids = [
            node_id
            for node_id in component_ids
            if str(node_id).startswith("TEXT::")
            and not str(nodes.get(node_id, {}).get("name") or "").startswith("无新区划")
            and not str(nodes.get(node_id, {}).get("name") or "").startswith("无旧区划")
        ]
        if text_ids:
            text_ids.sort(key=lambda node_id: (nodes[node_id].get("category") != "历史动态实体", nodes[node_id].get("name", "")))
            return text_ids[0]
        return next(iter(component_ids), None)

    def connect_orphan_dynamic_components() -> None:
        if not root_code or root_code not in nodes:
            return
        reachable = connected_component_ids(root_code)
        remaining = set(nodes) - reachable
        handled: set[str] = set()
        while remaining:
            start_id = remaining.pop()
            component = connected_component_ids(start_id)
            remaining -= component
            edge_candidates = [
                edge
                for edge in dynamic_edges
                if edge.get("source_id") in component or edge.get("target_id") in component
            ]
            if not edge_candidates:
                continue
            endpoint_names = {
                str(nodes.get(node_id, {}).get("name") or "")
                for node_id in component
                if node_id in nodes
            }
            edge = edge_candidates[0]
            parent_code = static_context_parent_code(edge.get("evidence"), endpoint_names)
            parent_id = add_static_path(parent_code)
            child_id = preferred_component_anchor(component)
            if not parent_id or not child_id or child_id in handled:
                continue
            add_link(
                parent_id,
                child_id,
                "下辖",
                "static",
                source_name=nodes.get(parent_id, {}).get("name"),
                target_name=nodes.get(child_id, {}).get("name"),
            )
            handled.add(child_id)

    if area:
        add_static_path(area["code"])
        for child in children_of(area["code"], None)[:60]:
            child_id = add_static_path(child["code"])
            add_link(str(area["code"]), child_id, "下辖", "static")

    for item in related_relations_for_area(keyword, limit=limit):
        source = str(item.get("subject") or "").strip()
        target = str(item.get("object") or "").strip()
        relation = str(item.get("relation") or "").strip()
        if not source or not target or not relation:
            continue
        if relation == "发生调整" and target.endswith("事件"):
            continue
        source_hit = keyword and (keyword in source or source in keyword)
        target_hit = keyword and (keyword in target or target in keyword)
        source_id = add_dynamic_endpoint(source, bool(source_hit))
        target_id = add_dynamic_endpoint(target, bool(target_hit))
        add_link(
            source_id,
            target_id,
            relation,
            "dynamic",
            source_name=source,
            target_name=target,
            event_id=item.get("event_id"),
            year=item.get("year"),
            evidence=item.get("evidence"),
            relation_type=(
                item.get("semantic_relation_type_labels")
                if view == "semantic"
                else item.get("relation_type_labels")
            )
            or item.get("type_label"),
        )
        if source_id and target_id:
            dynamic_edges.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "source": source,
                    "target": target,
                    "evidence": item.get("evidence"),
                }
            )

    connect_orphan_dynamic_components()

    return {
        "version": view,
        "nodes": list(nodes.values()),
        "links": links,
        "categories": [
            {"name": "查询区划"},
            {"name": "静态区划"},
            {"name": "历史动态实体"},
            {"name": "变更缺省端点"},
        ],
    }


def graph_overview() -> dict[str, Any]:
    with connect_db() as db:
        provinces = rows(
            db,
            """
            SELECT admin_code, admin_name, admin_level, full_path
            FROM static_admin_nodes
            WHERE admin_level = 'province'
            ORDER BY admin_code
            """,
        )
        cities = rows(
            db,
            """
            SELECT child.admin_code,
                   child.admin_name,
                   child.admin_level,
                   child.full_path,
                   parent.admin_code AS parent_code
            FROM static_affiliation_triples AS edge
            JOIN static_admin_nodes AS child ON child.node_id = edge.subject_node_id
            JOIN static_admin_nodes AS parent ON parent.node_id = edge.object_node_id
            WHERE child.admin_level = 'prefecture'
            ORDER BY child.admin_code
            """,
        )
    nodes = [
        {
            "id": "中国",
            "name": "中国",
            "symbolSize": 74,
            "category": "国家",
            "area_code": None,
            "value": "全国行政区划母图谱",
            "label": {"show": True},
        }
    ]
    links: list[dict[str, Any]] = []
    for province in provinces:
        code = str(province["admin_code"])
        count = count_records_by_keyword(str(province["admin_name"]))
        nodes.append(
            {
                "id": code,
                "name": province["admin_name"],
                "symbolSize": 38 + min(count, 18),
                "category": "省级区划",
                "area_code": code,
                "value": f"相关变更 {count} 条",
                "label": {"show": True},
            }
        )
        links.append(
            {"source": "中国", "target": code, "relation": "下辖", "name": "下辖"}
        )
    for city in cities:
        nodes.append(
            {
                "id": str(city["admin_code"]),
                "name": city["admin_name"],
                "symbolSize": 20,
                "category": "地市级区划",
                "area_code": str(city["admin_code"]),
                "value": city["full_path"],
                "label": {"show": False},
            }
        )
        links.append(
            {
                "source": str(city["parent_code"]),
                "target": str(city["admin_code"]),
                "relation": "下辖",
                "name": "下辖",
            }
        )
    return {
        "nodes": nodes,
        "links": links,
        "categories": [{"name": "国家"}, {"name": "省级区划"}, {"name": "地市级区划"}],
    }


def count_records_by_keyword(keyword: str) -> int:
    with connect_db() as db:
        return int(
            scalar(
                db,
                """
                SELECT COUNT(*)
                FROM records
                WHERE record_text LIKE ?
                   OR before_entities LIKE ?
                   OR after_entities LIKE ?
                """,
                [like(keyword), like(keyword), like(keyword)],
            )
            or 0
        )


def forecast_series(
    target: str,
    history: list[dict[str, Any]],
    forecast: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    model_key: str,
) -> list[dict[str, Any]]:
    model_spec = FORECAST_MODEL_BY_ID.get(model_key, FORECAST_MODEL_BY_ID["chosen"])
    output = [
        {
            "target": target,
            "year": item["year"],
            "kind": "history",
            "source_year_present": item.get("source_year_present", True),
            "actual": item["actual"],
            "predicted": item["actual"],
            "predicted_lower": None,
            "predicted_upper": None,
            "model": "history",
        }
        for item in history
    ]
    interval_rows = add_forecast_intervals(forecast, metrics, model_spec)
    for item in interval_rows:
        output.append(
            {
                "target": target,
                "year": item["forecast_year"],
                "kind": "forecast",
                "source_year_present": False,
                "actual": None,
                "predicted": item["selected_forecast"],
                "predicted_lower": item["lower95"],
                "predicted_upper": item["upper95"],
                "model": item["selected_model"],
                "holt_winters": item.get("holt_winters_forecast"),
                "linear_regression": item.get("linear_regression_forecast"),
                "exogenous_poisson": item.get("exogenous_regression_forecast"),
                "chosen_forecast": item.get("chosen_forecast"),
            }
        )
    return output


def add_forecast_intervals(
    forecast: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    model_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    metric_map = {(item.get("target"), item.get("model")): item for item in metrics}
    for item in forecast:
        model = model_spec.get("metric_model") or item.get("chosen_model")
        forecast_column = str(model_spec.get("column") or "chosen_forecast")
        raw_value = item.get(forecast_column)
        if raw_value in (None, ""):
            raw_value = item.get("chosen_forecast")
            model = item.get("chosen_model")
        metric = metric_map.get((item.get("target"), model)) or (
            metrics[0] if metrics else {}
        )
        rmse = float(metric.get("rmse") or 0)
        horizon = float(item.get("horizon") or 1)
        value = float(raw_value or 0)
        interval = 1.96 * rmse * math.sqrt(max(horizon, 1)) * 0.35
        item["selected_forecast"] = round(value, 4)
        item["selected_model"] = model
        item["lower95"] = max(0, round(value - interval, 4))
        item["upper95"] = round(value + interval, 4)
    return forecast


app = create_app()


def main() -> None:
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    main()
