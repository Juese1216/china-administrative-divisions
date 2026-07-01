"""生成知识图谱导入文件和可视化数据。

本脚本读取主线 RE 输出，生成 Neo4j 可导入的节点表、边表、Cypher 脚本，
并额外生成静态区划树和动态时间轴数据。图谱包含两部分：

1. 静态图谱：来自行政区划编码表的省、市、县、乡镇街道隶属关系。
2. 动态图谱：来自年度变更记录的旧区划、新区划和事件关系。

运行示例：

    conda run --no-capture-output -n nlpEnv python src/graph.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RE_DIR = Path("data/processed/relation_extraction")
DEFAULT_CLASSIFICATION_DIR = Path("data/processed/classification")
DEFAULT_OUTPUT_DIR = Path("data/processed/knowledge_graph")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 Neo4j 知识图谱导入文件。")
    parser.add_argument("--re-dir", type=Path, default=DEFAULT_RE_DIR)
    parser.add_argument(
        "--classification-dir", type=Path, default=DEFAULT_CLASSIFICATION_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"缺少输入文件：{path}")
    return pd.read_csv(path)


def stable_id(prefix: str, text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def normalize_entity_text(text: Any) -> str:
    value = clean_cell(text)
    if not value:
        return ""
    if value.startswith("无旧区划") or value.startswith("无新区划"):
        return value
    return value.strip(" ，。；;、")


def split_entities(value: Any) -> list[str]:
    text = clean_cell(value)
    if not text:
        return []
    return [part.strip() for part in text.split(" / ") if part.strip()]


def build_name_index(admin_nodes: pd.DataFrame) -> dict[str, str]:
    counts = admin_nodes["admin_name"].astype(str).value_counts().to_dict()
    index: dict[str, str] = {}
    for _, row in admin_nodes.iterrows():
        name = clean_cell(row["admin_name"])
        if name and counts.get(name, 0) == 1:
            index[name] = clean_cell(row["node_id"])
    return index


def entity_node_id(entity_text: str, name_index: dict[str, str]) -> str:
    text = normalize_entity_text(entity_text)
    if text in name_index:
        return name_index[text]
    return stable_id("ENT", text)


def build_nodes(
    records: pd.DataFrame,
    dynamic_triples: pd.DataFrame,
    static_admin_nodes: pd.DataFrame,
    name_index: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in static_admin_nodes.iterrows():
        rows.append(
            {
                "node_id": clean_cell(row["node_id"]),
                "node_type": "AdminArea",
                "name": clean_cell(row["admin_name"]),
                "admin_code": clean_cell(row["admin_code"]),
                "admin_level": clean_cell(row["admin_level"]),
                "year": "",
                "type_label": "",
                "text": clean_cell(row["full_path"]),
                "source": clean_cell(row["source_table"]),
            }
        )

    for _, row in records.iterrows():
        rows.append(
            {
                "node_id": f"REC_{clean_cell(row['record_id'])}",
                "node_type": "Record",
                "name": clean_cell(row["record_id"]),
                "admin_code": "",
                "admin_level": "",
                "year": int(row["year"]),
                "type_label": clean_cell(row["type_label"]),
                "text": clean_cell(row["record_text"]),
                "source": clean_cell(row["source_file"]),
            }
        )

    entity_texts: set[str] = set()
    for column in ["subject", "object"]:
        for value in dynamic_triples[column].tolist():
            text = normalize_entity_text(value)
            if text:
                entity_texts.add(text)
    for _, row in records.iterrows():
        for text in split_entities(row.get("before_entities", "")) + split_entities(
            row.get("after_entities", "")
        ):
            entity_texts.add(normalize_entity_text(text))

    existing_node_ids = {
        clean_cell(row["node_id"]) for _, row in static_admin_nodes.iterrows()
    }
    for text in sorted(entity_texts):
        node_id = entity_node_id(text, name_index)
        if node_id in existing_node_ids:
            continue
        rows.append(
            {
                "node_id": node_id,
                "node_type": "TextEntity",
                "name": text,
                "admin_code": "",
                "admin_level": "",
                "year": "",
                "type_label": "",
                "text": text,
                "source": "records",
            }
        )

    return pd.DataFrame(rows).drop_duplicates("node_id")


def build_edges(
    records: pd.DataFrame,
    dynamic_triples: pd.DataFrame,
    static_affiliation_triples: pd.DataFrame,
    name_index: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in static_affiliation_triples.iterrows():
        rows.append(
            {
                "edge_id": f"KG_E{len(rows) + 1:08d}",
                "start_id": clean_cell(row["subject_node_id"]),
                "end_id": clean_cell(row["object_node_id"]),
                "relation": "隶属于",
                "edge_type": "STATIC_AFFILIATION",
                "year": "",
                "effective_time": "",
                "type_label": "",
                "relation_type_ids": "",
                "relation_type_labels": "",
                "record_id": "",
                "evidence_text": "",
                "source": clean_cell(row["source_table"]),
            }
        )

    for _, row in dynamic_triples.iterrows():
        subject = normalize_entity_text(row["subject"])
        obj = normalize_entity_text(row["object"])
        if not subject or not obj:
            continue
        rows.append(
            {
                "edge_id": f"KG_E{len(rows) + 1:08d}",
                "start_id": entity_node_id(subject, name_index),
                "end_id": entity_node_id(obj, name_index),
                "relation": clean_cell(row["predicate"]),
                "edge_type": "DYNAMIC_RELATION",
                "year": int(row["year"]),
                "effective_time": clean_cell(row["effective_time"]),
                "type_label": clean_cell(row["type_label"]),
                "relation_type_ids": clean_cell(row.get("relation_type_ids", "")),
                "relation_type_labels": clean_cell(
                    row.get("relation_type_labels", "")
                ),
                "record_id": clean_cell(row["record_id"]),
                "evidence_text": clean_cell(row["evidence_text"]),
                "source": clean_cell(row["source_relation_type"]),
            }
        )

    for _, row in records.iterrows():
        record_node = f"REC_{clean_cell(row['record_id'])}"
        related = split_entities(row.get("before_entities", "")) + split_entities(
            row.get("after_entities", "")
        )
        for text in related:
            node_id = entity_node_id(text, name_index)
            rows.append(
                {
                    "edge_id": f"KG_E{len(rows) + 1:08d}",
                    "start_id": record_node,
                    "end_id": node_id,
                    "relation": "涉及",
                    "edge_type": "RECORD_ENTITY",
                    "year": int(row["year"]),
                    "effective_time": clean_cell(row["effective_time"]),
                    "type_label": clean_cell(row["type_label"]),
                    "relation_type_ids": clean_cell(row.get("relation_type_ids", "")),
                    "relation_type_labels": clean_cell(row.get("type_label", "")),
                    "record_id": clean_cell(row["record_id"]),
                    "evidence_text": clean_cell(row["record_text"]),
                    "source": "records",
                }
            )

    return pd.DataFrame(rows)


def build_static_tree(
    static_nodes: pd.DataFrame, static_edges: pd.DataFrame
) -> dict[str, Any]:
    node_lookup = {
        clean_cell(row["node_id"]): {
            "id": clean_cell(row["node_id"]),
            "name": clean_cell(row["admin_name"]),
            "code": clean_cell(row["admin_code"]),
            "level": clean_cell(row["admin_level"]),
            "children": [],
        }
        for _, row in static_nodes.iterrows()
    }
    for _, row in static_edges.iterrows():
        child_id = clean_cell(row["subject_node_id"])
        parent_id = clean_cell(row["object_node_id"])
        if child_id in node_lookup and parent_id in node_lookup:
            node_lookup[parent_id]["children"].append(node_lookup[child_id])

    def sort_children(node: dict[str, Any]) -> None:
        node["children"].sort(
            key=lambda item: (item.get("code", ""), item.get("name", ""))
        )
        for child in node["children"]:
            sort_children(child)

    root = node_lookup.get(
        "CN",
        {"id": "CN", "name": "中国", "code": "CN", "level": "country", "children": []},
    )
    sort_children(root)
    return root


def build_timeline(records: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, row in records.sort_values(["year", "record_id"]).iterrows():
        rows.append(
            {
                "record_id": clean_cell(row["record_id"]),
                "year": int(row["year"]),
                "type_label": clean_cell(row["type_label"]),
                "before_entities": split_entities(row.get("before_entities", "")),
                "after_entities": split_entities(row.get("after_entities", "")),
                "text": clean_cell(row["record_text"]),
            }
        )
    return rows


def write_cypher(output_dir: Path) -> None:
    cypher = """// Neo4j 导入脚本
// 使用方法：
// 1. 将 neo4j_nodes.csv 和 neo4j_edges.csv 放到 Neo4j import 目录。
// 2. 在 Neo4j Browser 或 cypher-shell 中执行本文件。

CREATE CONSTRAINT kg_node_id IF NOT EXISTS
FOR (n:KGNode) REQUIRE n.node_id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_nodes.csv' AS row
MERGE (n:KGNode {node_id: row.node_id})
SET n.node_type = row.node_type,
    n.name = row.name,
    n.admin_code = row.admin_code,
    n.admin_level = row.admin_level,
    n.year = row.year,
    n.type_label = row.type_label,
    n.text = row.text,
    n.source = row.source;

LOAD CSV WITH HEADERS FROM 'file:///neo4j_edges.csv' AS row
MATCH (a:KGNode {node_id: row.start_id})
MATCH (b:KGNode {node_id: row.end_id})
FOREACH (_ IN CASE WHEN row.edge_type = 'STATIC_AFFILIATION' THEN [1] ELSE [] END |
  MERGE (a)-[r:BELONGS_TO {edge_id: row.edge_id}]->(b)
  SET r.relation = row.relation,
      r.source = row.source
)
FOREACH (_ IN CASE WHEN row.edge_type = 'DYNAMIC_RELATION' THEN [1] ELSE [] END |
  MERGE (a)-[r:DYNAMIC_RELATION {edge_id: row.edge_id}]->(b)
  SET r.relation = row.relation,
      r.year = toInteger(row.year),
      r.effective_time = row.effective_time,
      r.type_label = row.type_label,
      r.relation_type_ids = row.relation_type_ids,
      r.relation_type_labels = row.relation_type_labels,
      r.record_id = row.record_id,
      r.evidence_text = row.evidence_text,
      r.source = row.source
)
FOREACH (_ IN CASE WHEN row.edge_type = 'RECORD_ENTITY' THEN [1] ELSE [] END |
  MERGE (a)-[r:MENTIONS {edge_id: row.edge_id}]->(b)
  SET r.relation = row.relation,
      r.year = toInteger(row.year),
      r.type_label = row.type_label,
      r.relation_type_ids = row.relation_type_ids,
      r.relation_type_labels = row.relation_type_labels,
      r.record_id = row.record_id,
      r.evidence_text = row.evidence_text,
      r.source = row.source
);

// 查询示例：查看某一年以前的动态演变
// MATCH (a:KGNode)-[r:DYNAMIC_RELATION]->(b:KGNode)
// WHERE r.year <= 2010
// RETURN a.name, r.relation, b.name, r.year, r.type_label
// ORDER BY r.year;

// 查询示例：查看某个区划的静态上级链
// MATCH path = (n:KGNode {name: '呈贡区'})-[:BELONGS_TO*1..4]->(parent:KGNode)
// RETURN path;
"""
    (output_dir / "neo4j_import.cypher").write_text(cypher, encoding="utf-8")


def write_outputs(
    output_dir: Path,
    outputs: dict[str, pd.DataFrame],
    static_tree: dict[str, Any],
    timeline: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
    (output_dir / "static_tree.json").write_text(
        json.dumps(static_tree, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "timeline.json").write_text(
        json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_cypher(output_dir)


def main() -> None:
    args = parse_args()
    records = read_csv(args.re_dir / "records.csv")
    dynamic_triples = read_csv(args.re_dir / "dynamic_triples.csv")
    static_admin_nodes = read_csv(args.re_dir / "static_admin_nodes.csv")
    static_affiliation_triples = read_csv(
        args.re_dir / "static_affiliation_triples.csv"
    )

    name_index = build_name_index(static_admin_nodes)
    nodes = build_nodes(records, dynamic_triples, static_admin_nodes, name_index)
    edges = build_edges(
        records, dynamic_triples, static_affiliation_triples, name_index
    )
    static_tree = build_static_tree(static_admin_nodes, static_affiliation_triples)
    timeline = build_timeline(records)

    overview = pd.DataFrame(
        [
            {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "static_node_count": len(static_admin_nodes),
                "static_edge_count": len(static_affiliation_triples),
                "dynamic_edge_count": len(dynamic_triples),
                "record_count": len(records),
            }
        ]
    )

    outputs = {
        "neo4j_nodes.csv": nodes,
        "neo4j_edges.csv": edges,
        "overview.csv": overview,
    }
    write_outputs(args.output_dir, outputs, static_tree, timeline)

    print(f"知识图谱输出目录：{args.output_dir}")
    print(overview.to_string(index=False))


if __name__ == "__main__":
    main()
