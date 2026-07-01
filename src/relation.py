"""关系抽取主流程：关系归档、句内关系抽取和记录级三元组整理。

这个文件已经把原来的关系类别归档、规则抽取、记录整理三段实现整合到
同一个顶层脚本中，不再依赖额外子包。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

schema_DEFAULT_STATS_DIR = Path("data/processed/relation_statistics")
schema_DEFAULT_PATTERN_CSV = Path(
    "data/processed/ner_rule_uie/sentence_pattern_summary.csv"
)
schema_DEFAULT_OUTPUT_DIR = Path("data/processed/relation_schema")
schema_RELATION_DEFINITIONS: list[dict[str, Any]] = [
    {
        "relation_type_id": "REVOKE_ADMIN",
        "relation_type_name": "撤销建制",
        "relation_group": "建置变化",
        "re_status": "core",
        "trigger_words": ["撤销", "撤"],
        "cue_column": "has_revoke",
        "sentence_regex": "撤销",
        "subject_hint": "被撤销的旧行政区划",
        "object_hint": "撤销事件或空对象；若同句有设立，后续可关联新行政区划",
        "triple_template": "旧行政区划 -> 撤销建制 -> 变更事件",
        "pattern_hint": "撤销 + ADMIN_AREA；常与“设立”同句出现。",
        "notes": "高频核心关系，RE 时通常作为事件起点。",
    },
    {
        "relation_type_id": "ESTABLISH_ADMIN",
        "relation_type_name": "设立建制",
        "relation_group": "建置变化",
        "re_status": "core",
        "trigger_words": [
            "设立",
            "增设",
            "新设",
            "新设立",
            "设",
            "新建",
            "组建",
            "恢复设立",
        ],
        "cue_column": "has_establish",
        "sentence_regex": "设立|增设|新设|新建|组建|恢复设立",
        "subject_hint": "新设行政区划；上级行政区划常由上下文或区划代码补齐",
        "object_hint": "设立事件或上级行政区划",
        "triple_template": "新行政区划 -> 设立建制 -> 变更事件",
        "pattern_hint": "设立 + 地级/县级/新的 + ADMIN_AREA；也可能写作“新设”。",
        "notes": "高频核心关系，常与撤销、行政区域承继、政府驻地同句或相邻句出现。",
    },
    {
        "relation_type_id": "RENAME_ADMIN",
        "relation_type_name": "名称变更",
        "relation_group": "名称变化",
        "re_status": "core",
        "trigger_words": ["更名", "更名为", "改名", "恢复为"],
        "cue_column": "has_rename",
        "sentence_regex": "更名|改名|恢复.*?名",
        "subject_hint": "旧行政区划名称",
        "object_hint": "新行政区划名称",
        "triple_template": "旧行政区划 -> 更名为 -> 新行政区划",
        "pattern_hint": "将/把 + ADMIN_AREA + 更名为 + ADMIN_AREA。",
        "notes": "PaddleNLP 中“为”会和“更名”共现，但“为”本身不是名称变更关系。",
    },
    {
        "relation_type_id": "MERGE_ADMIN",
        "relation_type_name": "合并建制",
        "relation_group": "建置变化",
        "re_status": "core",
        "trigger_words": ["合并"],
        "cue_column": "",
        "sentence_regex": "合并",
        "subject_hint": "合并前的一个或多个行政区划",
        "object_hint": "合并后的行政区划",
        "triple_template": "旧行政区划 -> 合并为 -> 新行政区划",
        "pattern_hint": "ADMIN_AREA 列表 + 合并/合并设立/合并组建 + ADMIN_AREA。",
        "notes": "低频但语义明确，下一步 RE 可先支持简单句式。",
    },
    {
        "relation_type_id": "TRANSFER_ADMIN",
        "relation_type_name": "区域划转",
        "relation_group": "隶属调整",
        "re_status": "core",
        "trigger_words": ["划归", "划入", "划出", "并入", "划...归"],
        "cue_column": "has_transfer",
        "sentence_regex": "划归|划入|划出|并入|划.{0,6}?归",
        "subject_hint": "被划转的行政区划或区域",
        "object_hint": "划入后的目标行政区划",
        "triple_template": "被划转区域 -> 划归 -> 目标行政区划",
        "pattern_hint": "将/从 + ADMIN_AREA/区域列表 + 划归/划入 + ADMIN_AREA + 管辖。",
        "notes": "切词有时把“划归”拆成“划/归”，所以触发词表保留拆分形态。",
    },
    {
        "relation_type_id": "JURISDICTION_ADMIN",
        "relation_type_name": "管辖隶属",
        "relation_group": "隶属调整",
        "re_status": "core",
        "trigger_words": ["管辖", "辖", "辖原", "原属", "所属"],
        "cue_column": "has_govern",
        "sentence_regex": "管辖|辖原|辖|原属|所属",
        "subject_hint": "上级行政区划",
        "object_hint": "被管辖的下级行政区划或区域列表",
        "triple_template": "上级行政区划 -> 管辖 -> 下级行政区划",
        "pattern_hint": "ADMIN_AREA + 辖/管辖 + ADMIN_AREA 列表。",
        "notes": "用于构建行政层级和调整后的下辖关系；与划归、直辖、代管存在交叉。",
    },
    {
        "relation_type_id": "DIRECT_ADMIN",
        "relation_type_name": "省级直辖",
        "relation_group": "隶属调整",
        "re_status": "core",
        "trigger_words": ["直辖"],
        "cue_column": "",
        "sentence_regex": "直辖",
        "subject_hint": "被直辖行政区划",
        "object_hint": "直接管辖的省级行政区划",
        "triple_template": "被直辖行政区划 -> 由...直辖 -> 省级行政区划",
        "pattern_hint": "ADMIN_AREA + 由 + 省/自治区 + 直辖。",
        "notes": "从管辖类中拆出，便于区分省直管县/县级市等特殊隶属关系。",
    },
    {
        "relation_type_id": "ENTRUST_ADMIN",
        "relation_type_name": "委托代管",
        "relation_group": "隶属调整",
        "re_status": "core",
        "trigger_words": ["代管"],
        "cue_column": "",
        "sentence_regex": "代管",
        "subject_hint": "被代管行政区划",
        "object_hint": "代管的地级市或行政区划",
        "triple_template": "被代管行政区划 -> 由...代管 -> 代管行政区划",
        "pattern_hint": "ADMIN_AREA + 由 + ADMIN_AREA + 代管。",
        "notes": "与直辖类似，属于管辖关系的特殊形态。",
    },
    {
        "relation_type_id": "GOV_RESIDENCE",
        "relation_type_name": "政府驻地",
        "relation_group": "驻地变化",
        "re_status": "core",
        "trigger_words": ["驻", "人民政府驻", "政府驻", "驻地"],
        "cue_column": "has_residence",
        "sentence_regex": "人民政府驻|政府驻|驻地|驻",
        "subject_hint": "人民政府机构或对应行政区划",
        "object_hint": "驻地地址或驻地行政区划",
        "triple_template": "人民政府/行政区划 -> 政府驻地 -> 地址或行政区划",
        "pattern_hint": "ORG/GOV_ORG + 驻 + ADDRESS/ADMIN_AREA。",
        "notes": "“驻”频次很高，其中一部分只是说明新设地区的驻地，不一定表示迁移。",
    },
    {
        "relation_type_id": "RESIDENCE_TRANSFER",
        "relation_type_name": "驻地迁移",
        "relation_group": "驻地变化",
        "re_status": "core",
        "trigger_words": ["迁至", "迁移", "迁"],
        "cue_column": "",
        "sentence_regex": "迁至|迁移|迁",
        "subject_hint": "人民政府驻地或对应行政区划",
        "object_hint": "新驻地；若有“由”，可同时抽取旧驻地",
        "triple_template": "人民政府/行政区划 -> 驻地迁至 -> 新驻地",
        "pattern_hint": "人民政府驻地由 + OLD_LOCATION + 迁至/迁移至 + NEW_LOCATION。",
        "notes": "属于政府驻地关系的变更型子类，优先单独抽取。",
    },
    {
        "relation_type_id": "AREA_INHERITANCE",
        "relation_type_name": "行政区域承继",
        "relation_group": "区域范围",
        "re_status": "core",
        "trigger_words": ["以...行政区域为...行政区域", "为"],
        "cue_column": "",
        "sentence_regex": "以.*?行政区域为.*?行政区域",
        "subject_hint": "新行政区划",
        "object_hint": "原行政区划或原行政区域",
        "triple_template": "新行政区划 -> 承继行政区域 -> 原行政区划",
        "pattern_hint": "以 + 原ADMIN_AREA + 的行政区域 + 为 + 新ADMIN_AREA + 的行政区域。",
        "notes": "“为”是高频辅助词，不能直接当关系；只有落在该句式中才归入本类。",
    },
    {
        "relation_type_id": "ADJUSTMENT_EVENT",
        "relation_type_name": "行政区划调整事件",
        "relation_group": "综合事件",
        "re_status": "event",
        "trigger_words": ["调整", "行政区划调整", "作如下调整", "调整后", "变更"],
        "cue_column": "has_adjustment",
        "sentence_regex": "行政区划调整|调整后|作如下调整|调整|变更",
        "subject_hint": "被调整的行政区划或区域",
        "object_hint": "调整事件；具体对象由划归/管辖/设立等子关系补充",
        "triple_template": "行政区划/区域 -> 发生调整 -> 变更事件",
        "pattern_hint": "调整 + ADMIN_AREA + 行政区划；后文常出现多个子关系。",
        "notes": "建议作为事件路由标签，不强行当作单一二元关系。",
    },
    {
        "relation_type_id": "SCOPE_CONSTRAINT",
        "relation_type_name": "范围包含排除",
        "relation_group": "区域范围",
        "re_status": "auxiliary",
        "trigger_words": ["不含", "不包括", "包括", "除外", "位于", "起", "转向"],
        "cue_column": "",
        "sentence_regex": "不含|不包括|包括|除外|位于|[东西南北]起|转向",
        "subject_hint": "被描述的行政区域范围",
        "object_hint": "被包含或排除的区域",
        "triple_template": "行政区域范围 -> 包含/排除 -> 区域",
        "pattern_hint": "括号或补充说明中的“不含/包括/除外”。",
        "notes": "不是主关系，但对精细化区域边界很有用，下一步可先作为修饰信息。",
    },
]
schema_EXCLUDED_CANDIDATE_KEYWORDS = {
    "公布",
    "执行",
    "有关",
    "同意",
    "批准",
    "备案",
    "如下",
    "管理",
    "简称",
    "行使",
    "相关",
}
schema_CONTEXT_DEPENDENT_CANDIDATES = {"恢复"}


def schema_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"缺少输入文件：{path}")
    return pd.read_csv(path)


def schema_compact_examples(sentences: pd.Series, limit: int = 3) -> str:
    examples: list[str] = []
    seen: set[str] = set()
    for sentence in sentences.fillna("").astype(str):
        if not sentence or sentence in seen:
            continue
        examples.append(sentence)
        seen.add(sentence)
        if len(examples) >= limit:
            break
    return " || ".join(examples)


def schema_verb_stats_for_triggers(
    verb_frequency: pd.DataFrame, triggers: list[str]
) -> dict[str, Any]:
    rows = verb_frequency[verb_frequency["verb"].astype(str).isin(triggers)].copy()
    if rows.empty:
        return {
            "paddle_verb_count": 0,
            "paddle_sentence_count": 0,
            "matched_paddle_verbs": "",
            "first_year_from_verbs": "",
            "last_year_from_verbs": "",
        }
    return {
        "paddle_verb_count": int(rows["count"].sum()),
        "paddle_sentence_count": int(rows["sentence_count"].sum()),
        "matched_paddle_verbs": " / ".join(rows["verb"].astype(str).tolist()),
        "first_year_from_verbs": int(rows["first_year"].min()),
        "last_year_from_verbs": int(rows["last_year"].max()),
    }


def schema_sentence_evidence(
    sentence_patterns: pd.DataFrame, cue_column: str, sentence_regex: str
) -> tuple[int, str]:
    if cue_column and cue_column in sentence_patterns.columns:
        mask = sentence_patterns[cue_column].astype(int).eq(1)
    else:
        mask = (
            sentence_patterns["sentence"]
            .fillna("")
            .astype(str)
            .str.contains(sentence_regex, regex=True)
        )
    rows = sentence_patterns[mask].copy()
    return (int(len(rows)), schema_compact_examples(rows["sentence"]))


def schema_build_relation_schema(
    verb_frequency: pd.DataFrame, sentence_patterns: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for order, definition in enumerate(schema_RELATION_DEFINITIONS, start=1):
        trigger_words = definition["trigger_words"]
        token_triggers = [
            word for word in trigger_words if re.fullmatch("[\\u4e00-\\u9fff]+", word)
        ]
        stats = schema_verb_stats_for_triggers(verb_frequency, token_triggers)
        evidence_count, examples = schema_sentence_evidence(
            sentence_patterns, definition["cue_column"], definition["sentence_regex"]
        )
        rows.append(
            {
                "priority": order,
                "relation_type_id": definition["relation_type_id"],
                "relation_type_name": definition["relation_type_name"],
                "relation_group": definition["relation_group"],
                "re_status": definition["re_status"],
                "trigger_words": " / ".join(trigger_words),
                "matched_paddle_verbs": stats["matched_paddle_verbs"],
                "paddle_verb_count": stats["paddle_verb_count"],
                "paddle_sentence_count": stats["paddle_sentence_count"],
                "sentence_evidence_count": evidence_count,
                "first_year_from_verbs": stats["first_year_from_verbs"],
                "last_year_from_verbs": stats["last_year_from_verbs"],
                "subject_hint": definition["subject_hint"],
                "object_hint": definition["object_hint"],
                "triple_template": definition["triple_template"],
                "pattern_hint": definition["pattern_hint"],
                "example_sentences": examples,
                "notes": definition["notes"],
            }
        )
    return pd.DataFrame(rows)


def schema_build_trigger_archive(verb_frequency: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for definition in schema_RELATION_DEFINITIONS:
        for trigger in definition["trigger_words"]:
            freq_row = verb_frequency[verb_frequency["verb"].astype(str).eq(trigger)]
            rows.append(
                {
                    "relation_type_id": definition["relation_type_id"],
                    "relation_type_name": definition["relation_type_name"],
                    "relation_group": definition["relation_group"],
                    "re_status": definition["re_status"],
                    "trigger_word": trigger,
                    "is_exact_paddle_verb": int(not freq_row.empty),
                    "paddle_count": (
                        int(freq_row["count"].iloc[0]) if not freq_row.empty else 0
                    ),
                    "paddle_sentence_count": (
                        int(freq_row["sentence_count"].iloc[0])
                        if not freq_row.empty
                        else 0
                    ),
                    "trigger_note": (
                        "短语/正则触发词" if freq_row.empty else "Paddle 动词统计命中"
                    ),
                }
            )
    return pd.DataFrame(rows)


def schema_classify_candidate(
    verb: str, count: int, trigger_to_relation: dict[str, dict[str, str]]
) -> tuple[str, str, str]:
    if verb in trigger_to_relation:
        item = trigger_to_relation[verb]
        return (
            item["relation_type_id"],
            item["relation_type_name"],
            item["review_label"],
        )
    if verb in {"为", "至"}:
        return ("AUX_CONNECTOR", "句式连接词", "auxiliary_pattern")
    if verb in schema_CONTEXT_DEPENDENT_CANDIDATES:
        return ("CONTEXT_DEPENDENT", "上下文依赖触发词", "context_dependent")
    if verb in schema_EXCLUDED_CANDIDATE_KEYWORDS:
        return ("NOT_RE_RELATION", "公告/审批/说明词", "exclude")
    if len(verb) == 1 or count <= 3:
        return ("TOKENIZATION_NOISE", "切词噪声或关系碎片", "review_or_noise")
    return ("UNASSIGNED", "暂未归类候选", "manual_review")


def schema_build_candidate_review(relation_candidates: pd.DataFrame) -> pd.DataFrame:
    trigger_to_relation: dict[str, dict[str, str]] = {}
    for definition in schema_RELATION_DEFINITIONS:
        for trigger in definition["trigger_words"]:
            if re.fullmatch("[\\u4e00-\\u9fff]+", trigger):
                trigger_to_relation[trigger] = {
                    "relation_type_id": definition["relation_type_id"],
                    "relation_type_name": definition["relation_type_name"],
                    "review_label": definition["re_status"],
                }
    rows: list[dict[str, Any]] = []
    for _, row in relation_candidates.iterrows():
        verb = str(row["candidate_relation"])
        count = int(row["count"])
        relation_type_id, relation_type_name, review_label = schema_classify_candidate(
            verb, count, trigger_to_relation
        )
        rows.append(
            {
                "candidate_relation": verb,
                "mapped_relation_type_id": relation_type_id,
                "mapped_relation_type_name": relation_type_name,
                "review_label": review_label,
                "relation_score": row["relation_score"],
                "count": count,
                "sentence_count": row["sentence_count"],
                "near_location_rate": row["near_location_rate"],
                "representative_frames": row["representative_frames"],
                "example_sentences": row["example_sentences"],
            }
        )
    return pd.DataFrame(rows)


def schema_build_overview(
    schema: pd.DataFrame, candidate_review: pd.DataFrame
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "relation_type_count": len(schema),
                "core_relation_type_count": int(schema["re_status"].eq("core").sum()),
                "event_relation_type_count": int(schema["re_status"].eq("event").sum()),
                "auxiliary_relation_type_count": int(
                    schema["re_status"].eq("auxiliary").sum()
                ),
                "candidate_count": len(candidate_review),
                "mapped_candidate_count": int(
                    candidate_review["review_label"]
                    .isin(["core", "event", "auxiliary"])
                    .sum()
                ),
                "excluded_candidate_count": int(
                    candidate_review["review_label"].eq("exclude").sum()
                ),
                "context_dependent_candidate_count": int(
                    candidate_review["review_label"].eq("context_dependent").sum()
                ),
                "review_or_noise_candidate_count": int(
                    candidate_review["review_label"].eq("review_or_noise").sum()
                ),
                "manual_review_candidate_count": int(
                    candidate_review["review_label"].eq("manual_review").sum()
                ),
            }
        ]
    )


def schema_markdown_table(frame: pd.DataFrame) -> str:
    """把小型 DataFrame 渲染成 Markdown 表格。"""
    if frame.empty:
        return "_无记录_"
    columns = [str(column) for column in frame.columns]
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(("---" for _ in columns)) + " |")
    for _, row in frame.iterrows():
        values = []
        for column in frame.columns:
            value = "" if pd.isna(row[column]) else str(row[column])
            value = value.replace("|", "\\|").replace("\n", " ")
            values.append(value)
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def schema_write_markdown(
    output_dir: Path, schema: pd.DataFrame, overview: pd.DataFrame
) -> None:
    core_rows = schema[schema["re_status"].eq("core")]
    event_rows = schema[schema["re_status"].eq("event")]
    aux_rows = schema[schema["re_status"].eq("auxiliary")]
    lines = [
        "# 关系类别归档",
        "",
        "本文件由 `src/relation.py` 生成，用于下一步 RE（关系抽取）。",
        "",
        "## 总览",
        "",
        schema_markdown_table(overview),
        "",
        "## 核心关系类型",
        "",
        schema_markdown_table(
            core_rows[
                [
                    "relation_type_id",
                    "relation_type_name",
                    "relation_group",
                    "trigger_words",
                    "sentence_evidence_count",
                    "triple_template",
                ]
            ]
        ),
        "",
        "## 事件/辅助类型",
        "",
        schema_markdown_table(
            pd.concat([event_rows, aux_rows])[
                [
                    "relation_type_id",
                    "relation_type_name",
                    "relation_group",
                    "re_status",
                    "trigger_words",
                    "sentence_evidence_count",
                    "notes",
                ]
            ]
        ),
        "",
        "## 使用建议",
        "",
        "1. 下一步 RE 优先实现 `re_status=core` 的关系。",
        "2. `ADJUSTMENT_EVENT` 先作为事件路由标签，不必强行抽成单一二元关系。",
        "3. `AREA_INHERITANCE` 只在“以...行政区域为...行政区域”句式中触发，不要把所有“为”都当关系。",
        "4. `SCOPE_CONSTRAINT` 先作为区域范围修饰信息，后续需要精细图谱时再接入。",
        "5. `relation_candidate_review.csv` 保留了 195 个统计候选的归类结果，方便说明我们做过人工复核。",
        "",
    ]
    (output_dir / "relation_schema_archive.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


rules_DEFAULT_ENTITY_CSV = Path("data/processed/ner_rule_uie/entities.csv")
rules_DEFAULT_SENTENCE_CSV = Path(
    "data/processed/ner_rule_uie/source_sentence_records.csv"
)
rules_DEFAULT_SCHEMA_CSV = Path(
    "data/processed/relation_schema/relation_type_archive.csv"
)
rules_DEFAULT_OUTPUT_DIR = Path("data/processed/relation_details")
rules_REQUIRED_ENTITY_COLUMNS = {
    "entity_id",
    "sentence_id",
    "source_file",
    "year",
    "line_no",
    "item_no",
    "sentence",
    "entity_text",
    "entity_type",
    "confidence",
    "start_pos",
    "end_pos",
    "admin_code",
}
rules_REQUIRED_SENTENCE_COLUMNS = {
    "sentence_id",
    "source_file",
    "year",
    "line_no",
    "item_no",
    "sentence",
}
rules_REQUIRED_SCHEMA_COLUMNS = {
    "relation_type_id",
    "relation_type_name",
    "relation_group",
    "re_status",
}
rules_CLAUSE_PUNCTUATION = "，,；;。:："
rules_ADMIN_CONTEXT_SUFFIXES = ("省", "自治区", "市", "地区", "自治州", "盟")
rules_ADMIN_TEXT_SUFFIXES = (
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


@dataclass
class rules_Mention:
    """NER 输出中的一个实体。"""

    entity_id: str
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float
    admin_code: str = ""
    admin_level: str = ""
    province: str = ""
    prefecture: str = ""
    county: str = ""
    full_path: str = ""


def rules_clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def rules_read_required_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"缺少输入文件：{path}")
    frame = pd.read_csv(path)
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise SystemExit(f"{path} 缺少字段：{', '.join(missing)}")
    return frame


def rules_make_mention(row: pd.Series) -> rules_Mention:
    return rules_Mention(
        entity_id=rules_clean_cell(row["entity_id"]),
        text=rules_clean_cell(row["entity_text"]),
        entity_type=rules_clean_cell(row["entity_type"]),
        start=int(row["start_pos"]),
        end=int(row["end_pos"]),
        confidence=float(row["confidence"]),
        admin_code=rules_clean_cell(row.get("admin_code", "")),
        admin_level=rules_clean_cell(row.get("admin_level", "")),
        province=rules_clean_cell(row.get("province", "")),
        prefecture=rules_clean_cell(row.get("prefecture", "")),
        county=rules_clean_cell(row.get("county", "")),
        full_path=rules_clean_cell(row.get("full_path", "")),
    )


def rules_build_mentions_by_sentence(
    entity_frame: pd.DataFrame,
) -> dict[str, list[rules_Mention]]:
    mentions_by_sentence: dict[str, list[rules_Mention]] = {}
    sorted_entities = entity_frame.sort_values(
        ["sentence_id", "start_pos", "end_pos", "entity_id"]
    )
    for sentence_id, group in sorted_entities.groupby("sentence_id", sort=False):
        mentions = [rules_make_mention(row) for _, row in group.iterrows()]
        mentions_by_sentence[str(sentence_id)] = mentions
    return mentions_by_sentence


def rules_load_schema(schema_csv: Path) -> dict[str, dict[str, str]]:
    schema_frame = rules_read_required_csv(schema_csv, rules_REQUIRED_SCHEMA_COLUMNS)
    return {
        str(row["relation_type_id"]): {
            "relation_type_name": rules_clean_cell(row["relation_type_name"]),
            "relation_group": rules_clean_cell(row["relation_group"]),
            "re_status": rules_clean_cell(row["re_status"]),
        }
        for _, row in schema_frame.iterrows()
    }


def rules_mention_to_columns(
    mention: rules_Mention | None, prefix: str
) -> dict[str, Any]:
    if mention is None:
        return {
            f"{prefix}_entity_id": "",
            f"{prefix}_text": "",
            f"{prefix}_type": "",
            f"{prefix}_admin_code": "",
            f"{prefix}_admin_level": "",
            f"{prefix}_start_pos": "",
            f"{prefix}_end_pos": "",
            f"{prefix}_confidence": "",
        }
    return {
        f"{prefix}_entity_id": mention.entity_id,
        f"{prefix}_text": mention.text,
        f"{prefix}_type": mention.entity_type,
        f"{prefix}_admin_code": mention.admin_code,
        f"{prefix}_admin_level": mention.admin_level,
        f"{prefix}_start_pos": mention.start,
        f"{prefix}_end_pos": mention.end,
        f"{prefix}_confidence": round(mention.confidence, 4),
    }


def rules_synthetic_object_columns(
    object_id: str, object_text: str, object_type: str, prefix: str = "object"
) -> dict[str, Any]:
    return {
        f"{prefix}_entity_id": object_id,
        f"{prefix}_text": object_text,
        f"{prefix}_type": object_type,
        f"{prefix}_admin_code": "",
        f"{prefix}_admin_level": "",
        f"{prefix}_start_pos": "",
        f"{prefix}_end_pos": "",
        f"{prefix}_confidence": "",
    }


def rules_trigger_span(
    sentence: str, trigger: str, start: int | None = None
) -> tuple[int, int]:
    if start is None:
        start = sentence.find(trigger)
    if start < 0:
        return (-1, -1)
    return (start, start + len(trigger))


def rules_clause_bounds(sentence: str, pos: int) -> tuple[int, int]:
    left = 0
    for index in range(pos - 1, -1, -1):
        if sentence[index] in rules_CLAUSE_PUNCTUATION:
            left = index + 1
            break
    right = len(sentence)
    for index in range(pos, len(sentence)):
        if sentence[index] in rules_CLAUSE_PUNCTUATION:
            right = index
            break
    return (left, right)


def rules_mentions_between(
    mentions: list[rules_Mention],
    start: int,
    end: int,
    entity_types: set[str] | None = None,
) -> list[rules_Mention]:
    result = []
    for mention in mentions:
        if mention.start < start or mention.end > end:
            continue
        if entity_types is not None and mention.entity_type not in entity_types:
            continue
        result.append(mention)
    return sorted(result, key=lambda item: (item.start, item.end, item.entity_id))


def rules_admin_mentions_between(
    mentions: list[rules_Mention], start: int, end: int
) -> list[rules_Mention]:
    return rules_mentions_between(mentions, start, end, {"ADMIN_AREA"})


def rules_gov_mentions_between(
    mentions: list[rules_Mention], start: int, end: int
) -> list[rules_Mention]:
    return rules_mentions_between(mentions, start, end, {"GOV_ORG"})


def rules_residence_mentions_between(
    mentions: list[rules_Mention], start: int, end: int
) -> list[rules_Mention]:
    return rules_mentions_between(mentions, start, end, {"ADMIN_AREA", "ADDRESS"})


def rules_closest_before(
    mentions: list[rules_Mention], pos: int, entity_types: set[str] | None = None
) -> rules_Mention | None:
    candidates = [
        mention
        for mention in mentions
        if mention.end <= pos
        and (entity_types is None or mention.entity_type in entity_types)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item.end, item.start), reverse=True)[0]


def rules_first_after(
    mentions: list[rules_Mention], pos: int, entity_types: set[str] | None = None
) -> rules_Mention | None:
    candidates = [
        mention
        for mention in mentions
        if mention.start >= pos
        and (entity_types is None or mention.entity_type in entity_types)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item.start, item.end))[0]


def rules_unique_mentions(mentions: list[rules_Mention]) -> list[rules_Mention]:
    seen: set[tuple[str, int, int, str]] = set()
    result = []
    for mention in mentions:
        key = (mention.text, mention.start, mention.end, mention.entity_type)
        if key in seen:
            continue
        seen.add(key)
        result.append(mention)
    return result


def rules_is_context_admin(
    mention: rules_Mention, next_mention: rules_Mention | None, sentence: str = ""
) -> bool:
    if next_mention is None:
        return False
    if not mention.text.endswith(rules_ADMIN_CONTEXT_SUFFIXES):
        return False
    if not sentence:
        return next_mention.start == mention.end
    gap = sentence[mention.end : next_mention.start] if sentence else ""
    return gap in {"", "县级", "地级", "省级", "的县级", "的地级"}


def rules_prefer_specific_admins(
    admins: list[rules_Mention], sentence: str = ""
) -> list[rules_Mention]:
    """去掉紧贴在具体区划名前面的省、市等上下文实体。"""
    admins = rules_unique_mentions(
        sorted(admins, key=lambda item: (item.start, item.end))
    )
    if len(admins) <= 1:
        return admins
    result: list[rules_Mention] = []
    for index, mention in enumerate(admins):
        next_mention = admins[index + 1] if index + 1 < len(admins) else None
        if rules_is_context_admin(mention, next_mention, sentence):
            continue
        result.append(mention)
    return result or admins


def rules_clean_residence_text(text: str) -> str:
    text = str(text).strip(" ，。；;、")
    for prefix in ("新设立的", "新设的", "新设", "地级", "县级"):
        if text.startswith(prefix) and len(text) > len(prefix):
            text = text[len(prefix) :]
    return text.strip(" ，。；;、")


def rules_clean_text_entity(text: str) -> str:
    text = str(text).strip(" ，。；;、：:（）()")
    for prefix in (
        "将其管辖的",
        "将管辖的",
        "其管辖的",
        "管辖的",
        "以及从",
        "并从",
        "和从",
        "、从",
        "将",
        "从",
    ):
        if text.startswith(prefix) and len(text) > len(prefix):
            text = text[len(prefix) :].strip(" ，。；;、：:（）()")
    for suffix in ("管辖", "行政区域", "管辖区域"):
        if text.endswith(suffix) and len(text) > len(suffix):
            text = text[: -len(suffix)].strip(" ，。；;、：:（）()")
    return text


def rules_make_text_mention(
    sentence_row: pd.Series,
    text: str,
    start: int,
    end: int,
    label: str,
    confidence: float = 0.68,
) -> rules_Mention | None:
    text = rules_clean_text_entity(text)
    if len(text) < 2:
        return None
    sentence_id = rules_clean_cell(sentence_row["sentence_id"])
    return rules_Mention(
        entity_id=f"TXT_{sentence_id}_{label}_{start}_{end}",
        text=text,
        entity_type="ADMIN_AREA_TEXT",
        start=start,
        end=end,
        confidence=confidence,
    )


def rules_best_admin_or_text(
    sentence_row: pd.Series,
    sentence: str,
    mentions: list[rules_Mention],
    start: int,
    end: int,
    label: str,
    prefer: str = "first",
) -> rules_Mention | None:
    admins = rules_prefer_specific_admins(
        rules_admin_mentions_between(mentions, start, end), sentence
    )
    segment = rules_clean_text_entity(sentence[start:end])
    exact_admin = next(
        (item for item in admins if item.start == start and item.end == end), None
    )
    if segment.endswith(rules_ADMIN_TEXT_SUFFIXES) and exact_admin is None:
        text_mention = rules_make_text_mention(sentence_row, segment, start, end, label)
        if text_mention is not None:
            return text_mention
    if not admins:
        return rules_make_text_mention(sentence_row, segment, start, end, label)
    return admins[-1] if prefer == "last" else admins[0]


def rules_first_specific_after(
    mentions: list[rules_Mention], pos: int, end: int
) -> rules_Mention | None:
    admins = rules_prefer_specific_admins(
        rules_admin_mentions_between(mentions, pos, end)
    )
    return admins[0] if admins else None


def rules_last_specific_before(
    mentions: list[rules_Mention], start: int, pos: int
) -> rules_Mention | None:
    admins = rules_prefer_specific_admins(
        rules_admin_mentions_between(mentions, start, pos)
    )
    return admins[-1] if admins else None


def rules_last_marker_after_left(
    sentence: str, markers: list[str], left: int, right: int
) -> int:
    marker_positions = []
    for marker in markers:
        index = sentence.rfind(marker, left, right)
        if index >= 0:
            marker_positions.append(index + len(marker))
    return max(marker_positions) if marker_positions else left


def rules_transfer_subject_start(sentence: str, left: int, trigger_start: int) -> int:
    """定位普通划转句中被划转对象列表的左边界。"""
    subject_start = rules_last_marker_after_left(
        sentence, ["将", "从"], left, trigger_start
    )
    governed_index = sentence.rfind("管辖的", subject_start, trigger_start)
    if governed_index >= 0:
        return governed_index + len("管辖的")
    possessive_index = sentence.find("的", subject_start, trigger_start)
    if possessive_index >= 0:
        return possessive_index + 1
    return subject_start


def rules_transfer_subject_end_for_huaru(
    sentence: str, trigger_end: int, right: int
) -> int:
    """定位“划入的...”对象列表的右边界。"""
    candidates = [right]
    for marker in ("以及从", "并从", "和从", "、从"):
        index = sentence.find(marker, trigger_end, right)
        if index >= 0:
            candidates.append(index)
    return min(candidates)


def rules_previous_clause_bounds(sentence: str, left: int) -> tuple[int, int]:
    if left <= 0:
        return (0, 0)
    prev_right = max(0, left - 1)
    prev_left = 0
    for index in range(prev_right - 1, -1, -1):
        if sentence[index] in rules_CLAUSE_PUNCTUATION:
            prev_left = index + 1
            break
    return (prev_left, prev_right)


def rules_next_clause_bounds(sentence: str, right: int) -> tuple[int, int]:
    if right >= len(sentence):
        return (len(sentence), len(sentence))
    next_left = right + 1
    next_right = len(sentence)
    for index in range(next_left, len(sentence)):
        if sentence[index] in rules_CLAUSE_PUNCTUATION:
            next_right = index
            break
    return (next_left, next_right)


def rules_first_trigger_after(
    sentence: str, start: int, end: int, trigger_pattern: str
) -> int:
    match = re.search(trigger_pattern, sentence[start:end])
    if match is None:
        return end
    return start + match.start()


class rules_RelationCollector:

    def __init__(self, relation_schema: dict[str, dict[str, str]]):
        self.relation_schema = relation_schema
        self.rows: list[dict[str, Any]] = []
        self.seen: set[tuple[str, str, str, str, str, str]] = set()

    def add(
        self,
        relation_type_id: str,
        sentence_row: pd.Series,
        subject: rules_Mention | None,
        object_mention: rules_Mention | None,
        trigger_text: str,
        trigger_start: int,
        trigger_end: int,
        extraction_rule: str,
        confidence: float,
        evidence_text: str,
        object_synthetic: dict[str, str] | None = None,
        qualifier: str = "",
    ) -> None:
        if subject is None:
            return
        meta = self.relation_schema.get(
            relation_type_id,
            {
                "relation_type_name": relation_type_id,
                "relation_group": "",
                "re_status": "",
            },
        )
        subject_columns = rules_mention_to_columns(subject, "subject")
        if object_synthetic is not None:
            object_columns = rules_synthetic_object_columns(
                object_id=object_synthetic["object_entity_id"],
                object_text=object_synthetic["object_text"],
                object_type=object_synthetic["object_type"],
            )
        else:
            if object_mention is None:
                return
            object_columns = rules_mention_to_columns(object_mention, "object")
        key = (
            rules_clean_cell(sentence_row["sentence_id"]),
            relation_type_id,
            str(subject_columns["subject_entity_id"]),
            str(subject_columns["subject_start_pos"]),
            str(object_columns["object_entity_id"]),
            str(object_columns["object_text"]),
        )
        if key in self.seen:
            return
        self.seen.add(key)
        row = {
            "triple_id": f"T{len(self.rows) + 1:06d}",
            "sentence_id": rules_clean_cell(sentence_row["sentence_id"]),
            "source_file": rules_clean_cell(sentence_row["source_file"]),
            "year": sentence_row["year"],
            "line_no": sentence_row["line_no"],
            "item_no": rules_clean_cell(sentence_row["item_no"]),
            "sentence": rules_clean_cell(sentence_row["sentence"]),
            "relation_type_id": relation_type_id,
            "relation_type_name": meta["relation_type_name"],
            "relation_group": meta["relation_group"],
            "re_status": meta["re_status"],
            "predicate": meta["relation_type_name"],
            **subject_columns,
            **object_columns,
            "trigger_text": trigger_text,
            "trigger_start_pos": trigger_start,
            "trigger_end_pos": trigger_end,
            "extraction_rule": extraction_rule,
            "confidence": round(float(confidence), 4),
            "qualifier": qualifier,
            "evidence_text": evidence_text,
        }
        self.rows.append(row)

    def event_object(
        self, sentence_row: pd.Series, relation_type_id: str
    ) -> dict[str, str]:
        meta = self.relation_schema.get(
            relation_type_id, {"relation_type_name": relation_type_id}
        )
        sentence_id = rules_clean_cell(sentence_row["sentence_id"])
        year = rules_clean_cell(sentence_row["year"])
        return {
            "object_entity_id": f"EV_{sentence_id}_{relation_type_id}",
            "object_text": f"{year}年{meta['relation_type_name']}事件",
            "object_type": "CHANGE_EVENT",
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def rules_extract_revoke(
    sentence_row: pd.Series,
    mentions: list[rules_Mention],
    collector: rules_RelationCollector,
) -> None:
    sentence = rules_clean_cell(sentence_row["sentence"])
    for match in re.finditer("撤销", sentence):
        left, right = rules_clause_bounds(sentence, match.start())
        subjects = rules_prefer_specific_admins(
            rules_admin_mentions_between(mentions, match.end(), right), sentence
        )
        event = collector.event_object(sentence_row, "REVOKE_ADMIN")
        for subject in subjects:
            collector.add(
                "REVOKE_ADMIN",
                sentence_row,
                subject,
                None,
                "撤销",
                match.start(),
                match.end(),
                "revoke_clause_admins_after_trigger",
                0.91,
                sentence[left:right],
                object_synthetic=event,
            )


def rules_extract_establish(
    sentence_row: pd.Series,
    mentions: list[rules_Mention],
    collector: rules_RelationCollector,
) -> None:
    sentence = rules_clean_cell(sentence_row["sentence"])
    pattern = re.compile("恢复设立|新设立|新设|增设|设立|设区|组建")
    for match in pattern.finditer(sentence):
        left, right = rules_clause_bounds(sentence, match.start())
        subject = rules_first_specific_after(mentions, match.end(), right)
        if subject is None and sentence[match.end() : right].startswith("市辖区"):
            parent = rules_last_specific_before(mentions, left, match.start())
            if parent is not None:
                subject = rules_make_text_mention(
                    sentence_row,
                    f"{parent.text}市辖区",
                    parent.start,
                    min(right, match.end() + len("市辖区")),
                    "ESTABLISH_DISTRICT",
                    0.72,
                )
        if subject is None and match.group(0) == "设区":
            parent = rules_last_specific_before(mentions, left, match.start())
            if parent is not None:
                subject = rules_make_text_mention(
                    sentence_row,
                    f"{parent.text}市辖区",
                    parent.start,
                    match.end(),
                    "ESTABLISH_DISTRICT",
                    0.7,
                )
        if subject is None:
            continue
        event = collector.event_object(sentence_row, "ESTABLISH_ADMIN")
        confidence = 0.86 if match.group(0).startswith("新") else 0.9
        collector.add(
            "ESTABLISH_ADMIN",
            sentence_row,
            subject,
            None,
            match.group(0),
            match.start(),
            match.end(),
            "establish_first_admin_after_trigger",
            confidence,
            sentence[left:right],
            object_synthetic=event,
        )


def rules_extract_rename(
    sentence_row: pd.Series,
    mentions: list[rules_Mention],
    collector: rules_RelationCollector,
) -> None:
    sentence = rules_clean_cell(sentence_row["sentence"])
    for match in re.finditer("更名为|改名为|恢复为", sentence):
        left, right = rules_clause_bounds(sentence, match.start())
        subject = rules_last_specific_before(mentions, left, match.start())
        obj = rules_first_specific_after(mentions, match.end(), right)
        if subject is None or obj is None or subject.entity_id == obj.entity_id:
            continue
        collector.add(
            "RENAME_ADMIN",
            sentence_row,
            subject,
            obj,
            match.group(0),
            match.start(),
            match.end(),
            "rename_closest_admin_before_after",
            0.94,
            sentence[left:right],
        )


def rules_extract_merge(
    sentence_row: pd.Series,
    mentions: list[rules_Mention],
    collector: rules_RelationCollector,
) -> None:
    sentence = rules_clean_cell(sentence_row["sentence"])
    for match in re.finditer("合并", sentence):
        left, right = rules_clause_bounds(sentence, match.start())
        subjects = rules_prefer_specific_admins(
            rules_admin_mentions_between(mentions, left, match.start()), sentence
        )
        if not subjects:
            prev_left, prev_right = rules_previous_clause_bounds(sentence, left)
            subjects = rules_prefer_specific_admins(
                rules_admin_mentions_between(mentions, prev_left, prev_right), sentence
            )
        obj = rules_first_specific_after(mentions, match.end(), right)
        if obj is None:
            next_left, next_right = rules_next_clause_bounds(sentence, right)
            obj = rules_first_specific_after(mentions, next_left, next_right)
        if obj is None:
            continue
        for subject in subjects:
            if subject.entity_id == obj.entity_id:
                continue
            collector.add(
                "MERGE_ADMIN",
                sentence_row,
                subject,
                obj,
                "合并",
                match.start(),
                match.end(),
                "merge_admins_before_to_first_admin_after",
                0.86,
                sentence[left:right],
            )


def rules_extract_transfer(
    sentence_row: pd.Series,
    mentions: list[rules_Mention],
    collector: rules_RelationCollector,
) -> None:
    sentence = rules_clean_cell(sentence_row["sentence"])
    for match in re.finditer("划归|划入|划出|并入", sentence):
        trigger = match.group(0)
        left, right = rules_clause_bounds(sentence, match.start())
        broader_left = rules_last_marker_after_left(
            sentence, ["将", "从"], 0, match.start()
        )
        if broader_left < match.start():
            left = min(left, broader_left)
        if trigger == "划入" and sentence[match.end() : match.end() + 2].startswith(
            "的"
        ):
            subject_start = match.end() + 1
            subject_end = rules_transfer_subject_end_for_huaru(
                sentence, match.end(), right
            )
            subjects = rules_prefer_specific_admins(
                rules_admin_mentions_between(mentions, subject_start, subject_end),
                sentence,
            )
            governing_trigger = sentence.rfind("辖", left, match.start())
            target = (
                rules_last_specific_before(mentions, left, governing_trigger)
                if governing_trigger >= 0
                else None
            )
            qualifier = ""
            source = rules_last_specific_before(mentions, left, match.start())
            if source is not None:
                qualifier = f"source_area={source.text}"
        else:
            subject_start = rules_transfer_subject_start(sentence, left, match.start())
            subjects = rules_prefer_specific_admins(
                rules_admin_mentions_between(mentions, subject_start, match.start()),
                sentence,
            )
            if not subjects and subject_start > left:
                fallback_start = rules_last_marker_after_left(
                    sentence, ["将", "从"], left, match.start()
                )
                subjects = rules_prefer_specific_admins(
                    rules_admin_mentions_between(
                        mentions, fallback_start, match.start()
                    ),
                    sentence,
                )
            target = rules_first_specific_after(mentions, match.end(), right)
            qualifier = ""
        if not subjects:
            text_subject = rules_make_text_mention(
                sentence_row,
                sentence[subject_start : match.start()],
                subject_start,
                match.start(),
                "TRANSFER_SUBJECT",
                0.62,
            )
            subjects = [text_subject] if text_subject is not None else []
        if target is None:
            target_end = rules_first_trigger_after(
                sentence, match.end(), right, "管辖|代管|直辖|。|；|;"
            )
            target = rules_make_text_mention(
                sentence_row,
                sentence[match.end() : target_end],
                match.end(),
                target_end,
                "TRANSFER_TARGET",
                0.66,
            )
        if target is None:
            continue
        for subject in subjects:
            if subject.entity_id == target.entity_id:
                continue
            collector.add(
                "TRANSFER_ADMIN",
                sentence_row,
                subject,
                target,
                trigger,
                match.start(),
                match.end(),
                "transfer_subjects_before_trigger_to_target_after",
                0.88,
                sentence[left:right],
                qualifier=qualifier,
            )


def rules_extract_jurisdiction(
    sentence_row: pd.Series,
    mentions: list[rules_Mention],
    collector: rules_RelationCollector,
) -> None:
    sentence = rules_clean_cell(sentence_row["sentence"])
    for match in re.finditer("由", sentence):
        trigger_index = sentence.find("管辖", match.end())
        if trigger_index < 0:
            continue
        left, right = rules_clause_bounds(sentence, match.start())
        if trigger_index > right:
            continue
        governed = rules_best_admin_or_text(
            sentence_row,
            sentence,
            mentions,
            left,
            match.start(),
            "JURISDICTION_GOVERNED",
            prefer="last",
        )
        controller = rules_best_admin_or_text(
            sentence_row,
            sentence,
            mentions,
            match.end(),
            trigger_index,
            "JURISDICTION_CONTROLLER",
            prefer="last",
        )
        if (
            controller is None
            or governed is None
            or controller.entity_id == governed.entity_id
        ):
            continue
        collector.add(
            "JURISDICTION_ADMIN",
            sentence_row,
            controller,
            governed,
            "管辖",
            trigger_index,
            trigger_index + len("管辖"),
            "jurisdiction_you_controller_trigger",
            (
                0.9
                if controller.entity_type == "ADMIN_AREA"
                and governed.entity_type == "ADMIN_AREA"
                else 0.72
            ),
            sentence[left:right],
        )
    for match in re.finditer("辖原|辖", sentence):
        if (
            match.start() > 0
            and sentence[match.start() - 1 : match.start() + 1] == "管辖"
        ):
            continue
        left, right = rules_clause_bounds(sentence, match.start())
        subject = rules_last_specific_before(mentions, left, match.start())
        object_start = match.end()
        tail = sentence[object_start:right]
        if (match.group(0) == "辖原" or tail.startswith("原")) and "的" in tail:
            object_start += tail.find("的") + 1
        objects = rules_prefer_specific_admins(
            rules_admin_mentions_between(mentions, object_start, right), sentence
        )
        if not objects:
            text_object = rules_make_text_mention(
                sentence_row,
                sentence[object_start:right],
                object_start,
                right,
                "JURISDICTION_OBJECT",
                0.6,
            )
            objects = [text_object] if text_object is not None else []
        if subject is None:
            continue
        for obj in objects:
            if obj.entity_id == subject.entity_id:
                continue
            collector.add(
                "JURISDICTION_ADMIN",
                sentence_row,
                subject,
                obj,
                match.group(0),
                match.start(),
                match.end(),
                "jurisdiction_subject_before_xia_objects_after",
                0.82,
                sentence[left:right],
            )
    for match in re.finditer("管辖的", sentence):
        left, right = rules_clause_bounds(sentence, match.start())
        subject = rules_last_specific_before(mentions, left, match.start())
        object_end = rules_first_trigger_after(
            sentence, match.end(), right, "划归|划入|划出|并入|设立|更名"
        )
        objects = rules_prefer_specific_admins(
            rules_admin_mentions_between(mentions, match.end(), object_end), sentence
        )
        if subject is None:
            continue
        for obj in objects:
            if obj.entity_id == subject.entity_id:
                continue
            collector.add(
                "JURISDICTION_ADMIN",
                sentence_row,
                subject,
                obj,
                "管辖",
                match.start(),
                match.start() + 2,
                "jurisdiction_guanxia_de",
                0.8,
                sentence[left:right],
            )


def rules_extract_direct_or_entrust(
    sentence_row: pd.Series,
    mentions: list[rules_Mention],
    collector: rules_RelationCollector,
    trigger: str,
    relation_type_id: str,
) -> None:
    sentence = rules_clean_cell(sentence_row["sentence"])
    for match in re.finditer("由", sentence):
        trigger_index = sentence.find(trigger, match.end())
        if trigger_index < 0:
            continue
        left, right = rules_clause_bounds(sentence, match.start())
        if trigger_index > right:
            continue
        subject_left = sentence.rfind("县级", left, match.start())
        if subject_left >= 0:
            subject_left += len("县级")
        else:
            subject_left = left
        subjects = rules_prefer_specific_admins(
            rules_admin_mentions_between(mentions, subject_left, match.start())
        )
        if not subjects:
            text_subject = rules_make_text_mention(
                sentence_row,
                sentence[subject_left : match.start()],
                subject_left,
                match.start(),
                f"{relation_type_id}_SUBJECT",
                0.68,
            )
            subjects = [text_subject] if text_subject is not None else []
        obj = rules_first_specific_after(mentions, match.end(), trigger_index)
        synthetic_context = None
        if obj is None and sentence[match.end() : trigger_index].strip() in {
            "省",
            "自治区",
        }:
            synthetic_context = {
                "object_entity_id": f"CTX_{rules_clean_cell(sentence_row['sentence_id'])}_{relation_type_id}_{trigger_index}",
                "object_text": "省级行政区",
                "object_type": "ADMIN_AREA_CONTEXT",
            }
        for subject in subjects:
            collector.add(
                relation_type_id,
                sentence_row,
                subject,
                obj,
                trigger,
                trigger_index,
                trigger_index + len(trigger),
                f"{relation_type_id.lower()}_you_trigger",
                0.9 if obj is not None else 0.76,
                sentence[left:right],
                object_synthetic=synthetic_context,
            )


def rules_extract_entrust_after_direct(
    sentence_row: pd.Series,
    mentions: list[rules_Mention],
    collector: rules_RelationCollector,
) -> None:
    """抽取“X由省直辖，Y市代管”中的委托代管关系。"""
    sentence = rules_clean_cell(sentence_row["sentence"])
    for match in re.finditer("代管", sentence):
        left, right = rules_clause_bounds(sentence, match.start())
        subject = None
        previous_you = sentence.rfind("由", 0, left)
        if previous_you >= 0:
            previous_left, _ = rules_clause_bounds(sentence, previous_you)
            subject = rules_last_specific_before(mentions, previous_left, previous_you)
        obj = rules_last_specific_before(mentions, left, match.start())
        if subject is None or obj is None or subject.entity_id == obj.entity_id:
            continue
        collector.add(
            "ENTRUST_ADMIN",
            sentence_row,
            subject,
            obj,
            "代管",
            match.start(),
            match.end(),
            "entrust_after_direct_clause",
            0.86,
            sentence[left:right],
        )


def rules_extract_residence(
    sentence_row: pd.Series,
    mentions: list[rules_Mention],
    collector: rules_RelationCollector,
) -> None:
    sentence = rules_clean_cell(sentence_row["sentence"])
    for match in re.finditer("驻", sentence):
        left, right = rules_clause_bounds(sentence, match.start())
        subject = rules_closest_before(mentions, match.start(), {"GOV_ORG"})
        if subject is None or subject.start < left:
            subject = rules_last_specific_before(mentions, left, match.start())
        obj = rules_first_after(mentions, match.end(), {"ADDRESS", "ADMIN_AREA"})
        object_synthetic = None
        if obj is None or obj.start > right:
            residence_text = rules_clean_residence_text(sentence[match.end() : right])
            if len(residence_text) < 2:
                continue
            object_synthetic = {
                "object_entity_id": f"TXT_{rules_clean_cell(sentence_row['sentence_id'])}_RES_{match.start()}",
                "object_text": residence_text,
                "object_type": "LOCATION_TEXT",
            }
            obj = None
        collector.add(
            "GOV_RESIDENCE",
            sentence_row,
            subject,
            obj,
            "驻",
            match.start(),
            match.end(),
            "residence_subject_before_zhu_object_after",
            0.88 if object_synthetic is None else 0.72,
            sentence[left:right],
            object_synthetic=object_synthetic,
        )


def rules_extract_residence_transfer(
    sentence_row: pd.Series,
    mentions: list[rules_Mention],
    collector: rules_RelationCollector,
) -> None:
    sentence = rules_clean_cell(sentence_row["sentence"])
    for match in re.finditer("迁移至|迁至|迁移", sentence):
        left, right = rules_clause_bounds(sentence, match.start())
        subject = rules_closest_before(mentions, match.start(), {"GOV_ORG"})
        if subject is None or subject.start < left:
            government_index = sentence.rfind("人民政府", left, match.start())
            subject = (
                rules_last_specific_before(mentions, left, government_index)
                if government_index >= 0
                else None
            )
        obj = rules_first_after(mentions, match.end(), {"ADDRESS", "ADMIN_AREA"})
        if subject is None or obj is None or obj.start > right:
            continue
        old_location = ""
        you_index = sentence.rfind("由", left, match.start())
        if you_index >= 0:
            old_mentions = rules_residence_mentions_between(
                mentions, you_index + 1, match.start()
            )
            if old_mentions:
                old_location = "old_residence=" + " / ".join(
                    (mention.text for mention in old_mentions)
                )
        collector.add(
            "RESIDENCE_TRANSFER",
            sentence_row,
            subject,
            obj,
            match.group(0),
            match.start(),
            match.end(),
            "residence_transfer_old_to_new",
            0.92,
            sentence[left:right],
            qualifier=old_location,
        )


def rules_extract_area_inheritance(
    sentence_row: pd.Series,
    mentions: list[rules_Mention],
    collector: rules_RelationCollector,
) -> None:
    sentence = rules_clean_cell(sentence_row["sentence"])
    pattern = re.compile(
        "以(?P<old>[^，,；;。]*?)(?:的)?(?:行政区域|管辖区域)为(?P<new>[^，,；;。]*?)(?:的)?行政区域"
    )
    for match in pattern.finditer(sentence):
        old_admins = rules_prefer_specific_admins(
            rules_admin_mentions_between(mentions, match.start("old"), match.end("old"))
        )
        new_admins = rules_prefer_specific_admins(
            rules_admin_mentions_between(mentions, match.start("new"), match.end("new"))
        )
        if not old_admins:
            old_text = rules_make_text_mention(
                sentence_row,
                sentence[match.start("old") : match.end("old")],
                match.start("old"),
                match.end("old"),
                "AREA_INHERIT_OLD",
                0.64,
            )
            old_admins = [old_text] if old_text is not None else []
        if not new_admins:
            new_text = rules_make_text_mention(
                sentence_row,
                sentence[match.start("new") : match.end("new")],
                match.start("new"),
                match.end("new"),
                "AREA_INHERIT_NEW",
                0.64,
            )
            new_admins = [new_text] if new_text is not None else []
        if not old_admins or not new_admins:
            continue
        for new_admin in new_admins:
            for old_admin in old_admins:
                if (
                    new_admin.entity_id == old_admin.entity_id
                    or new_admin.text == old_admin.text
                ):
                    continue
                collector.add(
                    "AREA_INHERITANCE",
                    sentence_row,
                    new_admin,
                    old_admin,
                    "以...行政区域为...行政区域",
                    match.start(),
                    match.end(),
                    "area_inheritance_regex",
                    0.9,
                    match.group(0),
                )


def rules_extract_adjustment_event(
    sentence_row: pd.Series,
    mentions: list[rules_Mention],
    collector: rules_RelationCollector,
) -> None:
    sentence = rules_clean_cell(sentence_row["sentence"])
    for match in re.finditer("行政区划调整|作如下调整|调整后|调整|变更", sentence):
        left, right = rules_clause_bounds(sentence, match.start())
        subjects = rules_prefer_specific_admins(
            rules_admin_mentions_between(mentions, left, right), sentence
        )
        if not subjects:
            continue
        event = collector.event_object(sentence_row, "ADJUSTMENT_EVENT")
        for subject in subjects:
            collector.add(
                "ADJUSTMENT_EVENT",
                sentence_row,
                subject,
                None,
                match.group(0),
                match.start(),
                match.end(),
                "adjustment_event_admins_in_clause",
                0.78,
                sentence[left:right],
                object_synthetic=event,
            )


def rules_extract_scope_constraint(
    sentence_row: pd.Series,
    mentions: list[rules_Mention],
    collector: rules_RelationCollector,
) -> None:
    sentence = rules_clean_cell(sentence_row["sentence"])
    for match in re.finditer("不含|不包括|包括|除外", sentence):
        left, right = rules_clause_bounds(sentence, match.start())
        subject = rules_last_specific_before(mentions, left, match.start())
        objects = rules_prefer_specific_admins(
            rules_admin_mentions_between(mentions, match.end(), right), sentence
        )
        if subject is None:
            subject = rules_first_specific_after(mentions, left, right)
        if subject is None:
            continue
        for obj in objects:
            if obj.entity_id == subject.entity_id:
                continue
            collector.add(
                "SCOPE_CONSTRAINT",
                sentence_row,
                subject,
                obj,
                match.group(0),
                match.start(),
                match.end(),
                "scope_constraint_in_clause",
                0.7,
                sentence[left:right],
            )


def rules_extract_sentence_relations(
    sentence_row: pd.Series,
    mentions: list[rules_Mention],
    collector: rules_RelationCollector,
) -> None:
    rules_extract_area_inheritance(sentence_row, mentions, collector)
    rules_extract_revoke(sentence_row, mentions, collector)
    rules_extract_establish(sentence_row, mentions, collector)
    rules_extract_rename(sentence_row, mentions, collector)
    rules_extract_merge(sentence_row, mentions, collector)
    rules_extract_transfer(sentence_row, mentions, collector)
    rules_extract_direct_or_entrust(
        sentence_row, mentions, collector, "直辖", "DIRECT_ADMIN"
    )
    rules_extract_direct_or_entrust(
        sentence_row, mentions, collector, "代管", "ENTRUST_ADMIN"
    )
    rules_extract_entrust_after_direct(sentence_row, mentions, collector)
    rules_extract_residence_transfer(sentence_row, mentions, collector)
    rules_extract_residence(sentence_row, mentions, collector)
    rules_extract_jurisdiction(sentence_row, mentions, collector)
    rules_extract_adjustment_event(sentence_row, mentions, collector)
    rules_extract_scope_constraint(sentence_row, mentions, collector)


def rules_build_sentence_relation_summary(
    sentence_frame: pd.DataFrame, triples: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    grouped = triples.groupby("sentence_id") if not triples.empty else {}
    for _, sentence_row in sentence_frame.iterrows():
        sentence_id = rules_clean_cell(sentence_row["sentence_id"])
        if not triples.empty and sentence_id in grouped.groups:
            group = grouped.get_group(sentence_id)
            relation_types = " / ".join(
                sorted(group["relation_type_id"].astype(str).unique())
            )
            relation_names = " / ".join(
                sorted(group["relation_type_name"].astype(str).unique())
            )
            relation_count = len(group)
        else:
            relation_types = ""
            relation_names = ""
            relation_count = 0
        rows.append(
            {
                "sentence_id": sentence_id,
                "source_file": sentence_row["source_file"],
                "year": sentence_row["year"],
                "line_no": sentence_row["line_no"],
                "item_no": rules_clean_cell(sentence_row["item_no"]),
                "sentence": sentence_row["sentence"],
                "relation_count": relation_count,
                "relation_type_ids": relation_types,
                "relation_type_names": relation_names,
            }
        )
    return pd.DataFrame(rows)


def rules_build_relation_type_summary(triples: pd.DataFrame) -> pd.DataFrame:
    if triples.empty:
        return pd.DataFrame(
            columns=[
                "relation_type_id",
                "relation_type_name",
                "relation_group",
                "triple_count",
                "sentence_count",
                "subject_count",
                "object_count",
                "avg_confidence",
            ]
        )
    summary = (
        triples.groupby(
            ["relation_type_id", "relation_type_name", "relation_group"], dropna=False
        )
        .agg(
            triple_count=("triple_id", "size"),
            sentence_count=("sentence_id", "nunique"),
            subject_count=("subject_text", "nunique"),
            object_count=("object_text", "nunique"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
        .sort_values(["triple_count", "relation_type_id"], ascending=[False, True])
    )
    summary["avg_confidence"] = summary["avg_confidence"].round(4)
    return summary


def rules_build_rule_samples(triples: pd.DataFrame, per_type: int = 5) -> pd.DataFrame:
    if triples.empty:
        return pd.DataFrame()
    return (
        triples.sort_values(["relation_type_id", "confidence"], ascending=[True, False])
        .groupby("relation_type_id", group_keys=False)
        .head(per_type)
        .reset_index(drop=True)
    )


def rules_is_publication_notice_sentence(sentence: Any) -> bool:
    text = rules_clean_cell(sentence)
    if not text:
        return False
    compact = text.strip(" （）()")
    has_publication_date = (
        re.search("\\d{4}年\\d{1,2}月\\d{1,2}日(?:公告|公布)", compact) is not None
    )
    if not has_publication_date:
        return False
    return "人民政府" in compact or "民政部" in compact


def rules_build_events(triples: pd.DataFrame) -> pd.DataFrame:
    if triples.empty:
        return pd.DataFrame()
    event_triples = triples[triples["object_type"].eq("CHANGE_EVENT")].copy()
    if event_triples.empty:
        return pd.DataFrame()
    rows = []
    for event_id, group in event_triples.groupby("object_entity_id", sort=False):
        first = group.iloc[0]
        rows.append(
            {
                "event_id": event_id,
                "event_text": first["object_text"],
                "relation_type_id": first["relation_type_id"],
                "relation_type_name": first["relation_type_name"],
                "sentence_id": first["sentence_id"],
                "source_file": first["source_file"],
                "year": first["year"],
                "line_no": first["line_no"],
                "item_no": first["item_no"],
                "sentence": first["sentence"],
                "related_entity_count": group["subject_entity_id"].nunique(),
                "related_entities": " / ".join(
                    group["subject_text"].astype(str).drop_duplicates()
                ),
                "trigger_texts": " / ".join(
                    group["trigger_text"].astype(str).drop_duplicates()
                ),
            }
        )
    return pd.DataFrame(rows)


def rules_build_overview(
    sentence_frame: pd.DataFrame,
    entity_frame: pd.DataFrame,
    triples: pd.DataFrame,
    sentence_summary: pd.DataFrame,
    zero_relation_sentences: pd.DataFrame,
    excluded_publication_notices: pd.DataFrame,
) -> pd.DataFrame:
    relation_sentence_count = (
        int(sentence_summary["relation_count"].gt(0).sum())
        if not sentence_summary.empty
        else 0
    )
    raw_zero_relation_sentence_count = int(
        len(sentence_frame) - relation_sentence_count
    )
    return pd.DataFrame(
        [
            {
                "sentence_count": len(sentence_frame),
                "entity_count": len(entity_frame),
                "triple_count": len(triples),
                "relation_sentence_count": relation_sentence_count,
                "zero_relation_sentence_count": int(len(zero_relation_sentences)),
                "raw_zero_relation_sentence_count": raw_zero_relation_sentence_count,
                "excluded_publication_notice_sentence_count": int(
                    len(excluded_publication_notices)
                ),
                "relation_type_count": (
                    int(triples["relation_type_id"].nunique())
                    if not triples.empty
                    else 0
                ),
                "avg_triples_per_relation_sentence": (
                    round(len(triples) / relation_sentence_count, 4)
                    if relation_sentence_count
                    else 0
                ),
            }
        ]
    )


def rules_write_outputs(output_dir: Path, outputs: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")


records_DEFAULT_SENTENCE_CSV = Path(
    "data/processed/ner_rule_uie/source_sentence_records.csv"
)
records_DEFAULT_ENTITY_CSV = Path("data/processed/ner_rule_uie/entities.csv")
records_DEFAULT_TRIPLE_CSV = Path("data/processed/relation_details/triples.csv")
records_DEFAULT_ADMIN_XLSX = Path(
    "data/source/admin_codes/行政区划编码表_20260427.xlsx"
)
records_DEFAULT_OUTPUT_DIR = Path("data/processed/relation_extraction")
records_RELATION_TO_ACTION = {
    "RENAME_ADMIN": "更名",
    "MERGE_ADMIN": "合并",
    "REVOKE_ADMIN": "撤销",
    "ESTABLISH_ADMIN": "设立",
    "TRANSFER_ADMIN": "划归",
    "JURISDICTION_ADMIN": "隶属",
    "DIRECT_ADMIN": "直辖",
    "ENTRUST_ADMIN": "代管",
    "RESIDENCE_TRANSFER": "驻地迁移",
    "GOV_RESIDENCE": "政府驻地",
    "AREA_INHERITANCE": "区域承继",
    "ADJUSTMENT_EVENT": "调整",
    "SCOPE_CONSTRAINT": "范围约束",
}
records_SOURCE_RELATION_TYPE_IDS = {
    "TEXT_UNIT_TRANSFER": ["TRANSFER_ADMIN"],
    "TEXT_UNIT_JURISDICTION": ["JURISDICTION_ADMIN"],
    "TEXT_RESIDENCE_TRANSFER": ["RESIDENCE_TRANSFER"],
    "TEXT_GOV_RESIDENCE": ["GOV_RESIDENCE"],
}
records_RELATION_TYPE_NAMES = {
    definition["relation_type_id"]: definition["relation_type_name"]
    for definition in schema_RELATION_DEFINITIONS
}
records_RELATION_LABEL_ORDER = [
    definition["relation_type_id"] for definition in schema_RELATION_DEFINITIONS
]
records_FALLBACK_RELATION_TYPE_ID = "ADJUSTMENT_EVENT"
records_UNIT_SUFFIXES = (
    "街道办事处",
    "村委会",
    "居委会",
    "街道",
    "社区",
    "苏木",
    "镇",
    "乡",
    "村",
)
records_UNIT_GROUP_RE = re.compile(
    "(?P<names>[\\u4e00-\\u9fa5、和及与]+?)(?:[0-9一二三四五六七八九十百]+个)?(?P<suffix>街道办事处|村委会|居委会|街道|社区|苏木|镇|乡|村)"
)
records_TRANSFER_TEXT_RE = re.compile(
    "将(?P<source>.+?)划归(?P<target>[\\u4e00-\\u9fa5]{2,30}(?:区|县|市|旗|盟|地区|自治州))管辖"
)
records_JURISDICTION_TEXT_RE = re.compile(
    "(?P<parent>[\\u4e00-\\u9fa5]{2,20}(?:区|县|市|旗|盟|地区|自治州))[:：]?辖(?P<children>[^。；;]+)"
)
records_RESIDENCE_TRANSFER_TEXT_RE = re.compile(
    "驻地由(?P<old>.+?)(?:迁移至|迁至|迁移)(?P<new>[^。；;]+)"
)
records_GOV_RESIDENCE_TEXT_RE = re.compile(
    "(?P<area>[\\u4e00-\\u9fa5]{2,20}(?:区|县|市|旗|盟|地区|自治州))人民政府驻(?!地)(?P<place>[^。；;]+)"
)


def records_clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def records_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"缺少输入文件：{path}")
    return pd.read_csv(path)


def records_unique_join(values: list[Any]) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = records_clean_cell(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return " / ".join(result)


def records_strip_parentheses(text: str) -> str:
    return re.sub("[（(].*?[）)]", "", text)


def records_clean_unit_name(text: str, suffix: str) -> str:
    value = records_clean_cell(text).strip(" ，。；;、:：")
    for marker in ("以及", "并", "和", "及", "与"):
        if value.startswith(marker) and len(value) > len(marker):
            value = value[len(marker) :]
    if "的" in value:
        value = value.rsplit("的", 1)[-1]
    if "驻地" in value:
        value = value.rsplit("驻地", 1)[-1]
    if not value:
        return ""
    if value.endswith(records_UNIT_SUFFIXES):
        return value
    return value + suffix


def records_split_unit_names(name_text: str, suffix: str) -> list[str]:
    text = records_clean_cell(name_text)
    text = re.sub("[0-9一二三四五六七八九十百]+个$", "", text)
    parts = re.split("[、,，]|以及|和|及|与", text)
    names: list[str] = []
    for part in parts:
        name = records_clean_unit_name(part, suffix)
        if len(name) >= 2 and (
            not any((bad in name for bad in ("公路", "国道", "铁路", "界线", "区域")))
        ):
            names.append(name)
    return names


def records_extract_unit_names(text: str) -> list[str]:
    cleaned = records_strip_parentheses(records_clean_cell(text))
    names: list[str] = []
    for match in records_UNIT_GROUP_RE.finditer(cleaned):
        names.extend(
            records_split_unit_names(match.group("names"), match.group("suffix"))
        )
    return list(dict.fromkeys(names))


def records_compact_location(text: str) -> str:
    value = records_strip_parentheses(records_clean_cell(text)).strip(" ，。；;、")
    units = records_extract_unit_names(value)
    if units:
        return units[-1]
    return value


def records_supplemental_record_relations(record_text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for match in records_TRANSFER_TEXT_RE.finditer(record_text):
        target = records_clean_cell(match.group("target"))
        for source in records_extract_unit_names(match.group("source")):
            rows.append(
                {
                    "subject": source,
                    "predicate": "划归",
                    "object": target,
                    "source_relation_type": "TEXT_UNIT_TRANSFER",
                    "evidence_text": match.group(0),
                }
            )
    for match in records_JURISDICTION_TEXT_RE.finditer(record_text):
        parent = records_clean_cell(match.group("parent"))
        for child in records_extract_unit_names(match.group("children")):
            rows.append(
                {
                    "subject": child,
                    "predicate": "隶属于",
                    "object": parent,
                    "source_relation_type": "TEXT_UNIT_JURISDICTION",
                    "evidence_text": match.group(0),
                }
            )
    for match in records_RESIDENCE_TRANSFER_TEXT_RE.finditer(record_text):
        old_location = records_compact_location(match.group("old"))
        new_location = records_compact_location(match.group("new"))
        if old_location and new_location:
            rows.append(
                {
                    "subject": old_location,
                    "predicate": "驻地迁至",
                    "object": new_location,
                    "source_relation_type": "TEXT_RESIDENCE_TRANSFER",
                    "evidence_text": match.group(0),
                }
            )
    for match in records_GOV_RESIDENCE_TEXT_RE.finditer(record_text):
        area = records_clean_cell(match.group("area"))
        place = records_compact_location(match.group("place"))
        if (
            area
            and place
            and (not any((marker in place for marker in ("迁至", "迁移", "驻地由"))))
        ):
            rows.append(
                {
                    "subject": area,
                    "predicate": "政府驻地",
                    "object": place,
                    "source_relation_type": "TEXT_GOV_RESIDENCE",
                    "evidence_text": match.group(0),
                }
            )
    return rows


def records_is_publication_notice(sentence: Any) -> bool:
    text = records_clean_cell(sentence).strip(" （）()")
    if not text:
        return False
    has_date = re.search("\\d{4}年\\d{1,2}月\\d{1,2}日(?:公告|公布)", text) is not None
    return has_date and ("人民政府" in text or "民政部" in text)


def records_make_record_key(row: pd.Series) -> str:
    item_no = records_clean_cell(row.get("item_no", ""))
    if item_no:
        return f"{records_clean_cell(row['source_file'])}::{item_no}"
    return f"{records_clean_cell(row['source_file'])}::{records_clean_cell(row['sentence_id'])}"


def records_relation_types_for_group(group: pd.DataFrame) -> set[str]:
    if group.empty:
        return set()
    return {
        records_clean_cell(value)
        for value in group["relation_type_id"].tolist()
        if records_clean_cell(value)
    }


def records_sorted_relation_types(relation_types: set[str]) -> list[str]:
    known = [item for item in relation_types if item in records_RELATION_TYPE_NAMES]
    if not known:
        return [records_FALLBACK_RELATION_TYPE_ID]
    order_index = {
        relation_type_id: index
        for index, relation_type_id in enumerate(records_RELATION_LABEL_ORDER)
    }
    return sorted(set(known), key=lambda item: order_index[item])


def records_relation_type_names(relation_type_ids: list[str]) -> list[str]:
    return [
        records_RELATION_TYPE_NAMES.get(relation_type_id, relation_type_id)
        for relation_type_id in relation_type_ids
    ]


def records_dynamic_relation_type_ids(
    source_relation: str, record_relation_type_ids: str
) -> list[str]:
    if source_relation in records_SOURCE_RELATION_TYPE_IDS:
        return records_SOURCE_RELATION_TYPE_IDS[source_relation]
    source_ids = [
        item.strip()
        for item in records_clean_cell(source_relation).split("+")
        if item.strip()
    ]
    known_source_ids = records_sorted_relation_types(set(source_ids))
    if known_source_ids:
        return known_source_ids
    fallback_ids = [
        item.strip()
        for item in records_clean_cell(record_relation_type_ids).split("/")
        if item.strip()
    ]
    known_fallback_ids = records_sorted_relation_types(set(fallback_ids))
    return known_fallback_ids or [records_FALLBACK_RELATION_TYPE_ID]


def records_relation_label_confidence(
    triple_group: pd.DataFrame, relation_type_ids: list[str]
) -> float:
    if (
        len(relation_type_ids) == 1
        and relation_type_ids[0] == records_FALLBACK_RELATION_TYPE_ID
        and triple_group.empty
    ):
        return 0.6
    if "confidence" not in triple_group.columns or triple_group.empty:
        return 0.9
    values = pd.to_numeric(triple_group["confidence"], errors="coerce").dropna()
    if values.empty:
        return 0.9
    return round(float(values.mean()), 4)


def records_admin_text(value: Any) -> str:
    text = records_clean_cell(value)
    if text.endswith("事件"):
        return ""
    if text.startswith("无旧区划") or text.startswith("无新区划"):
        return ""
    return text


def records_entities_from_triples(
    group: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    old_entities: list[str] = []
    new_entities: list[str] = []
    actions: list[str] = []
    for _, row in group.iterrows():
        relation_type = records_clean_cell(row["relation_type_id"])
        subject = records_admin_text(row.get("subject_text", ""))
        obj = records_admin_text(row.get("object_text", ""))
        actions.append(
            records_RELATION_TO_ACTION.get(
                relation_type, records_clean_cell(row.get("trigger_text", ""))
            )
        )
        if relation_type in {
            "REVOKE_ADMIN",
            "RENAME_ADMIN",
            "MERGE_ADMIN",
            "TRANSFER_ADMIN",
            "DIRECT_ADMIN",
            "ENTRUST_ADMIN",
        }:
            old_entities.append(subject)
        if relation_type == "AREA_INHERITANCE":
            old_entities.append(obj)
            new_entities.append(subject)
        elif relation_type in {
            "RENAME_ADMIN",
            "MERGE_ADMIN",
            "TRANSFER_ADMIN",
            "DIRECT_ADMIN",
            "ENTRUST_ADMIN",
        }:
            new_entities.append(obj)
        elif relation_type == "JURISDICTION_ADMIN":
            old_entities.append(obj)
            new_entities.append(subject)
        elif relation_type in {
            "ESTABLISH_ADMIN",
            "GOV_RESIDENCE",
            "RESIDENCE_TRANSFER",
        }:
            new_entities.append(subject if relation_type == "ESTABLISH_ADMIN" else obj)
    return (
        [x for x in old_entities if x],
        [x for x in new_entities if x],
        [x for x in actions if x],
    )


def records_regex_fallback_entities(
    text: str, old_entities: list[str], new_entities: list[str], actions: list[str]
) -> None:
    for match in re.finditer(
        "(?P<old>[\\u4e00-\\u9fa5]{2,20}[市县区旗])更名为(?P<new>[\\u4e00-\\u9fa5]{2,20}[市县区旗])",
        text,
    ):
        old_entities.append(match.group("old"))
        new_entities.append(match.group("new"))
        actions.append("更名")
    for match in re.finditer(
        "撤销(?P<old>[\\u4e00-\\u9fa5]{2,20}[市县区旗盟州地区])", text
    ):
        old_entities.append(match.group("old"))
        actions.append("撤销")
    for match in re.finditer(
        "设立(?:地级|县级|新的)?(?P<new>[\\u4e00-\\u9fa5]{2,20}[市县区旗盟州地区])",
        text,
    ):
        new_entities.append(match.group("new"))
        actions.append("设立")


def records_build_records(sentences: pd.DataFrame, triples: pd.DataFrame) -> pd.DataFrame:
    sentence_frame = sentences.copy()
    sentence_frame["record_key"] = sentence_frame.apply(records_make_record_key, axis=1)
    triple_frame = triples.copy()
    if not triple_frame.empty:
        triple_frame["record_key"] = triple_frame.apply(records_make_record_key, axis=1)
    triple_groups = {
        key: group.copy()
        for key, group in triple_frame.groupby("record_key", sort=False)
    }
    rows: list[dict[str, Any]] = []
    group_columns = ["source_file", "record_key", "year", "item_no"]
    grouped = sentence_frame.sort_values(
        ["source_file", "line_no", "sentence_id"]
    ).groupby(group_columns, dropna=False, sort=False)
    for index, (_, sentence_group) in enumerate(grouped, start=1):
        record_key = records_clean_cell(sentence_group.iloc[0]["record_key"])
        content_group = sentence_group[
            ~sentence_group["sentence"].map(records_is_publication_notice)
        ].copy()
        if content_group.empty:
            continue
        text = "".join(
            (records_clean_cell(value) for value in content_group["sentence"].tolist())
        )
        triple_group = triple_groups.get(record_key, pd.DataFrame())
        old_entities, new_entities, actions = records_entities_from_triples(
            triple_group
        )
        records_regex_fallback_entities(text, old_entities, new_entities, actions)
        for relation in records_supplemental_record_relations(text):
            old_entities.append(relation["subject"])
            new_entities.append(relation["object"])
            actions.append(relation["predicate"])
        relation_types = records_relation_types_for_group(triple_group)
        relation_type_ids = records_sorted_relation_types(relation_types)
        relation_type_names = records_relation_type_names(relation_type_ids)
        if (
            len(relation_type_ids) == 1
            and relation_type_ids[0] == records_FALLBACK_RELATION_TYPE_ID
            and not actions
        ):
            actions.append("调整")
        type_id = records_unique_join(relation_type_ids)
        type_label = records_unique_join(relation_type_names)
        confidence = records_relation_label_confidence(triple_group, relation_type_ids)
        rule = (
            "无细粒度关系，归入综合事件"
            if not relation_types
            else "RE 关系类型集合"
        )
        rows.append(
            {
                "record_id": f"CR{len(rows) + 1:06d}",
                "record_key": record_key,
                "source_file": records_clean_cell(
                    sentence_group.iloc[0]["source_file"]
                ),
                "year": int(sentence_group.iloc[0]["year"]),
                "effective_time": f"{int(sentence_group.iloc[0]['year'])}年",
                "item_no": records_clean_cell(sentence_group.iloc[0]["item_no"]),
                "sentence_ids": records_unique_join(
                    content_group["sentence_id"].tolist()
                ),
                "record_text": text,
                "before_entities": records_unique_join(old_entities),
                "after_entities": records_unique_join(new_entities),
                "action_keywords": records_unique_join(actions),
                "type_id": type_id,
                "type_label": type_label,
                "classification_confidence": confidence,
                "classification_rule": rule,
                "relation_type_ids": records_unique_join(relation_type_ids),
            }
        )
    return pd.DataFrame(rows)


def records_dynamic_triples_from_relation_triples(
    records: pd.DataFrame, triples: pd.DataFrame
) -> pd.DataFrame:
    record_lookup = records.set_index("record_key").to_dict("index")
    triple_frame = triples.copy()
    if not triple_frame.empty:
        triple_frame["record_key"] = triple_frame.apply(records_make_record_key, axis=1)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()

    def add_row(
        record_key: str,
        subject: str,
        predicate: str,
        obj: str,
        source_relation: str,
        evidence: str,
    ) -> None:
        if not subject or not obj or subject == obj:
            return
        record = record_lookup.get(record_key)
        if record is None:
            return
        key = (record_key, subject, predicate, obj)
        if key in seen:
            return
        seen.add(key)
        relation_type_ids = records_dynamic_relation_type_ids(
            source_relation, record.get("relation_type_ids", "")
        )
        relation_type_labels = records_relation_type_names(relation_type_ids)
        rows.append(
            {
                "dynamic_triple_id": f"DT{len(rows) + 1:06d}",
                "record_id": record["record_id"],
                "record_key": record_key,
                "year": record["year"],
                "effective_time": record["effective_time"],
                "type_label": record["type_label"],
                "relation_type_ids": records_unique_join(relation_type_ids),
                "relation_type_labels": records_unique_join(relation_type_labels),
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "source_relation_type": source_relation,
                "evidence_text": evidence,
            }
        )

    for record_key, group in triple_frame.groupby("record_key", sort=False):
        relation_types = set(group["relation_type_id"].astype(str))
        revoked = (
            group[group["relation_type_id"].eq("REVOKE_ADMIN")]["subject_text"]
            .map(records_admin_text)
            .tolist()
        )
        established = (
            group[group["relation_type_id"].eq("ESTABLISH_ADMIN")]["subject_text"]
            .map(records_admin_text)
            .tolist()
        )
        if revoked and established:
            for old in revoked:
                for new in established:
                    add_row(
                        record_key,
                        old,
                        "变更为",
                        new,
                        "REVOKE_ADMIN+ESTABLISH_ADMIN",
                        records_unique_join(group["sentence"].tolist()),
                    )
        for _, row in group.iterrows():
            relation_type = records_clean_cell(row["relation_type_id"])
            subject = records_admin_text(row.get("subject_text", ""))
            obj = records_admin_text(row.get("object_text", ""))
            raw_obj = records_clean_cell(row.get("object_text", ""))
            evidence = records_clean_cell(
                row.get("evidence_text", row.get("sentence", ""))
            )
            if relation_type == "RENAME_ADMIN":
                add_row(record_key, subject, "变更为", obj, relation_type, evidence)
            elif relation_type == "AREA_INHERITANCE":
                add_row(record_key, obj, "变更为", subject, relation_type, evidence)
            elif relation_type == "MERGE_ADMIN":
                add_row(record_key, subject, "合并为", obj, relation_type, evidence)
            elif relation_type == "TRANSFER_ADMIN":
                add_row(record_key, subject, "划归", obj, relation_type, evidence)
            elif relation_type == "JURISDICTION_ADMIN":
                add_row(record_key, obj, "隶属于", subject, relation_type, evidence)
            elif relation_type == "DIRECT_ADMIN":
                add_row(record_key, subject, "直辖于", obj, relation_type, evidence)
            elif relation_type == "ENTRUST_ADMIN":
                add_row(record_key, subject, "代管于", obj, relation_type, evidence)
            elif relation_type == "RESIDENCE_TRANSFER":
                add_row(record_key, subject, "驻地迁至", obj, relation_type, evidence)
            elif relation_type == "GOV_RESIDENCE":
                if (
                    obj not in {"地", "地迁移"}
                    and (not any((marker in obj for marker in ("迁至", "迁移"))))
                    and (
                        not any(
                            (
                                marker in evidence
                                for marker in ("驻地由", "迁至", "迁移至")
                            )
                        )
                    )
                ):
                    add_row(
                        record_key, subject, "政府驻地", obj, relation_type, evidence
                    )
            elif relation_type == "ADJUSTMENT_EVENT":
                add_row(
                    record_key, subject, "发生调整", raw_obj, relation_type, evidence
                )
            elif relation_type == "SCOPE_CONSTRAINT":
                add_row(record_key, subject, "范围约束", obj, relation_type, evidence)
        if "REVOKE_ADMIN" in relation_types and "ESTABLISH_ADMIN" not in relation_types:
            for old in revoked:
                add_row(
                    record_key,
                    old,
                    "撤销",
                    "无新区划（撤销）",
                    "REVOKE_ADMIN",
                    records_unique_join(group["sentence"].tolist()),
                )
        if "ESTABLISH_ADMIN" in relation_types and "REVOKE_ADMIN" not in relation_types:
            for new in established:
                add_row(
                    record_key,
                    "无旧区划（新设）",
                    "设立",
                    new,
                    "ESTABLISH_ADMIN",
                    records_unique_join(group["sentence"].tolist()),
                )
    for _, record in records.iterrows():
        record_key = records_clean_cell(record["record_key"])
        for relation in records_supplemental_record_relations(
            records_clean_cell(record["record_text"])
        ):
            add_row(
                record_key,
                relation["subject"],
                relation["predicate"],
                relation["object"],
                relation["source_relation_type"],
                relation["evidence_text"],
            )
    return pd.DataFrame(rows)


def records_parse_admin_path(path_text: str) -> list[str]:
    return [
        part.strip()
        for part in records_clean_cell(path_text).split(",")
        if part.strip() and part.strip() != "中国"
    ]


def records_format_admin_code(value: Any, width: int) -> str:
    text = records_clean_cell(value)
    if not text:
        return ""
    if re.fullmatch("\\d+(?:\\.0)?", text):
        text = str(int(float(text)))
    return text.zfill(width)


def records_admin_level_from_code(code: str, sheet_name: str) -> str:
    if sheet_name == "乡镇街道":
        return "township"
    if code.endswith("0000"):
        return "province"
    if code.endswith("00"):
        return "prefecture"
    return "county"


def records_parent_code_for_admin(
    code: str, level: str, known_area_codes: set[str]
) -> str:
    if level == "province":
        return "CN"
    if level == "prefecture":
        return code[:2] + "0000"
    if level == "county":
        prefecture = code[:4] + "00"
        return prefecture if prefecture in known_area_codes else code[:2] + "0000"
    county = code[:6]
    if county in known_area_codes:
        return county
    prefecture = code[:4] + "00"
    return prefecture if prefecture in known_area_codes else code[:2] + "0000"


def records_build_static_admin_graph(
    admin_xlsx: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not admin_xlsx.exists():
        raise SystemExit(f"缺少行政区划编码表：{admin_xlsx}")
    area_frame = pd.read_excel(admin_xlsx, sheet_name="省市区")
    township_frame = pd.read_excel(admin_xlsx, sheet_name="乡镇街道")
    known_area_codes = {
        records_format_admin_code(value, 6) for value in area_frame["code"].tolist()
    }
    nodes = [
        {
            "node_id": "CN",
            "admin_code": "CN",
            "admin_name": "中国",
            "admin_level": "country",
            "province": "",
            "prefecture": "",
            "county": "",
            "full_path": "中国",
            "source_table": "system",
        }
    ]
    for sheet_name, frame, width in [
        ("省市区", area_frame, 6),
        ("乡镇街道", township_frame, 9),
    ]:
        for _, row in frame.iterrows():
            code = records_format_admin_code(row["code"], width)
            path_parts = records_parse_admin_path(row["name"])
            if not code or not path_parts:
                continue
            level = records_admin_level_from_code(code, sheet_name)
            nodes.append(
                {
                    "node_id": f"ADM_{code}",
                    "admin_code": code,
                    "admin_name": path_parts[-1],
                    "admin_level": level,
                    "province": path_parts[0] if len(path_parts) >= 1 else "",
                    "prefecture": path_parts[1] if len(path_parts) >= 2 else "",
                    "county": path_parts[2] if len(path_parts) >= 3 else "",
                    "full_path": "中国/" + "/".join(path_parts),
                    "source_table": sheet_name,
                }
            )
    node_frame = pd.DataFrame(nodes).drop_duplicates("node_id")
    name_lookup = node_frame.set_index("node_id")["admin_name"].to_dict()
    code_lookup = {
        records_clean_cell(row["admin_code"]): records_clean_cell(row["node_id"])
        for _, row in node_frame.iterrows()
    }
    relation_rows = []
    for _, row in node_frame.iterrows():
        if row["node_id"] == "CN":
            continue
        code = records_clean_cell(row["admin_code"])
        level = records_clean_cell(row["admin_level"])
        parent_code = records_parent_code_for_admin(
            code[:6] if level != "township" else code, level, known_area_codes
        )
        parent_node_id = (
            "CN" if parent_code == "CN" else code_lookup.get(parent_code, "CN")
        )
        relation_rows.append(
            {
                "static_triple_id": f"ST{len(relation_rows) + 1:06d}",
                "subject_node_id": row["node_id"],
                "subject_code": code,
                "subject_name": row["admin_name"],
                "predicate": "隶属于",
                "object_node_id": parent_node_id,
                "object_name": name_lookup.get(parent_node_id, "中国"),
                "source_table": row["source_table"],
                "version_date": "2026-04-27",
            }
        )
    return (node_frame, pd.DataFrame(relation_rows))


def records_build_record_entity_table(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in records.iterrows():
        rows.append(
            {
                "record_id": row["record_id"],
                "year": row["year"],
                "record_text": row["record_text"],
                "变更前实体": row["before_entities"],
                "变更后实体": row["after_entities"],
                "变更类型": row["type_label"],
                "变更动作关键词": row["action_keywords"],
                "置信度": row["classification_confidence"],
            }
        )
    return pd.DataFrame(rows)


def records_write_outputs(output_dir: Path, outputs: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行关系抽取全流程。")
    parser.add_argument("--stats-dir", type=Path, default=schema_DEFAULT_STATS_DIR)
    parser.add_argument(
        "--sentence-pattern-csv", type=Path, default=schema_DEFAULT_PATTERN_CSV
    )
    parser.add_argument(
        "--schema-output-dir", type=Path, default=schema_DEFAULT_OUTPUT_DIR
    )
    parser.add_argument("--entity-csv", type=Path, default=rules_DEFAULT_ENTITY_CSV)
    parser.add_argument("--sentence-csv", type=Path, default=rules_DEFAULT_SENTENCE_CSV)
    parser.add_argument(
        "--details-output-dir", type=Path, default=rules_DEFAULT_OUTPUT_DIR
    )
    parser.add_argument("--admin-xlsx", type=Path, default=records_DEFAULT_ADMIN_XLSX)
    parser.add_argument("--output-dir", type=Path, default=records_DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--debug-output",
        action="store_true",
        help="输出关系候选复核、句级关系诊断表和规则样例。",
    )
    return parser.parse_args()


def run_relation_schema(args: argparse.Namespace) -> None:
    stats_dir = args.stats_dir
    output_dir = args.schema_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    verb_frequency = schema_read_csv(stats_dir / "verb_frequency.csv")
    relation_candidates = schema_read_csv(stats_dir / "relation_type_candidates.csv")
    sentence_patterns = schema_read_csv(args.sentence_pattern_csv)
    schema_frame = schema_build_relation_schema(verb_frequency, sentence_patterns)
    triggers = schema_build_trigger_archive(verb_frequency)
    candidate_review = schema_build_candidate_review(relation_candidates)
    overview = schema_build_overview(schema_frame, candidate_review)
    schema_frame.to_csv(
        output_dir / "relation_type_archive.csv", index=False, encoding="utf-8-sig"
    )
    triggers.to_csv(
        output_dir / "relation_trigger_words.csv", index=False, encoding="utf-8-sig"
    )
    overview.to_csv(
        output_dir / "relation_schema_overview.csv", index=False, encoding="utf-8-sig"
    )
    if args.debug_output:
        candidate_review.to_csv(
            output_dir / "relation_candidate_review.csv",
            index=False,
            encoding="utf-8-sig",
        )
        schema_json = {
            "relations": schema_frame.to_dict(orient="records"),
            "triggers": triggers.to_dict(orient="records"),
            "overview": overview.iloc[0].to_dict(),
        }
        (output_dir / "relation_schema.json").write_text(
            json.dumps(schema_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        schema_write_markdown(output_dir, schema_frame, overview)
    print(f"关系类别归档输出目录：{output_dir}")
    print("\n归档总览：")
    print(overview.to_string(index=False))


def run_relation_details(args: argparse.Namespace) -> None:
    entity_frame = rules_read_required_csv(
        args.entity_csv, rules_REQUIRED_ENTITY_COLUMNS
    )
    sentence_frame = rules_read_required_csv(
        args.sentence_csv, rules_REQUIRED_SENTENCE_COLUMNS
    )
    relation_schema = rules_load_schema(
        args.schema_output_dir / "relation_type_archive.csv"
    )
    mentions_by_sentence = rules_build_mentions_by_sentence(entity_frame)
    collector = rules_RelationCollector(relation_schema)
    for _, sentence_row in sentence_frame.iterrows():
        sentence_id = rules_clean_cell(sentence_row["sentence_id"])
        mentions = mentions_by_sentence.get(sentence_id, [])
        rules_extract_sentence_relations(sentence_row, mentions, collector)
    triples = collector.to_frame()
    sentence_summary = rules_build_sentence_relation_summary(sentence_frame, triples)
    relation_type_summary = rules_build_relation_type_summary(triples)
    publication_notice_mask = sentence_summary["sentence"].map(
        rules_is_publication_notice_sentence
    )
    excluded_publication_notices = sentence_summary[
        sentence_summary["relation_count"].eq(0) & publication_notice_mask
    ].copy()
    zero_relation_sentences = sentence_summary[
        sentence_summary["relation_count"].eq(0) & ~publication_notice_mask
    ].copy()
    rule_samples = rules_build_rule_samples(triples)
    events = rules_build_events(triples)
    overview = rules_build_overview(
        sentence_frame,
        entity_frame,
        triples,
        sentence_summary,
        zero_relation_sentences,
        excluded_publication_notices,
    )
    outputs = {
        "triples.csv": triples,
        "relation_type_summary.csv": relation_type_summary,
        "re_overview.csv": overview,
    }
    if args.debug_output:
        outputs.update(
            {
                "relation_events.csv": events,
                "sentence_relation_summary.csv": sentence_summary,
                "zero_relation_sentences.csv": zero_relation_sentences,
                "excluded_publication_notices.csv": excluded_publication_notices,
                "relation_rule_samples.csv": rule_samples,
            }
        )
    rules_write_outputs(args.details_output_dir, outputs)
    print(f"RE 句内关系输出目录：{args.details_output_dir}")
    print("\nRE 总览：")
    print(overview.to_string(index=False))


def run_relation_records(args: argparse.Namespace) -> None:
    sentences = records_read_csv(args.sentence_csv)
    entities = records_read_csv(args.entity_csv)
    triples = records_read_csv(args.details_output_dir / "triples.csv")
    admin_nodes, static_triples = records_build_static_admin_graph(args.admin_xlsx)
    records_frame = records_build_records(sentences, triples)
    dynamic_triples = records_dynamic_triples_from_relation_triples(records_frame, triples)
    record_entity_table = records_build_record_entity_table(records_frame)
    relation_type_count = len(
        {
            item.strip()
            for value in records_frame.get("relation_type_ids", pd.Series()).tolist()
            for item in records_clean_cell(value).split("/")
            if item.strip()
        }
    )
    overview = pd.DataFrame(
        [
            {
                "record_count": len(records_frame),
                "ner_entity_count": len(entities),
                "dynamic_triple_count": len(dynamic_triples),
                "static_admin_node_count": len(admin_nodes),
                "static_affiliation_triple_count": len(static_triples),
                "type_count": relation_type_count,
            }
        ]
    )
    outputs = {
        "records.csv": records_frame,
        "ner_entities_by_record.csv": record_entity_table,
        "dynamic_triples.csv": dynamic_triples,
        "static_admin_nodes.csv": admin_nodes,
        "static_affiliation_triples.csv": static_triples,
        "overview.csv": overview,
    }
    records_write_outputs(args.output_dir, outputs)
    print(f"RE 主输出目录：{args.output_dir}")
    print(overview.to_string(index=False))
    print("\n关系标签组合统计：")
    print(records_frame["type_label"].value_counts().to_string())


def main() -> None:
    args = parse_args()
    run_relation_schema(args)
    run_relation_details(args)
    run_relation_records(args)


if __name__ == "__main__":
    main()
