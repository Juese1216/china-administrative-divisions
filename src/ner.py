"""使用正则、行政区划词表和可选 UIE 做 NER。

这个脚本是当前项目的 NER 主线。

本脚本不做机器学习训练，也不读取动词统计结果。动词统计只用于后续
判断“有哪些关系类型”，而 NER 只负责把句子里的实体抽出来。

默认安全运行示例：

    conda run -n nlpEnv python src/ner.py

默认不会跑全量 UIE，因为 Paddle UIE 在 Mac CPU 上可能占用很多核心。
如果只想验证 UIE 增强入口，可以先只跑前几条句子：

    conda run -n nlpEnv python src/ner.py --enable-uie --max-sentences 3
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# 这些环境变量必须尽量早设置，放在 pandas/numpy/paddle 导入之前。
# 它们能降低 CPU 线程数，但 Paddle UIE 推理后端仍可能额外开线程，
# 所以脚本默认不自动启用全量 UIE。
for thread_env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "FLAGS_cpu_math_library_num_threads",
):
    os.environ.setdefault(thread_env_name, "1")

import pandas as pd


DEFAULT_ADMIN_XLSX = Path("data/source/admin_codes/行政区划编码表_20260427.xlsx")
DEFAULT_SOURCE_DIR = Path("data/source/mca_changes")
DEFAULT_SENTENCE_CSV = Path("data/processed/paddle_nlp_tokens/sentence_records.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/ner_rule_uie")

# UIE 的 schema 可以理解成“让模型找什么”。这里故意只放实体，不放关系。
DEFAULT_UIE_SCHEMA = ["行政区划", "人民政府", "政府驻地地址"]

# 这几个词通常只是类别名或泛称，不应当单独作为行政区划实体输出。
ADMIN_STOP_NAMES = {
    "中国",
    "市辖区",
    "行政区",
    "行政区域",
    "辖区",
    "特别行政区",
    "维吾尔自治区",
    "壮族自治区",
    "回族自治区",
    "自治区",
    "自治州",
    "自治县",
    "自治旗",
    "地区",
    "盟",
    "新区",
    "开发区",
    "管理区",
    "矿区",
    "林区",
    "特区",
    "省",
    "市",
    "区",
    "县",
    "旗",
    "镇",
    "乡",
    "民族乡",
    "苏木",
    "街道",
}

# 正则会先抓到一段“像行政区划名”的文本，再清掉前面的句法噪声。
# 注意：这里不用“新”“原”“驻”这类单字做无条件前缀，因为它们也可能是
# 真实地名开头，例如新乡市、原平市、驻马店市。
SAFE_LEADING_PREFIXES = (
    "将其管辖的",
    "将管辖的",
    "将所辖",
    "其管辖的",
    "管辖的",
    "代管的",
    "所辖",
    "驻新设立的",
    "驻新设的",
    "新设立的",
    "新设的",
    "新组建的",
    "原县级",
    "原地级",
    "原省级",
    "县级",
    "地级",
    "省级",
    "更名为",
    "改设为",
    "改为",
    "迁移至",
    "迁至",
    "划归",
    "划入",
    "划出",
    "撤销",
    "设立",
    "增设",
    "调整",
    "原属",
    "驻地由",
    "不含",
    "不包括",
    "包括",
    "以及",
    "管辖原",
    "管辖",
    "辖原",
    "辖",
    "的",
    "由",
    "以",
)

# 有些单字前缀可以是句法词，也可以是地名的一部分。只有当完整字符串
# 不在词表中时，才尝试剥离它们。
CONDITIONAL_LEADING_PREFIXES = ("将", "原")

# 顿号、逗号之间常见“甲、乙和丙”这种并列结构。正则从“和”开始命中时，
# 如果完整词不在词表中，就尝试去掉“和/及/与”。
CONJUNCTION_LEADING_PREFIXES = ("和", "及", "与")

# 这类短语是正则按“若干汉字 + 行政后缀”误切出来的句法片段，
# 例如“县级玉门市、敦煌市由省直辖”里的“由省”。
RELATION_PREFIX_NOISE = (
    "撤销",
    "设立",
    "调整",
    "将",
    "以",
    "由",
    "其",
    "并入",
    "划归",
    "管辖",
    "代管",
    "更名",
    "批准",
)

ADMIN_SUFFIXES = (
    "特别行政区",
    "维吾尔自治区",
    "壮族自治区",
    "回族自治区",
    "自治区",
    "自治州",
    "自治县",
    "自治旗",
    "民族乡",
    "开发区",
    "管理区",
    "矿区",
    "林区",
    "新区",
    "特区",
    "地区",
    "街道",
    "苏木",
    "省",
    "市",
    "区",
    "县",
    "旗",
    "镇",
    "乡",
    "盟",
)

# 非贪婪匹配很重要：例如“撤销云南省畹町市”，应该先抓到“撤销云南省”，
# 清洗成“云南省”，然后继续抓“畹町市”，而不是一次吞成一个长实体。
ADMIN_SUFFIX_PATTERN = re.compile(
    r"[\u4e00-\u9fff]{1,30}?"
    r"(?:"
    + "|".join(re.escape(suffix) for suffix in ADMIN_SUFFIXES)
    + r")"
)

GOVERNMENT_ORG_PATTERN = re.compile(r"[\u4e00-\u9fff]{0,20}(?:人民政府|民政部|国务院)")

ADDRESS_PATTERN = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9]{1,35}"
    r"(?:大道|大街|街道|路|街|巷|弄)"
    r"[\u4e00-\u9fffA-Za-z0-9\-]*号?"
)

BAD_ADMIN_CONTEXT_SUBSTRINGS = (
    "人民政府",
    "行政区域",
    "部分区域",
    "所辖区域",
    "所辖区",
    "委员会",
    "办事处",
    "居委会",
    "村委会",
    "划归",
    "管辖",
    "迁至",
    "迁移",
    "驻地",
    "为界",
    "公路",
    "铁路",
    "国道",
    "所属",
)

CHANGE_CUE_PATTERNS = {
    "revoke": re.compile(r"撤销"),
    "establish": re.compile(r"设立|设置|增设"),
    "rename": re.compile(r"更名"),
    "transfer": re.compile(r"划归|划入|划出"),
    "govern": re.compile(r"管辖|辖|直辖|代管"),
    "residence": re.compile(r"人民政府驻|政府驻|驻地|迁至|迁移"),
    "adjustment": re.compile(r"行政区划调整|调整"),
}

UIE_SCHEMA_TO_ENTITY_TYPE = {
    "行政区划": "ADMIN_AREA",
    "行政区划名称": "ADMIN_AREA",
    "地名": "ADMIN_AREA",
    "人民政府": "GOV_ORG",
    "人民政府机构": "GOV_ORG",
    "政府机构": "GOV_ORG",
    "政府驻地": "ADDRESS",
    "政府驻地地址": "ADDRESS",
    "驻地地址": "ADDRESS",
    "地址": "ADDRESS",
}


@dataclass
class AdminRecord:
    """行政区划编码表中的一条记录。"""

    code: str
    name: str
    level: str
    full_path: str
    province: str
    prefecture: str
    county: str


@dataclass
class EntityCandidate:
    """一个候选实体。

    先叫“候选”，是因为同一句话里不同方法可能抽到互相重叠的结果。
    后面会做合并和重叠消解，最后才写入 entities.csv。
    """

    sentence_id: str
    source_file: str
    year: int | str
    line_no: int | str
    item_no: str
    sentence: str
    entity_text: str
    normalized_name: str
    entity_type: str
    method: str
    confidence: float
    start_pos: int
    end_pos: int
    admin_code: str = ""
    admin_level: str = ""
    province: str = ""
    prefecture: str = ""
    county: str = ""
    full_path: str = ""
    is_from_code_table: int = 0
    is_ambiguous: int = 0
    is_uie_enhanced: int = 0
    note: str = ""


@dataclass
class LexiconIndex:
    """行政区划词表索引。

    词表可能包含 60 万级名称，如果每个句子都重新计算长度和首字集合，
    全量运行会很慢。所以这里一次构建，后面反复复用。
    """

    lookup: dict[str, list[AdminRecord]]
    name_lengths: list[int]
    first_chars: set[str]


ENTITY_COLUMNS = [
    "entity_id",
    "sentence_id",
    "source_file",
    "year",
    "line_no",
    "item_no",
    "sentence",
    "entity_text",
    "normalized_name",
    "entity_type",
    "method",
    "confidence",
    "start_pos",
    "end_pos",
    "admin_code",
    "admin_level",
    "province",
    "prefecture",
    "county",
    "full_path",
    "is_from_code_table",
    "is_ambiguous",
    "is_uie_enhanced",
    "note",
]

UIE_RAW_COLUMNS = [
    "sentence_id",
    "schema_name",
    "entity_text",
    "start_pos",
    "end_pos",
    "probability",
    "entity_type",
    "sentence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NER with regex, lexicon matching, and Paddle UIE.")
    parser.add_argument("--admin-xlsx", type=Path, default=DEFAULT_ADMIN_XLSX)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--sentence-csv", type=Path, default=DEFAULT_SENTENCE_CSV)
    parser.add_argument(
        "--input-mode",
        choices=["source", "sentence-csv"],
        default="source",
        help="Default reads raw MCA txt files directly. Use sentence-csv only for compatibility.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-township",
        action="store_true",
        help="Only load province/city/county codes. Default also loads township/street codes.",
    )
    parser.add_argument(
        "--enable-uie",
        action="store_true",
        help="Enable Paddle UIE enhancement. Use small batches on Mac CPU; prefer GPU for full data.",
    )
    parser.add_argument(
        "--disable-uie",
        action="store_true",
        help="Deprecated compatibility flag. UIE is disabled by default unless --enable-uie is set.",
    )
    parser.add_argument(
        "--strict-uie",
        action="store_true",
        help="Abort if Paddle UIE cannot be loaded. Default keeps regex + lexicon results.",
    )
    parser.add_argument("--uie-schema", default=",".join(DEFAULT_UIE_SCHEMA))
    parser.add_argument("--uie-device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--uie-batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--limit-files", type=int, default=0, help="Use first N source txt files for quick tests.")
    parser.add_argument("--max-sentences", type=int, default=0, help="Use first N sentences for a quick test.")
    parser.add_argument("--preview-rows", type=int, default=20)
    parser.add_argument(
        "--debug-output",
        action="store_true",
        help="Write audit-oriented intermediate CSV files. Default keeps only report essentials.",
    )
    return parser.parse_args()


def configure_cpu_threads(cpu_threads: int) -> None:
    """限制 CPU 线程，避免 Mac 上第一次跑 Paddle 时占用过猛。"""

    if cpu_threads <= 0:
        return
    for thread_env_name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "FLAGS_cpu_math_library_num_threads",
    ):
        os.environ[thread_env_name] = str(cpu_threads)


def code_level(code: str) -> str:
    """根据行政区划代码判断层级。"""

    code = str(code).strip()
    if len(code) >= 9:
        return "township"
    if code.endswith("0000"):
        return "province"
    if code.endswith("00"):
        return "prefecture"
    return "county"


def parse_admin_record(code: str, raw_path: str) -> AdminRecord | None:
    """把编码表里的路径拆成实体名、省、市、区县等字段。"""

    parts = [part.strip() for part in str(raw_path).split(",") if part and part.strip() and part.strip() != "中国"]
    if not parts:
        return None

    name = parts[-1]
    if not is_valid_entity_text(name):
        return None

    level = code_level(code)
    province = parts[0] if parts else ""
    prefecture = ""
    county = ""

    if level == "province":
        province = name
    elif level == "prefecture":
        prefecture = name
    elif level == "county":
        county = name
        prefecture = parts[-2] if len(parts) >= 2 else province
    else:
        county = parts[-2] if len(parts) >= 2 else ""
        prefecture = parts[-3] if len(parts) >= 3 else province

    return AdminRecord(
        code=str(code).strip(),
        name=name,
        level=level,
        full_path="/".join(parts),
        province=province,
        prefecture=prefecture,
        county=county,
    )


def load_admin_lookup(admin_xlsx: Path, include_township: bool) -> dict[str, list[AdminRecord]]:
    """读取行政区划编码表，构造成 name -> records 的词表。

    同名地名可能对应多个代码，例如“北京市”既可能是省级，也可能是地级。
    所以这里的 value 是 list，而不是单条记录。
    """

    if not admin_xlsx.exists():
        raise SystemExit(f"行政区划编码表不存在：{admin_xlsx}")

    sheets = ["省市区"]
    if include_township:
        sheets.append("乡镇街道")

    lookup: dict[str, list[AdminRecord]] = {}
    for sheet in sheets:
        frame = pd.read_excel(admin_xlsx, sheet_name=sheet, dtype={"code": str, "name": str})
        for _, row in frame.iterrows():
            record = parse_admin_record(str(row["code"]), str(row["name"]))
            if record is None:
                continue
            lookup.setdefault(record.name, []).append(record)

    return lookup


def build_lexicon_index(admin_lookup: dict[str, list[AdminRecord]]) -> LexiconIndex:
    names = list(admin_lookup.keys())
    return LexiconIndex(
        lookup=admin_lookup,
        name_lengths=sorted({len(name) for name in names}, reverse=True),
        first_chars={name[0] for name in names if name},
    )


def clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", str(line).strip())


def is_title_line(line: str) -> bool:
    return "县级以上行政区划变更情况" in line


def is_sequence_note_line(line: str) -> bool:
    return line.startswith("（以行政区划变更公布时间为序")


def infer_year(path: Path) -> int | None:
    match = re.match(r"(\d{4})_", path.name)
    return int(match.group(1)) if match else None


def split_sentences(text: str) -> list[str]:
    text = clean_line(text)
    if not text:
        return []
    pieces = re.split(r"(?<=[。！？；;])\s*", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def source_files(source_dir: Path, limit_files: int) -> list[Path]:
    files = sorted(path for path in source_dir.glob("*.txt") if path.is_file())
    if limit_files > 0:
        files = files[:limit_files]
    if not files:
        raise SystemExit(f"原始 txt 目录中没有文件：{source_dir}")
    return files


def parse_source_sentences(source_dir: Path, limit_files: int, max_sentences: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """直接读取民政部年度 txt，并切成 NER 句子。

    这一步只做文本结构解析，不调用 Paddle，也不做实体抽取。
    """

    sentence_rows: list[dict[str, Any]] = []
    line_rows: list[dict[str, Any]] = []
    sentence_index = 0

    for path in source_files(source_dir, limit_files):
        year = infer_year(path)
        item_no = ""
        for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = clean_line(raw_line)
            if not line:
                line_rows.append(
                    {
                        "source_file": path.name,
                        "year": year,
                        "line_no": line_no,
                        "item_no": item_no,
                        "raw_line": raw_line,
                        "parsed_text": "",
                        "is_skipped": 1,
                        "skip_reason": "empty",
                        "sentence_count": 0,
                    }
                )
                continue

            skip_reason = ""
            parsed_text = line
            if is_title_line(line):
                skip_reason = "title"
            elif is_sequence_note_line(line):
                skip_reason = "sequence_note"
            else:
                item_match = re.match(r"^([一二三四五六七八九十百〇零两]+)、(.+)$", line)
                if item_match:
                    item_no = item_match.group(1)
                    parsed_text = item_match.group(2).strip()

            sentences = [] if skip_reason else split_sentences(parsed_text)
            line_rows.append(
                {
                    "source_file": path.name,
                    "year": year,
                    "line_no": line_no,
                    "item_no": item_no,
                    "raw_line": raw_line,
                    "parsed_text": parsed_text if not skip_reason else "",
                    "is_skipped": int(bool(skip_reason)),
                    "skip_reason": skip_reason,
                    "sentence_count": len(sentences),
                }
            )

            for sentence in sentences:
                sentence_index += 1
                sentence_rows.append(
                    {
                        "sentence_id": f"S{sentence_index:06d}",
                        "source_file": path.name,
                        "year": year,
                        "line_no": line_no,
                        "item_no": item_no,
                        "sentence": sentence,
                    }
                )
                if max_sentences > 0 and len(sentence_rows) >= max_sentences:
                    return pd.DataFrame(sentence_rows), pd.DataFrame(line_rows)

    return pd.DataFrame(sentence_rows), pd.DataFrame(line_rows)


def load_sentences(sentence_csv: Path, max_sentences: int) -> pd.DataFrame:
    """读取已经整理好的变更句子表。"""

    if not sentence_csv.exists():
        raise SystemExit(
            f"句子文件不存在：{sentence_csv}\n"
            "请先运行 src/paddle_nlp_tokenize.py 生成 sentence_records.csv。"
        )

    required = {"sentence_id", "source_file", "year", "line_no", "item_no", "sentence"}
    frame = pd.read_csv(sentence_csv)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"句子文件缺少字段：{', '.join(missing)}")

    if max_sentences > 0:
        frame = frame.head(max_sentences).copy()
    return frame


def load_sentence_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """按输入模式读取句子，并返回句子表、原始行表和输入说明。"""

    if args.input_mode == "source":
        sentence_frame, source_line_frame = parse_source_sentences(args.source_dir, args.limit_files, args.max_sentences)
        if sentence_frame.empty:
            raise SystemExit(f"未能从原始 txt 中解析出句子：{args.source_dir}")
        return sentence_frame, source_line_frame, f"source txt directory: {args.source_dir}"

    sentence_frame = load_sentences(args.sentence_csv, args.max_sentences)
    source_line_frame = pd.DataFrame()
    return sentence_frame, source_line_frame, f"sentence csv: {args.sentence_csv}"


def is_valid_entity_text(text: str) -> bool:
    """过滤明显不应作为实体的短词、泛称和说明性短语。"""

    text = str(text).strip(" ，。；、（）()")
    if len(text) < 2:
        return False
    if text.startswith(("个", "部分")):
        return False
    if text.startswith(("的", "和", "及", "与")) and text[1:] in ADMIN_STOP_NAMES:
        return False
    if re.search(r"[0-9一二三四五六七八九十百千万]+个(?:街道|镇|乡|村|区|县|旗|苏木)", text):
        return False
    if "部分区" in text or "部分区域" in text:
        return False
    if text in ADMIN_STOP_NAMES:
        return False
    if (text.endswith("行政区") and not text.endswith("特别行政区")) or text.endswith("行政区域"):
        return False
    if any(mark in text for mark in ("如下", "情况", "同意", "批复")):
        return False
    return True


def has_admin_suffix(text: str) -> bool:
    return any(str(text).endswith(suffix) for suffix in ADMIN_SUFFIXES)


def clean_entity_text(text: str) -> str:
    return str(text).strip(" \t\r\n，。；、：（）()")


def strip_leading_admin_noise(
    text: str,
    start_pos: int,
    admin_lookup: dict[str, list[AdminRecord]],
) -> tuple[str, int]:
    """去掉正则误吞进去的动词、介词或修饰词。

    例如：
    - “撤销云南省” -> “云南省”
    - “设立地级通辽市” -> “通辽市”
    - “原县级通辽市” -> “通辽市”

    如果完整文本本来就在行政区划词表里，就不剥离，避免误伤真实地名。
    """

    text = clean_entity_text(text)
    changed = True
    while changed and text not in admin_lookup:
        changed = False

        for prefix in sorted(SAFE_LEADING_PREFIXES, key=len, reverse=True):
            if text.startswith(prefix) and len(text) > len(prefix) + 1:
                text = text[len(prefix) :]
                start_pos += len(prefix)
                changed = True
                break
        if changed:
            text = clean_entity_text(text)
            continue

        for prefix in CONDITIONAL_LEADING_PREFIXES:
            if not text.startswith(prefix) or len(text) <= len(prefix) + 1:
                continue
            possible = text[len(prefix) :]
            if possible in admin_lookup or has_admin_suffix(possible):
                text = possible
                start_pos += len(prefix)
                changed = True
                break

        if changed:
            text = clean_entity_text(text)
            continue

        for prefix in CONJUNCTION_LEADING_PREFIXES:
            if not text.startswith(prefix) or text in admin_lookup or len(text) <= len(prefix) + 1:
                continue
            possible = text[len(prefix) :]
            if has_admin_suffix(possible) and is_valid_entity_text(possible):
                text = possible
                start_pos += len(prefix)
                changed = True
                break

    return clean_entity_text(text), start_pos


def sentence_meta(row: pd.Series) -> dict[str, Any]:
    """把句子的公共字段复制到实体记录里。"""

    item_no = "" if pd.isna(row["item_no"]) else str(row["item_no"])
    return {
        "sentence_id": str(row["sentence_id"]),
        "source_file": str(row["source_file"]),
        "year": row["year"],
        "line_no": row["line_no"],
        "item_no": item_no,
        "sentence": str(row["sentence"]),
    }


def admin_summary(records: list[AdminRecord]) -> dict[str, Any]:
    """把一个实体名对应的一个或多个行政区划记录压成输出字段。"""

    return {
        "admin_code": "|".join(record.code for record in records),
        "admin_level": "|".join(sorted({record.level for record in records})),
        "province": "|".join(sorted({record.province for record in records if record.province})),
        "prefecture": "|".join(sorted({record.prefecture for record in records if record.prefecture})),
        "county": "|".join(sorted({record.county for record in records if record.county})),
        "full_path": "|".join(record.full_path for record in records[:5]),
        "is_from_code_table": 1,
        "is_ambiguous": int(len(records) > 1),
        "note": f"code_table_matches={len(records)}" if len(records) > 1 else "",
    }


def add_admin_fields(candidate: EntityCandidate, admin_lookup: dict[str, list[AdminRecord]]) -> EntityCandidate:
    records = admin_lookup.get(candidate.normalized_name, [])
    if not records:
        return candidate

    summary = admin_summary(records)
    candidate.admin_code = str(summary["admin_code"])
    candidate.admin_level = str(summary["admin_level"])
    candidate.province = str(summary["province"])
    candidate.prefecture = str(summary["prefecture"])
    candidate.county = str(summary["county"])
    candidate.full_path = str(summary["full_path"])
    candidate.is_from_code_table = int(summary["is_from_code_table"])
    candidate.is_ambiguous = int(summary["is_ambiguous"])

    notes = [note for note in [candidate.note, str(summary["note"])] if note]
    candidate.note = "; ".join(notes)
    return candidate


def make_candidate(
    row: pd.Series,
    entity_text: str,
    entity_type: str,
    method: str,
    confidence: float,
    start_pos: int,
    end_pos: int,
    admin_lookup: dict[str, list[AdminRecord]],
    note: str = "",
) -> EntityCandidate | None:
    """创建候选实体，同时补充行政区划代码字段。"""

    entity_text = clean_entity_text(entity_text)
    if not is_valid_entity_text(entity_text):
        return None
    if entity_type == "ADMIN_AREA" and any(part in entity_text for part in BAD_ADMIN_CONTEXT_SUBSTRINGS):
        return None
    if (
        entity_type == "ADMIN_AREA"
        and entity_text not in admin_lookup
        and len(entity_text) <= 4
        and entity_text.startswith(RELATION_PREFIX_NOISE)
    ):
        return None
    if entity_type == "ADMIN_AREA" and entity_text not in admin_lookup and len(entity_text) == 2 and entity_text.endswith(("市", "区", "旗")):
        return None
    if start_pos < 0 or end_pos <= start_pos:
        return None

    candidate = EntityCandidate(
        **sentence_meta(row),
        entity_text=entity_text,
        normalized_name=entity_text,
        entity_type=entity_type,
        method=method,
        confidence=float(confidence),
        start_pos=int(start_pos),
        end_pos=int(end_pos),
        is_uie_enhanced=int("paddle_uie" in method),
        note=note,
    )
    return add_admin_fields(candidate, admin_lookup)


def is_bad_exact_context(sentence: str, start_pos: int, end_pos: int, name: str) -> bool:
    """过滤词表精确匹配里的少数伪命中。"""

    next_char = sentence[end_pos : end_pos + 1]
    if next_char == "级":
        return True
    if name == "和县" and next_char in {"级", "的"}:
        return True
    return False


def is_bad_regex_admin_context(sentence: str, start_pos: int, end_pos: int, name: str) -> bool:
    """过滤正则行政地名里的句法碎片。"""

    if is_bad_exact_context(sentence, start_pos, end_pos, name):
        return True
    if name.startswith(("东起", "西至", "南起", "北至")):
        return True
    next_char = sentence[end_pos : end_pos + 1]
    if next_char == "级" and name.startswith(("原", "县", "地")):
        return True
    return False


def extract_lexicon_entities(row: pd.Series, lexicon_index: LexiconIndex) -> list[EntityCandidate]:
    """用行政区划编码表做精确词表匹配。

    这是最高置信度来源，因为命中的实体可以直接回填行政区划代码。
    """

    sentence = str(row["sentence"])
    admin_lookup = lexicon_index.lookup
    candidates: list[EntityCandidate] = []

    for start_pos, char in enumerate(sentence):
        if char not in lexicon_index.first_chars:
            continue
        remaining = len(sentence) - start_pos
        for length in lexicon_index.name_lengths:
            if length > remaining:
                continue
            text = sentence[start_pos : start_pos + length]
            if text not in admin_lookup:
                continue
            end_pos = start_pos + length
            if is_bad_exact_context(sentence, start_pos, end_pos, text):
                continue
            records = admin_lookup[text]
            confidence = 0.97 if len(records) > 1 else 0.99
            candidate = make_candidate(
                row=row,
                entity_text=text,
                entity_type="ADMIN_AREA",
                method="lexicon_exact",
                confidence=confidence,
                start_pos=start_pos,
                end_pos=end_pos,
                admin_lookup=admin_lookup,
            )
            if candidate is not None:
                candidates.append(candidate)

    return candidates


def extract_regex_admin_entities(row: pd.Series, admin_lookup: dict[str, list[AdminRecord]]) -> list[EntityCandidate]:
    """用行政区划后缀正则补充历史地名或词表未覆盖名称。"""

    sentence = str(row["sentence"])
    candidates: list[EntityCandidate] = []

    for match in ADMIN_SUFFIX_PATTERN.finditer(sentence):
        text, start_pos = strip_leading_admin_noise(match.group(0), match.start(), admin_lookup)
        end_pos = start_pos + len(text)
        if not has_admin_suffix(text):
            continue
        if is_bad_regex_admin_context(sentence, start_pos, end_pos, text):
            continue
        if not is_valid_entity_text(text):
            continue

        confidence = 0.86 if text in admin_lookup else 0.72
        candidate = make_candidate(
            row=row,
            entity_text=text,
            entity_type="ADMIN_AREA",
            method="regex_admin_area",
            confidence=confidence,
            start_pos=start_pos,
            end_pos=end_pos,
            admin_lookup=admin_lookup,
        )
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def extract_regex_government_entities(
    row: pd.Series,
    admin_lookup: dict[str, list[AdminRecord]],
) -> list[EntityCandidate]:
    """用正则抽取“人民政府”等机构实体。"""

    sentence = str(row["sentence"])
    candidates: list[EntityCandidate] = []

    for match in GOVERNMENT_ORG_PATTERN.finditer(sentence):
        candidate = make_candidate(
            row=row,
            entity_text=match.group(0),
            entity_type="GOV_ORG",
            method="regex_government_org",
            confidence=0.78,
            start_pos=match.start(),
            end_pos=match.end(),
            admin_lookup=admin_lookup,
        )
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def is_address_context(sentence: str, start_pos: int, text: str) -> bool:
    """判断地址正则命中是否更像驻地地址，而不是普通街道名。"""

    if "街道办事处" in text:
        return False
    if re.search(r"[0-9一二三四五六七八九十百千万]+个(?:街道|镇|乡|村)", text):
        return False
    left = sentence[max(0, start_pos - 4) : start_pos]
    if any(key in left for key in ("驻", "迁至", "驻地", "治所")):
        return True
    return "号" in text


def strip_leading_address_noise(text: str, start_pos: int) -> tuple[str, int]:
    """去掉地址正则误吞进去的“人民政府驻”等前缀。"""

    text = clean_entity_text(text)
    markers = (
        "人民政府驻",
        "政府驻",
        "驻地迁至",
        "驻地由",
        "迁移至",
        "迁至",
        "驻",
    )
    for marker in sorted(markers, key=len, reverse=True):
        index = text.rfind(marker)
        if index < 0:
            continue
        cut_pos = index + len(marker)
        if cut_pos < len(text):
            text = text[cut_pos:]
            start_pos += cut_pos
            break
    return clean_entity_text(text), start_pos


def extract_regex_address_entities(
    row: pd.Series,
    admin_lookup: dict[str, list[AdminRecord]],
) -> list[EntityCandidate]:
    """用正则抽取政府驻地或道路门牌地址。"""

    sentence = str(row["sentence"])
    candidates: list[EntityCandidate] = []

    for match in ADDRESS_PATTERN.finditer(sentence):
        text, start_pos = strip_leading_address_noise(match.group(0), match.start())
        if not is_address_context(sentence, start_pos, text):
            continue
        candidate = make_candidate(
            row=row,
            entity_text=text,
            entity_type="ADDRESS",
            method="regex_address",
            confidence=0.68,
            start_pos=start_pos,
            end_pos=start_pos + len(text),
            admin_lookup=admin_lookup,
        )
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def load_uie(schema: list[str], args: argparse.Namespace) -> Any | None:
    """加载 PaddleNLP UIE 模型。

    第一次运行时，PaddleNLP 可能会下载模型，所以会比普通正则慢。
    """

    if not args.enable_uie or args.disable_uie:
        return None

    try:
        import paddle
        from paddlenlp import Taskflow

        paddle.set_device(args.uie_device)
        return Taskflow("information_extraction", schema=schema, batch_size=args.uie_batch_size)
    except Exception as exc:  # pragma: no cover - 这里主要处理本机环境问题
        message = f"Paddle UIE 加载失败：{type(exc).__name__}: {exc}"
        if args.strict_uie:
            raise SystemExit(message) from exc
        print(f"[warning] {message}")
        print("[warning] 本次会继续输出正则 + 词表结果，但 UIE 增强结果为空。")
        return None


def normalize_uie_result(result: Any) -> dict[str, list[dict[str, Any]]]:
    """兼容 UIE 返回的字典结构。"""

    if isinstance(result, dict):
        return {str(key): value for key, value in result.items() if isinstance(value, list)}
    return {}


def run_uie_entities(
    sentence_frame: pd.DataFrame,
    admin_lookup: dict[str, list[AdminRecord]],
    schema: list[str],
    args: argparse.Namespace,
) -> tuple[list[EntityCandidate], pd.DataFrame, str]:
    """调用 UIE，并把模型输出转为候选实体。

    UIE 在这里是“增强”：它不是唯一来源，而是补充正则和词表漏掉的实体。
    """

    taskflow = load_uie(schema, args)
    if taskflow is None:
        return [], pd.DataFrame(columns=UIE_RAW_COLUMNS), "disabled_or_unavailable"

    candidates: list[EntityCandidate] = []
    raw_rows: list[dict[str, Any]] = []
    rows = list(sentence_frame.iterrows())

    for batch_start in range(0, len(rows), args.uie_batch_size):
        batch = rows[batch_start : batch_start + args.uie_batch_size]
        texts = [str(row["sentence"]) for _, row in batch]
        try:
            batch_results = taskflow(texts)
        except Exception as exc:
            message = f"Paddle UIE 推理失败：{type(exc).__name__}: {exc}"
            if args.strict_uie:
                raise SystemExit(message) from exc
            print(f"[warning] {message}")
            return candidates, pd.DataFrame(raw_rows, columns=UIE_RAW_COLUMNS), "failed_during_inference"

        for (_, row), result in zip(batch, batch_results):
            normalized = normalize_uie_result(result)
            for schema_name, items in normalized.items():
                entity_type = UIE_SCHEMA_TO_ENTITY_TYPE.get(schema_name, "ADMIN_AREA")
                for item in items:
                    text = clean_entity_text(item.get("text", ""))
                    start_pos = item.get("start", -1)
                    end_pos = item.get("end", -1)
                    probability = float(item.get("probability", 0.0))
                    raw_rows.append(
                        {
                            "sentence_id": row["sentence_id"],
                            "schema_name": schema_name,
                            "entity_text": text,
                            "start_pos": start_pos,
                            "end_pos": end_pos,
                            "probability": probability,
                            "entity_type": entity_type,
                            "sentence": row["sentence"],
                        }
                    )

                    candidate = make_candidate(
                        row=row,
                        entity_text=text,
                        entity_type=entity_type,
                        method="paddle_uie",
                        confidence=probability if probability > 0 else 0.7,
                        start_pos=int(start_pos),
                        end_pos=int(end_pos),
                        admin_lookup=admin_lookup,
                        note=f"uie_schema={schema_name}",
                    )
                    if candidate is not None:
                        candidates.append(candidate)

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    return candidates, pd.DataFrame(raw_rows, columns=UIE_RAW_COLUMNS), "enabled"


def method_priority(method: str) -> int:
    """方法优先级：词表最高，正则其次，UIE 用来补漏和增强。"""

    methods = set(str(method).split("|"))
    if "lexicon_exact" in methods:
        return 100
    if "regex_admin_area" in methods:
        return 75
    if "paddle_uie" in methods:
        return 70
    if "regex_government_org" in methods:
        return 55
    if "regex_address" in methods:
        return 50
    return 0


def candidate_score(candidate: EntityCandidate) -> tuple[int, int, float]:
    length = candidate.end_pos - candidate.start_pos
    return method_priority(candidate.method), length, candidate.confidence


def merge_exact_duplicates(candidates: list[EntityCandidate]) -> list[EntityCandidate]:
    """合并同一句、同位置、同类型、同文本的重复候选。"""

    merged: dict[tuple[str, int, int, str, str], EntityCandidate] = {}

    for candidate in sorted(candidates, key=candidate_score, reverse=True):
        key = (
            candidate.sentence_id,
            candidate.start_pos,
            candidate.end_pos,
            candidate.entity_type,
            candidate.normalized_name,
        )
        existing = merged.get(key)
        if existing is None:
            merged[key] = candidate
            continue

        methods = sorted(set(existing.method.split("|")) | set(candidate.method.split("|")))
        existing.method = "|".join(methods)
        existing.confidence = max(existing.confidence, candidate.confidence)
        existing.is_uie_enhanced = int(existing.is_uie_enhanced or candidate.is_uie_enhanced)

        if not existing.admin_code and candidate.admin_code:
            existing.admin_code = candidate.admin_code
            existing.admin_level = candidate.admin_level
            existing.province = candidate.province
            existing.prefecture = candidate.prefecture
            existing.county = candidate.county
            existing.full_path = candidate.full_path
            existing.is_from_code_table = candidate.is_from_code_table
            existing.is_ambiguous = candidate.is_ambiguous

        notes = [note for note in [existing.note, candidate.note] if note]
        existing.note = "; ".join(dict.fromkeys(notes))

    return list(merged.values())


def overlaps(left: EntityCandidate, right: EntityCandidate) -> bool:
    if left.sentence_id != right.sentence_id:
        return False
    return left.start_pos < right.end_pos and right.start_pos < left.end_pos


def resolve_overlaps(candidates: list[EntityCandidate]) -> list[EntityCandidate]:
    """解决同一句里的重叠候选。

    这里保留“来源更可靠、实体更长、置信度更高”的候选。
    """

    chosen: list[EntityCandidate] = []
    for candidate in sorted(candidates, key=candidate_score, reverse=True):
        if any(overlaps(candidate, kept) for kept in chosen):
            continue
        chosen.append(candidate)
    return sorted(chosen, key=lambda item: (item.sentence_id, item.start_pos, item.end_pos, item.entity_text))


def extract_rule_entities(
    sentence_frame: pd.DataFrame,
    admin_lookup: dict[str, list[AdminRecord]],
    lexicon_index: LexiconIndex,
) -> list[EntityCandidate]:
    """执行词表匹配和正则抽取。"""

    all_candidates: list[EntityCandidate] = []
    for _, row in sentence_frame.iterrows():
        all_candidates.extend(extract_lexicon_entities(row, lexicon_index))
        all_candidates.extend(extract_regex_admin_entities(row, admin_lookup))
        all_candidates.extend(extract_regex_government_entities(row, admin_lookup))
        all_candidates.extend(extract_regex_address_entities(row, admin_lookup))
    return all_candidates


def candidates_to_frame(candidates: list[EntityCandidate]) -> pd.DataFrame:
    rows = [asdict(candidate) for candidate in candidates]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=ENTITY_COLUMNS)
    frame.insert(0, "entity_id", [f"E{index:06d}" for index in range(1, len(frame) + 1)])
    return frame[ENTITY_COLUMNS]


def build_sentence_entities(entity_frame: pd.DataFrame, sentence_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = entity_frame.groupby("sentence_id", sort=False) if not entity_frame.empty else {}

    for _, sentence_row in sentence_frame.iterrows():
        sentence_id = str(sentence_row["sentence_id"])
        group = grouped.get_group(sentence_id) if hasattr(grouped, "groups") and sentence_id in grouped.groups else pd.DataFrame()
        rows.append(
            {
                **sentence_meta(sentence_row),
                "entity_count": len(group),
                "admin_area_count": int(group["entity_type"].eq("ADMIN_AREA").sum()) if not group.empty else 0,
                "gov_org_count": int(group["entity_type"].eq("GOV_ORG").sum()) if not group.empty else 0,
                "address_count": int(group["entity_type"].eq("ADDRESS").sum()) if not group.empty else 0,
                "coded_admin_count": int(group["admin_code"].fillna("").astype(str).ne("").sum()) if not group.empty else 0,
                "entities": " / ".join(group["entity_text"].astype(str).tolist()) if not group.empty else "",
            }
        )

    return pd.DataFrame(rows)


def build_zero_entity_sentences(entity_frame: pd.DataFrame, sentence_frame: pd.DataFrame) -> pd.DataFrame:
    if entity_frame.empty:
        return sentence_frame.copy()
    covered_sentence_ids = set(entity_frame["sentence_id"].astype(str))
    zero_frame = sentence_frame[~sentence_frame["sentence_id"].astype(str).isin(covered_sentence_ids)].copy()
    return zero_frame.reset_index(drop=True)


def build_source_file_coverage(
    sentence_frame: pd.DataFrame,
    entity_frame: pd.DataFrame,
    source_line_frame: pd.DataFrame,
) -> pd.DataFrame:
    sentence_stats = (
        sentence_frame.groupby(["source_file", "year"], dropna=False)
        .agg(sentence_count=("sentence_id", "size"))
        .reset_index()
    )

    if entity_frame.empty:
        entity_stats = pd.DataFrame(columns=["source_file", "entity_count", "sentence_with_entity_count"])
    else:
        entity_stats = (
            entity_frame.groupby("source_file", dropna=False)
            .agg(
                entity_count=("entity_id", "size"),
                sentence_with_entity_count=("sentence_id", "nunique"),
            )
            .reset_index()
        )

    if source_line_frame.empty:
        line_stats = (
            sentence_frame.groupby("source_file", dropna=False)
            .agg(parsed_line_count=("line_no", "nunique"))
            .reset_index()
        )
        line_stats["source_line_count"] = line_stats["parsed_line_count"]
        line_stats["skipped_line_count"] = 0
    else:
        line_stats = (
            source_line_frame.groupby("source_file", dropna=False)
            .agg(
                source_line_count=("line_no", "size"),
                skipped_line_count=("is_skipped", "sum"),
                parsed_line_count=("is_skipped", lambda values: int((values == 0).sum())),
            )
            .reset_index()
        )

    coverage = sentence_stats.merge(line_stats, on="source_file", how="left").merge(entity_stats, on="source_file", how="left")
    coverage["entity_count"] = coverage["entity_count"].fillna(0).astype(int)
    coverage["sentence_with_entity_count"] = coverage["sentence_with_entity_count"].fillna(0).astype(int)
    coverage["zero_entity_sentence_count"] = coverage["sentence_count"] - coverage["sentence_with_entity_count"]
    coverage["sentence_entity_coverage_ratio"] = (
        coverage["sentence_with_entity_count"] / coverage["sentence_count"].replace(0, pd.NA)
    ).fillna(0).round(6)

    columns = [
        "source_file",
        "year",
        "source_line_count",
        "skipped_line_count",
        "parsed_line_count",
        "sentence_count",
        "sentence_with_entity_count",
        "zero_entity_sentence_count",
        "sentence_entity_coverage_ratio",
        "entity_count",
    ]
    return coverage[columns].sort_values(["year", "source_file"]).reset_index(drop=True)


def build_sentence_pattern_summary(sentence_frame: pd.DataFrame, entity_frame: pd.DataFrame) -> pd.DataFrame:
    entity_counts = (
        entity_frame.groupby("sentence_id", dropna=False)
        .agg(entity_count=("entity_id", "size"))
        .reset_index()
        if not entity_frame.empty
        else pd.DataFrame(columns=["sentence_id", "entity_count"])
    )

    rows: list[dict[str, Any]] = []
    for _, row in sentence_frame.iterrows():
        sentence = str(row["sentence"])
        matched_labels = [label for label, pattern in CHANGE_CUE_PATTERNS.items() if pattern.search(sentence)]
        rows.append(
            {
                **sentence_meta(row),
                "sentence_length": len(sentence),
                "cue_labels": "|".join(matched_labels),
                "primary_cue": matched_labels[0] if matched_labels else "other",
                **{f"has_{label}": int(label in matched_labels) for label in CHANGE_CUE_PATTERNS},
            }
        )

    result = pd.DataFrame(rows)
    result = result.merge(entity_counts, on="sentence_id", how="left")
    result["entity_count"] = result["entity_count"].fillna(0).astype(int)
    return result


def build_source_pattern_summary(sentence_pattern_frame: pd.DataFrame) -> pd.DataFrame:
    if sentence_pattern_frame.empty:
        return pd.DataFrame()
    aggregations: dict[str, Any] = {
        "sentence_count": ("sentence_id", "size"),
        "avg_entity_count": ("entity_count", "mean"),
    }
    for label in CHANGE_CUE_PATTERNS:
        aggregations[f"{label}_sentence_count"] = (f"has_{label}", "sum")

    result = (
        sentence_pattern_frame.groupby(["source_file", "year"], dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(["year", "source_file"])
    )
    result["avg_entity_count"] = result["avg_entity_count"].round(4)
    return result


def build_entity_frequency(entity_frame: pd.DataFrame) -> pd.DataFrame:
    if entity_frame.empty:
        return pd.DataFrame(
            columns=[
                "entity_type",
                "normalized_name",
                "count",
                "sentence_count",
                "first_year",
                "last_year",
                "methods",
                "admin_codes",
                "admin_levels",
                "example_sentence",
            ]
        )

    rows: list[dict[str, Any]] = []
    for (entity_type, name), group in entity_frame.groupby(["entity_type", "normalized_name"], dropna=False):
        rows.append(
            {
                "entity_type": entity_type,
                "normalized_name": name,
                "count": len(group),
                "sentence_count": group["sentence_id"].nunique(),
                "first_year": group["year"].min(),
                "last_year": group["year"].max(),
                "methods": "|".join(sorted(set("|".join(group["method"].astype(str)).split("|")))),
                "admin_codes": "|".join(sorted(code for code in group["admin_code"].fillna("").astype(str).unique() if code)),
                "admin_levels": "|".join(sorted(level for level in group["admin_level"].fillna("").astype(str).unique() if level)),
                "example_sentence": str(group.iloc[0]["sentence"]),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["count", "entity_type", "normalized_name"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def build_method_summary(entity_frame: pd.DataFrame) -> pd.DataFrame:
    if entity_frame.empty:
        return pd.DataFrame(columns=["entity_type", "method", "entity_count", "unique_entity_count", "avg_confidence"])

    rows: list[dict[str, Any]] = []
    expanded = entity_frame.assign(method_part=entity_frame["method"].astype(str).str.split("|")).explode("method_part")
    for (entity_type, method), group in expanded.groupby(["entity_type", "method_part"], dropna=False):
        rows.append(
            {
                "entity_type": entity_type,
                "method": method,
                "entity_count": len(group),
                "unique_entity_count": group["normalized_name"].nunique(),
                "avg_confidence": round(float(group["confidence"].mean()), 4),
            }
        )

    return pd.DataFrame(rows).sort_values(["entity_count", "entity_type", "method"], ascending=[False, True, True])


def build_unmatched_admin_entities(entity_frame: pd.DataFrame) -> pd.DataFrame:
    if entity_frame.empty:
        return pd.DataFrame()
    admin_entities = entity_frame[entity_frame["entity_type"].eq("ADMIN_AREA")].copy()
    unmatched = admin_entities[admin_entities["admin_code"].fillna("").astype(str).eq("")]
    return build_entity_frequency(unmatched)


def build_overview(
    sentence_frame: pd.DataFrame,
    entity_frame: pd.DataFrame,
    admin_lookup: dict[str, list[AdminRecord]],
    uie_status: str,
    source_file_coverage: pd.DataFrame,
) -> pd.DataFrame:
    admin_entities = entity_frame[entity_frame["entity_type"].eq("ADMIN_AREA")] if not entity_frame.empty else entity_frame
    return pd.DataFrame(
        [
            {
                "source_file_count": sentence_frame["source_file"].nunique(),
                "sentence_count": len(sentence_frame),
                "sentence_with_entity_count": int(source_file_coverage["sentence_with_entity_count"].sum())
                if not source_file_coverage.empty
                else 0,
                "zero_entity_sentence_count": int(source_file_coverage["zero_entity_sentence_count"].sum())
                if not source_file_coverage.empty
                else 0,
                "admin_lexicon_name_count": len(admin_lookup),
                "entity_count": len(entity_frame),
                "unique_entity_count": entity_frame["normalized_name"].nunique() if not entity_frame.empty else 0,
                "admin_entity_count": len(admin_entities),
                "coded_admin_entity_count": int(admin_entities["admin_code"].fillna("").astype(str).ne("").sum())
                if not admin_entities.empty
                else 0,
                "unmatched_admin_entity_count": int(admin_entities["admin_code"].fillna("").astype(str).eq("").sum())
                if not admin_entities.empty
                else 0,
                "uie_status": uie_status,
                "uie_enhanced_entity_count": int(entity_frame["is_uie_enhanced"].sum()) if not entity_frame.empty else 0,
            }
        ]
    )


def build_output_summary(output_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"file": filename, "rows": len(frame), "columns": len(frame.columns)} for filename, frame in output_tables.items()]
    )


def write_outputs(output_dir: Path, output_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    output_summary = build_output_summary(output_tables)

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in output_tables.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")

    return output_summary


def main() -> None:
    args = parse_args()
    configure_cpu_threads(args.cpu_threads)

    schema = [item.strip() for item in args.uie_schema.split(",") if item.strip()]
    include_township = not args.no_township

    admin_lookup = load_admin_lookup(args.admin_xlsx, include_township=include_township)
    lexicon_index = build_lexicon_index(admin_lookup)
    sentence_frame, source_line_frame, input_description = load_sentence_inputs(args)

    rule_candidates = extract_rule_entities(sentence_frame, admin_lookup, lexicon_index)
    uie_candidates, uie_raw_frame, uie_status = run_uie_entities(sentence_frame, admin_lookup, schema, args)

    merged_candidates = merge_exact_duplicates(rule_candidates + uie_candidates)
    final_candidates = resolve_overlaps(merged_candidates)
    entity_frame = candidates_to_frame(final_candidates)

    sentence_entities = build_sentence_entities(entity_frame, sentence_frame)
    entity_frequency = build_entity_frequency(entity_frame)
    method_summary = build_method_summary(entity_frame)
    unmatched_admin = build_unmatched_admin_entities(entity_frame)
    zero_entity_sentences = build_zero_entity_sentences(entity_frame, sentence_frame)
    source_file_coverage = build_source_file_coverage(sentence_frame, entity_frame, source_line_frame)
    sentence_pattern_summary = build_sentence_pattern_summary(sentence_frame, entity_frame)
    source_pattern_summary = build_source_pattern_summary(sentence_pattern_summary)
    overview = build_overview(sentence_frame, entity_frame, admin_lookup, uie_status, source_file_coverage)

    output_tables = {
        "source_sentence_records.csv": sentence_frame,
        "sentence_pattern_summary.csv": sentence_pattern_summary,
        "entities.csv": entity_frame,
        "entity_frequency.csv": entity_frequency,
        "ner_overview.csv": overview,
    }
    if args.debug_output:
        output_tables.update(
            {
                "source_line_records.csv": source_line_frame,
                "source_file_coverage.csv": source_file_coverage,
                "source_pattern_summary.csv": source_pattern_summary,
                "sentence_entities.csv": sentence_entities,
                "zero_entity_sentences.csv": zero_entity_sentences,
                "unmatched_admin_entities.csv": unmatched_admin,
                "ner_method_summary.csv": method_summary,
                "uie_raw_results.csv": uie_raw_frame,
            }
        )
    output_summary = write_outputs(args.output_dir, output_tables)

    print(f"行政区划编码表：{args.admin_xlsx}")
    print(f"句子输入：{input_description}")
    print(f"NER 输出目录：{args.output_dir}")
    print(f"词表层级：{'省市区 + 乡镇街道' if include_township else '仅省市区'}")
    print(f"UIE 状态：{uie_status}")
    print()
    print("关键输出表：")
    print(output_summary.to_string(index=False))
    print()
    print("NER 总览：")
    print(overview.to_string(index=False))
    print()
    print(f"实体预览：前 {max(0, args.preview_rows)} 行")
    if not entity_frame.empty and args.preview_rows > 0:
        columns = [
            "entity_id",
            "sentence_id",
            "entity_text",
            "entity_type",
            "method",
            "admin_code",
            "confidence",
        ]
        print(entity_frame[columns].head(args.preview_rows).to_string(index=False))


if __name__ == "__main__":
    main()
