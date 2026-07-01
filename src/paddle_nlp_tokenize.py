"""使用 PaddleNLP 做切词和词性标注。

本脚本只负责读取民政部年度区划变更文本，切分句子，调用 PaddleNLP 做词性
标注，并导出 token 级 CSV。它不做关系统计、NER 或地名修复。

运行示例：

    conda run -n nlpEnv python src/paddle_nlp_tokenize.py
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = Path("data/source/mca_changes")
DEFAULT_OUTPUT_DIR = Path("data/processed/paddle_nlp_tokens")

POS_NAMES = {
    "n": "普通名词",
    "f": "方位名词",
    "s": "处所名词",
    "t": "时间名词",
    "nr": "人名",
    "ns": "地名",
    "nt": "机构团体名",
    "nw": "作品名",
    "nz": "其他专名",
    "v": "动词",
    "vd": "副动词",
    "vn": "名动词",
    "a": "形容词",
    "ad": "副形词",
    "an": "名形词",
    "d": "副词",
    "m": "数量词",
    "q": "量词",
    "r": "代词",
    "p": "介词",
    "c": "连词",
    "u": "助词",
    "xc": "其他虚词",
    "w": "标点符号",
    "PER": "人名",
    "LOC": "地名",
    "ORG": "机构名",
    "TIME": "时间",
}

PUNCTUATION_TOKENS = {"，", "。", "；", ";", ":", "：", "、", "（", "）", "(", ")"}
LOCATION_POS = {"LOC", "ns", "s"}
VERB_PREFIXES = ("v",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PaddleNLP POS tagging and export token CSV."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--limit-sentences", type=int, default=0)
    return parser.parse_args()


def limit_cpu_threads(cpu_threads: int) -> None:
    """在 Paddle 启动前限制常见数学库的 CPU 线程数。"""
    threads = str(max(1, cpu_threads))
    os.environ.setdefault("OMP_NUM_THREADS", threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", threads)
    os.environ.setdefault("MKL_NUM_THREADS", threads)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", threads)
    os.environ.setdefault("FLAGS_cpu_math_library_num_threads", threads)


def set_paddle_device(device: str) -> str:
    try:
        import paddle
    except ImportError as exc:
        raise SystemExit("Paddle is not installed in the nlpEnv environment.") from exc

    if device == "auto":
        has_cuda = bool(getattr(paddle, "is_compiled_with_cuda", lambda: False)())
        device = "gpu" if has_cuda else "cpu"
    try:
        paddle.set_device(device)
    except Exception as exc:  # noqa: BLE001 - show environment issues directly.
        raise SystemExit(f"Failed to set Paddle device to {device!r}: {exc}") from exc
    return device


def load_pos_tagger() -> Any:
    try:
        from paddlenlp import Taskflow
    except ImportError as exc:
        raise SystemExit(
            "PaddleNLP is not installed in the nlpEnv environment."
        ) from exc
    return Taskflow("pos_tagging")


def clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def is_title_line(line: str) -> bool:
    return "县级以上行政区划变更情况" in line


def infer_year(path: Path) -> int | None:
    match = re.match(r"(\d{4})_", path.name)
    return int(match.group(1)) if match else None


def split_sentences(text: str) -> list[str]:
    text = clean_line(text)
    if not text:
        return []
    pieces = re.split(r"(?<=[。！？；;])\s*", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def source_files(input_dir: Path, limit_files: int) -> list[Path]:
    files = sorted(path for path in input_dir.glob("*.txt") if path.is_file())
    if limit_files > 0:
        files = files[:limit_files]
    if not files:
        raise SystemExit(f"No txt files found in {input_dir}")
    return files


def iter_sentence_records(
    files: list[Path], limit_sentences: int
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sentence_index = 0

    for path in files:
        year = infer_year(path)
        item_no = ""
        for line_no, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = clean_line(raw_line)
            if (
                not line
                or is_title_line(line)
                or line.startswith("（以行政区划变更公布时间为序")
            ):
                continue

            item_match = re.match(r"^([一二三四五六七八九十百〇零两]+)、(.+)$", line)
            if item_match:
                item_no = item_match.group(1)
                line = item_match.group(2).strip()

            for sentence in split_sentences(line):
                sentence_index += 1
                records.append(
                    {
                        "sentence_id": f"S{sentence_index:06d}",
                        "source_file": path.name,
                        "year": year,
                        "line_no": line_no,
                        "item_no": item_no,
                        "sentence": sentence,
                    }
                )
                if limit_sentences > 0 and len(records) >= limit_sentences:
                    return records
    return records


def normalize_taskflow_result(result: Any) -> list[list[tuple[str, str]]]:
    if result and isinstance(result[0], tuple):
        return [result]
    return result


def find_token_position(sentence: str, token: str, cursor: int) -> tuple[int, int, int]:
    start = sentence.find(token, cursor)
    if start < 0:
        start = cursor
    end = start + len(token)
    return start, end, end


def is_verb_pos(pos: str) -> bool:
    return pos.startswith(VERB_PREFIXES)


def is_location_pos(pos: str) -> bool:
    return pos in LOCATION_POS


def near_location(
    tagged_tokens: list[tuple[str, str]], token_index_zero_based: int, window: int = 3
) -> bool:
    start = max(0, token_index_zero_based - window)
    end = min(len(tagged_tokens), token_index_zero_based + window + 1)
    return any(is_location_pos(pos) for _, pos in tagged_tokens[start:end])


def tag_sentences(
    records: list[dict[str, Any]], batch_size: int, sleep_seconds: float
) -> list[dict[str, Any]]:
    tagger = load_pos_tagger()
    rows: list[dict[str, Any]] = []

    for start_index in range(0, len(records), batch_size):
        batch_records = records[start_index : start_index + batch_size]
        sentences = [record["sentence"] for record in batch_records]
        tagged_batch = normalize_taskflow_result(tagger(sentences))
        if len(tagged_batch) != len(batch_records):
            raise SystemExit(
                "PaddleNLP returned a different number of tagging results than input sentences."
            )

        for record, tagged_tokens in zip(batch_records, tagged_batch):
            tagged_tokens = [(str(token), str(pos)) for token, pos in tagged_tokens]
            cursor = 0
            total_tokens = len(tagged_tokens)

            for token_index, (token, pos) in enumerate(tagged_tokens, start=1):
                start_pos, end_pos, cursor = find_token_position(
                    record["sentence"], token, cursor
                )
                prev_token = (
                    tagged_tokens[token_index - 2][0] if token_index > 1 else ""
                )
                prev_pos = tagged_tokens[token_index - 2][1] if token_index > 1 else ""
                next_token = (
                    tagged_tokens[token_index][0] if token_index < total_tokens else ""
                )
                next_pos = (
                    tagged_tokens[token_index][1] if token_index < total_tokens else ""
                )

                rows.append(
                    {
                        **record,
                        "token_index": token_index,
                        "token": token,
                        "pos": pos,
                        "pos_name": POS_NAMES.get(pos, pos),
                        "start_pos": start_pos,
                        "end_pos": end_pos,
                        "is_verb": int(is_verb_pos(pos)),
                        "is_location": int(is_location_pos(pos)),
                        "token_position_ratio": (
                            round(token_index / total_tokens, 6) if total_tokens else 0
                        ),
                        "prev_token": prev_token,
                        "prev_pos": prev_pos,
                        "next_token": next_token,
                        "next_pos": next_pos,
                        "is_sentence_initial": int(token_index == 1),
                        "is_after_punctuation": int(
                            prev_token in PUNCTUATION_TOKENS or prev_pos == "w"
                        ),
                        "near_location": int(
                            near_location(tagged_tokens, token_index - 1)
                        ),
                    }
                )

        done = min(start_index + batch_size, len(records))
        print(f"PaddleNLP POS tagged sentences: {done}/{len(records)}")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return rows


def build_tokenize_overview(
    records: list[dict[str, Any]], token_frame: pd.DataFrame
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sentence_count": len(records),
                "token_count": len(token_frame),
                "verb_token_count": int(token_frame["is_verb"].sum()),
                "unique_verb_count": token_frame[
                    token_frame["is_verb"].astype(int).eq(1)
                ]["token"].nunique(),
                "location_like_token_count": int(token_frame["is_location"].sum()),
            }
        ]
    )


def main() -> None:
    args = parse_args()
    limit_cpu_threads(args.cpu_threads)
    device = set_paddle_device(args.device)
    files = source_files(args.input_dir, args.limit_files)
    records = iter_sentence_records(files, args.limit_sentences)
    if not records:
        raise SystemExit("No sentences found. Please check input files.")

    print(f"using paddle device: {device}")
    print(f"source files: {len(files)}")
    print(f"sentences: {len(records)}")

    token_rows = tag_sentences(records, args.batch_size, args.sleep_seconds)
    token_frame = pd.DataFrame(token_rows)
    sentence_frame = pd.DataFrame(records)
    overview = build_tokenize_overview(records, token_frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    token_frame.to_csv(
        args.output_dir / "paddle_pos_tokens.csv", index=False, encoding="utf-8-sig"
    )
    sentence_frame.to_csv(
        args.output_dir / "sentence_records.csv", index=False, encoding="utf-8-sig"
    )
    overview.to_csv(
        args.output_dir / "tokenize_overview.csv", index=False, encoding="utf-8-sig"
    )

    print(f"saved token output directory: {args.output_dir}")
    print(overview.to_string(index=False))


if __name__ == "__main__":
    main()
