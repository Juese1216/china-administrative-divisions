"""根据 PaddleNLP token 结果统计关系候选。

本脚本只读取 `paddle_pos_tokens.csv`，使用 pandas 统计动词、上下文、
共现关系和可能的变更动作。它不再调用 PaddleNLP。

运行示例：

    conda run -n nlpEnv python src/relation_statistics.py
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TOKEN_CSV = Path("data/processed/paddle_nlp_tokens/paddle_pos_tokens.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/relation_statistics")

REQUIRED_COLUMNS = {
    "sentence_id",
    "source_file",
    "year",
    "line_no",
    "item_no",
    "sentence",
    "token",
    "pos",
    "pos_name",
    "is_verb",
    "is_location",
    "token_position_ratio",
    "prev_token",
    "prev_pos",
    "next_token",
    "next_pos",
    "is_sentence_initial",
    "is_after_punctuation",
    "near_location",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build statistical relation candidates from PaddleNLP tokens."
    )
    parser.add_argument("--token-csv", type=Path, default=DEFAULT_TOKEN_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--min-verb-count",
        type=int,
        default=1,
        help="Minimum count for optional filtering. The default 1 means all verbs are included.",
    )
    parser.add_argument("--top-examples", type=int, default=3)
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=20,
        help="Rows printed in the terminal preview. CSV files always contain the full result.",
    )
    parser.add_argument(
        "--debug-output",
        action="store_true",
        help="Write full token/context diagnostic CSV files. Default keeps only report essentials.",
    )
    return parser.parse_args()


def load_token_frame(token_csv: Path) -> pd.DataFrame:
    if not token_csv.exists():
        raise SystemExit(f"Token CSV does not exist: {token_csv}")
    frame = pd.read_csv(token_csv)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise SystemExit(f"Token CSV is missing required columns: {', '.join(missing)}")
    return frame


def build_sentence_stats(token_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sentence_id, group in token_frame.groupby("sentence_id", sort=False):
        pos_counter = Counter(group["pos"].astype(str))
        first = group.iloc[0]
        verbs = group[group["is_verb"].astype(int).eq(1)]["token"].astype(str).tolist()
        rows.append(
            {
                "sentence_id": sentence_id,
                "source_file": first["source_file"],
                "year": first["year"],
                "line_no": first["line_no"],
                "item_no": first["item_no"],
                "sentence": first["sentence"],
                "token_count": len(group),
                "verb_count": int(group["is_verb"].astype(int).sum()),
                "location_count": int(group["is_location"].astype(int).sum()),
                "verbs": " / ".join(verbs),
                "pos_distribution": ";".join(
                    f"{pos}:{count}" for pos, count in sorted(pos_counter.items())
                ),
            }
        )
    return pd.DataFrame(rows)


def build_verb_tokens(token_frame: pd.DataFrame) -> pd.DataFrame:
    """返回所有被 PaddleNLP 标记为动词或动词性词的 token。"""
    verbs = token_frame[token_frame["is_verb"].astype(int).eq(1)].copy()
    return verbs.sort_values(["sentence_id", "token_index"]).reset_index(drop=True)


def build_verb_frequency(token_frame: pd.DataFrame, min_count: int) -> pd.DataFrame:
    verbs = build_verb_tokens(token_frame)
    if verbs.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    total_verbs = len(verbs)
    for verb, group in verbs.groupby("token", dropna=False):
        count = len(group)
        if count < min_count:
            continue
        rows.append(
            {
                "verb": verb,
                "count": count,
                "frequency_ratio": round(count / total_verbs, 6),
                "sentence_count": group["sentence_id"].nunique(),
                "first_year": int(group["year"].min()),
                "last_year": int(group["year"].max()),
                "pos_tags": "/".join(sorted(group["pos"].astype(str).unique())),
                "pos_names": "/".join(sorted(group["pos_name"].astype(str).unique())),
                "initial_rate": round(
                    group["is_sentence_initial"].astype(int).mean(), 6
                ),
                "after_punctuation_rate": round(
                    group["is_after_punctuation"].astype(int).mean(), 6
                ),
                "near_location_rate": round(
                    group["near_location"].astype(int).mean(), 6
                ),
                "avg_position_ratio": round(
                    group["token_position_ratio"].astype(float).mean(), 6
                ),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(["count", "verb"], ascending=[False, True]).reset_index(
        drop=True
    )
    result.insert(0, "rank", range(1, len(result) + 1))
    result["cumulative_count"] = result["count"].cumsum()
    result["cumulative_ratio"] = (result["cumulative_count"] / total_verbs).round(6)
    return result


def build_context_frames(token_frame: pd.DataFrame, min_count: int) -> pd.DataFrame:
    verbs = build_verb_tokens(token_frame)
    if verbs.empty:
        return pd.DataFrame()
    verbs["frame"] = (
        verbs["prev_pos"].fillna("").astype(str)
        + ":"
        + verbs["prev_token"].fillna("").astype(str)
        + " <- "
        + verbs["token"].astype(str)
        + " -> "
        + verbs["next_pos"].fillna("").astype(str)
        + ":"
        + verbs["next_token"].fillna("").astype(str)
    )
    frame_counts = (
        verbs.groupby(["token", "frame"], dropna=False)
        .agg(
            count=("token", "size"),
            sentence_count=("sentence_id", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            near_location_rate=("near_location", "mean"),
        )
        .reset_index()
        .rename(columns={"token": "verb"})
    )
    frame_counts = frame_counts[frame_counts["count"].ge(min_count)].copy()
    if frame_counts.empty:
        return frame_counts
    frame_counts["near_location_rate"] = frame_counts["near_location_rate"].round(6)
    return frame_counts.sort_values(
        ["count", "verb", "frame"], ascending=[False, True, True]
    ).reset_index(drop=True)


def build_verb_cooccurrence(
    sentence_frame: pd.DataFrame, min_count: int
) -> pd.DataFrame:
    pair_counter: Counter[tuple[str, str]] = Counter()
    sentence_counter: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], str] = {}

    for _, row in sentence_frame.iterrows():
        verbs = [verb for verb in str(row["verbs"]).split(" / ") if verb]
        unique_verbs = sorted(set(verbs))
        for left, right in combinations(unique_verbs, 2):
            pair = (left, right)
            pair_counter[pair] += verbs.count(left) + verbs.count(right)
            sentence_counter[pair] += 1
            examples.setdefault(pair, str(row["sentence"]))

    rows = [
        {
            "verb_a": left,
            "verb_b": right,
            "cooccurrence_score": score,
            "sentence_count": sentence_counter[(left, right)],
            "example_sentence": examples[(left, right)],
        }
        for (left, right), score in pair_counter.items()
        if sentence_counter[(left, right)] >= min_count
    ]
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["sentence_count", "cooccurrence_score", "verb_a", "verb_b"],
        ascending=[False, False, True, True],
    )


def collect_examples(token_frame: pd.DataFrame, verb: str, max_examples: int) -> str:
    examples = (
        token_frame[token_frame["token"].astype(str).eq(str(verb))]
        .drop_duplicates("sentence_id")["sentence"]
        .head(max_examples)
        .astype(str)
        .tolist()
    )
    return " || ".join(examples)


def build_relation_candidates(
    token_frame: pd.DataFrame,
    verb_frequency: pd.DataFrame,
    context_frames: pd.DataFrame,
    top_examples: int,
) -> pd.DataFrame:
    if verb_frequency.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    max_count = max(float(verb_frequency["count"].max()), 1.0)
    for _, row in verb_frequency.iterrows():
        count = int(row["count"])
        frequency_score = math.log1p(count) / math.log1p(max_count)
        context_score = (
            0.45 * float(row["near_location_rate"])
            + 0.25 * float(row["after_punctuation_rate"])
            + 0.20 * float(row["initial_rate"])
            + 0.10 * (1 - float(row["avg_position_ratio"]))
        )
        relation_score = round(frequency_score * context_score, 6)
        frames = (
            context_frames[context_frames["verb"].astype(str).eq(str(row["verb"]))][
                "frame"
            ]
            .head(5)
            .tolist()
        )
        rows.append(
            {
                "candidate_relation": row["verb"],
                "relation_score": relation_score,
                "count": count,
                "frequency_ratio": row["frequency_ratio"],
                "sentence_count": row["sentence_count"],
                "first_year": row["first_year"],
                "last_year": row["last_year"],
                "near_location_rate": row["near_location_rate"],
                "initial_rate": row["initial_rate"],
                "after_punctuation_rate": row["after_punctuation_rate"],
                "representative_frames": " || ".join(frames),
                "example_sentences": collect_examples(
                    token_frame, str(row["verb"]), top_examples
                ),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["relation_score", "count", "candidate_relation"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_overview(
    token_frame: pd.DataFrame, relation_candidates: pd.DataFrame
) -> pd.DataFrame:
    verbs = build_verb_tokens(token_frame)
    return pd.DataFrame(
        [
            {
                "sentence_count": token_frame["sentence_id"].nunique(),
                "token_count": len(token_frame),
                "verb_token_count": int(token_frame["is_verb"].astype(int).sum()),
                "unique_verb_count": verbs["token"].nunique(),
                "location_like_token_count": int(
                    token_frame["is_location"].astype(int).sum()
                ),
                "candidate_relation_count": len(relation_candidates),
            }
        ]
    )


def build_output_summary(output_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for filename, frame in output_tables.items():
        rows.append(
            {"file": filename, "rows": len(frame), "columns": len(frame.columns)}
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    token_frame = load_token_frame(args.token_csv)
    verb_tokens = build_verb_tokens(token_frame)
    sentence_frame = build_sentence_stats(token_frame)
    verb_frequency = build_verb_frequency(token_frame, args.min_verb_count)
    context_frames = build_context_frames(token_frame, args.min_verb_count)
    cooccurrence = build_verb_cooccurrence(sentence_frame, args.min_verb_count)
    relation_candidates = build_relation_candidates(
        token_frame,
        verb_frequency,
        context_frames,
        args.top_examples,
    )
    overview = build_overview(token_frame, relation_candidates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verb_frequency.to_csv(
        args.output_dir / "verb_frequency.csv", index=False, encoding="utf-8-sig"
    )
    relation_candidates.to_csv(
        args.output_dir / "relation_type_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overview.to_csv(
        args.output_dir / "relation_statistics_overview.csv",
        index=False,
        encoding="utf-8-sig",
    )

    output_tables = {
        "verb_frequency.csv": verb_frequency,
        "relation_type_candidates.csv": relation_candidates,
        "relation_statistics_overview.csv": overview,
    }
    if args.debug_output:
        debug_tables = {
            "verb_tokens.csv": verb_tokens,
            "sentence_pos_stats.csv": sentence_frame,
            "verb_context_frames.csv": context_frames,
            "verb_cooccurrence.csv": cooccurrence,
        }
        for filename, frame in debug_tables.items():
            frame.to_csv(args.output_dir / filename, index=False, encoding="utf-8-sig")
        output_tables.update(debug_tables)
    output_summary = build_output_summary(output_tables)

    print(f"loaded token csv: {args.token_csv}")
    print(f"saved statistics output directory: {args.output_dir}")
    print()
    print("关键输出表：")
    print(output_summary.to_string(index=False))
    print()
    print("Overall statistics:")
    print(overview.to_string(index=False))
    print()
    print(f"Relation candidate preview: first {max(0, args.preview_rows)} rows only.")
    print("Open relation_type_candidates.csv for the full table.")
    if relation_candidates.empty:
        print("<empty>")
    elif args.preview_rows > 0:
        preview_columns = [
            "candidate_relation",
            "relation_score",
            "count",
            "sentence_count",
            "near_location_rate",
        ]
        print(
            relation_candidates[preview_columns]
            .head(args.preview_rows)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
