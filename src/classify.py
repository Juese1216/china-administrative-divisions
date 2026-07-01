"""完成 13 类关系类型弱监督分类和文本分类器训练。

本脚本读取 RE 生成的 `records.csv`，把记录中的 `relation_type_ids` 展开为
13 类细粒度关系标签。由于一条行政区划调整记录可能同时包含撤销、设立、
驻地、区域承继等多个关系，本任务按多标签分类处理，而不是压成单一粗类。

运行示例：

    conda run --no-capture-output -n nlpEnv python src/classify.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer


DEFAULT_RECORDS = Path("data/processed/relation_extraction/records.csv")
DEFAULT_TRIPLES = Path("data/processed/relation_details/triples.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/classification")

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

RELATION_LABEL_ORDER = [
    "REVOKE_ADMIN",
    "ESTABLISH_ADMIN",
    "RENAME_ADMIN",
    "MERGE_ADMIN",
    "TRANSFER_ADMIN",
    "JURISDICTION_ADMIN",
    "DIRECT_ADMIN",
    "ENTRUST_ADMIN",
    "GOV_RESIDENCE",
    "RESIDENCE_TRANSFER",
    "AREA_INHERITANCE",
    "ADJUSTMENT_EVENT",
    "SCOPE_CONSTRAINT",
]

FALLBACK_RELATION_TYPE_ID = "ADJUSTMENT_EVENT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练并输出 13 类关系类型分类结果。")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--triples", type=Path, default=DEFAULT_TRIPLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=0,
        help=(
            "Limit weak-supervision records collected for each relation class. "
            "Default 0 uses all records."
        ),
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument(
        "--disable-sentence-augmentation",
        action="store_true",
        help="Only train on record-level samples. Default also adds sentence-level relation samples from training records.",
    )
    parser.add_argument(
        "--debug-output",
        action="store_true",
        help="Write feature weights and sklearn text report. Default keeps only report/Web essentials.",
    )
    return parser.parse_args()


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def read_records(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"缺少变更记录文件：{path}")
    frame = pd.read_csv(path)
    required = {
        "record_id",
        "year",
        "record_key",
        "record_text",
        "relation_type_ids",
        "classification_confidence",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{path} 缺少字段：{', '.join(missing)}")
    return frame


def read_triples(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {"source_file", "sentence_id", "sentence", "relation_type_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{path} 缺少字段：{', '.join(missing)}")
    return frame


def split_relation_type_ids(value: Any) -> list[str]:
    raw_ids = [
        item.strip()
        for item in clean_cell(value).split("/")
        if item.strip() and item.strip().lower() != "nan"
    ]
    known = [item for item in raw_ids if item in RELATION_TYPE_NAMES]
    if not known:
        return [FALLBACK_RELATION_TYPE_ID]
    order_index = {label: index for index, label in enumerate(RELATION_LABEL_ORDER)}
    return sorted(set(known), key=lambda item: order_index[item])


def relation_type_name(relation_type_id: str) -> str:
    return RELATION_TYPE_NAMES.get(relation_type_id, relation_type_id)


def relation_confidence(row: pd.Series, relation_type_id: str) -> float:
    if relation_type_id == FALLBACK_RELATION_TYPE_ID and not clean_cell(
        row.get("relation_type_ids", "")
    ):
        return 0.6
    value = pd.to_numeric(row.get("classification_confidence", 0.9), errors="coerce")
    if pd.isna(value):
        return 0.9
    return round(float(value), 4)


def build_rule_outputs(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for _, row in records.iterrows():
        relation_type_ids = split_relation_type_ids(row.get("relation_type_ids", ""))
        is_fallback = not clean_cell(row.get("relation_type_ids", ""))
        for relation_type_id in relation_type_ids:
            rows.append(
                {
                    "classification_id": f"{clean_cell(row['record_id'])}_{relation_type_id}",
                    "record_id": row["record_id"],
                    "year": row["year"],
                    "item_no": row.get("item_no", ""),
                    "record_text": row["record_text"],
                    "before_entities": row.get("before_entities", ""),
                    "after_entities": row.get("after_entities", ""),
                    "action_keywords": row.get("action_keywords", ""),
                    "type_id": relation_type_id,
                    "type_label": relation_type_name(relation_type_id),
                    "classification_confidence": relation_confidence(
                        row, relation_type_id
                    ),
                    "classification_rule": (
                        "无细粒度关系，归入综合事件"
                        if is_fallback
                        else "RE 关系类型弱监督标签"
                    ),
                    "source_main_type_id": row.get("type_id", ""),
                    "source_main_type_label": row.get("type_label", ""),
                    "relation_type_ids": " / ".join(relation_type_ids),
                }
            )
    rule_frame = pd.DataFrame(rows)
    frequency = (
        rule_frame.groupby(["type_id", "type_label"], dropna=False)
        .agg(
            record_count=("record_id", "nunique"),
            label_assignment_count=("classification_id", "size"),
            avg_confidence=("classification_confidence", "mean"),
        )
        .reset_index()
        .sort_values(["record_count", "type_id"], ascending=[False, True])
    )
    frequency["avg_confidence"] = frequency["avg_confidence"].round(4)
    return rule_frame, frequency


def build_record_label_frame(
    records: pd.DataFrame, max_samples_per_class: int
) -> pd.DataFrame:
    rows = []
    for _, row in records.iterrows():
        label_ids = split_relation_type_ids(row.get("relation_type_ids", ""))
        rows.append(
            {
                "record_id": row["record_id"],
                "record_key": row["record_key"],
                "year": row["year"],
                "item_no": row.get("item_no", ""),
                "record_text": row["record_text"],
                "label_ids": " / ".join(label_ids),
                "label_names": " / ".join(
                    relation_type_name(label_id) for label_id in label_ids
                ),
                "label_count": len(label_ids),
                "label_source": "relation_type_weak_label",
                "source_main_type_id": row.get("type_id", ""),
                "source_main_type_label": row.get("type_label", ""),
            }
        )
    frame = pd.DataFrame(rows)
    if max_samples_per_class <= 0:
        return frame
    selected: set[str] = set()
    for relation_type_id in RELATION_LABEL_ORDER:
        class_rows = frame[
            frame["label_ids"].map(
                lambda value: relation_type_id
                in [item.strip() for item in value.split("/") if item.strip()]
            )
        ].sort_values(["label_count", "record_id"])
        selected.update(class_rows.head(max_samples_per_class)["record_id"].tolist())
    return frame[frame["record_id"].isin(selected)].reset_index(drop=True)


def triple_record_key(row: pd.Series) -> str:
    item_no = clean_cell(row.get("item_no", ""))
    if item_no:
        return f"{clean_cell(row['source_file'])}::{item_no}"
    return f"{clean_cell(row['source_file'])}::{clean_cell(row['sentence_id'])}"


def build_sentence_training_frame(
    triples: pd.DataFrame,
    record_label_frame: pd.DataFrame,
    train_record_ids: set[str],
) -> pd.DataFrame:
    if triples.empty:
        return pd.DataFrame(columns=record_label_frame.columns)
    record_lookup = record_label_frame.set_index("record_key").to_dict("index")
    frame = triples.copy()
    frame["record_key"] = frame.apply(triple_record_key, axis=1)
    frame = frame[
        frame["record_key"].isin(record_lookup)
        & frame["relation_type_id"].isin(RELATION_TYPE_NAMES)
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=record_label_frame.columns)
    rows = []
    grouped = frame.groupby(["record_key", "sentence_id"], sort=False)
    for (record_key, sentence_id), group in grouped:
        record = record_lookup.get(record_key)
        if not record or record["record_id"] not in train_record_ids:
            continue
        labels = [
            label
            for label in RELATION_LABEL_ORDER
            if label in set(group["relation_type_id"].astype(str))
        ]
        if not labels:
            continue
        rows.append(
            {
                "record_id": f"{record['record_id']}::{sentence_id}",
                "record_key": record_key,
                "year": record["year"],
                "item_no": record.get("item_no", ""),
                "record_text": clean_cell(group.iloc[0]["sentence"]),
                "label_ids": " / ".join(labels),
                "label_names": " / ".join(relation_type_name(label) for label in labels),
                "label_count": len(labels),
                "label_source": "sentence_relation_weak_label",
                "source_main_type_id": record.get("label_ids", ""),
                "source_main_type_label": record.get("label_names", ""),
            }
        )
    return pd.DataFrame(rows, columns=record_label_frame.columns)


def label_list(value: Any) -> list[str]:
    return [item.strip() for item in clean_cell(value).split("/") if item.strip()]


def label_coverage(indices: list[int], label_sets: dict[int, set[str]]) -> set[str]:
    covered: set[str] = set()
    for index in indices:
        covered.update(label_sets[index])
    return covered


def ensure_split_label_coverage(
    train_indices: list[int], test_indices: list[int], label_sets: dict[int, set[str]]
) -> tuple[list[int], list[int]]:
    all_labels = set(RELATION_LABEL_ORDER)
    train = list(train_indices)
    test = list(test_indices)
    for target, source in [(train, test), (test, train)]:
        missing = all_labels - label_coverage(target, label_sets)
        for label in sorted(missing):
            candidates = [index for index in source if label in label_sets[index]]
            if not candidates:
                continue
            moved = candidates[0]
            source.remove(moved)
            target.append(moved)
    return sorted(train), sorted(test)


def ensure_at_least_one_prediction(
    binary_predictions: Any, probabilities: Any
) -> Any:
    adjusted = binary_predictions.copy()
    for row_index in range(adjusted.shape[0]):
        if adjusted[row_index].sum() == 0:
            adjusted[row_index, probabilities[row_index].argmax()] = 1
    return adjusted


def train_text_classifier(
    training_frame: pd.DataFrame,
    full_frame: pd.DataFrame,
    triples: pd.DataFrame,
    test_size: float,
    use_sentence_augmentation: bool,
) -> dict[str, Any]:
    if len(training_frame) < 2:
        raise SystemExit("分类训练失败：至少需要 2 条样本。")

    training_frame = training_frame.reset_index(drop=True)
    label_sets = {
        index: set(label_list(value))
        for index, value in enumerate(training_frame["label_ids"].tolist())
    }
    train_indices, test_indices = train_test_split(
        list(range(len(training_frame))),
        test_size=test_size,
        random_state=42,
    )
    train_indices, test_indices = ensure_split_label_coverage(
        list(train_indices), list(test_indices), label_sets
    )

    train_record_frame = training_frame.iloc[train_indices].copy()
    test_frame = training_frame.iloc[test_indices].copy()
    sentence_frame = (
        build_sentence_training_frame(
            triples, training_frame, set(train_record_frame["record_id"].tolist())
        )
        if use_sentence_augmentation
        else pd.DataFrame(columns=training_frame.columns)
    )
    train_frame = pd.concat(
        [train_record_frame, sentence_frame], ignore_index=True, sort=False
    )
    mlb = MultiLabelBinarizer(classes=RELATION_LABEL_ORDER)
    train_y = mlb.fit_transform(train_frame["label_ids"].map(label_list))
    test_y = mlb.transform(test_frame["label_ids"].map(label_list))

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="liblinear",
                    )
                ),
            ),
        ]
    )
    model.fit(train_frame["record_text"], train_y)

    test_probabilities = model.predict_proba(test_frame["record_text"])
    test_pred = ensure_at_least_one_prediction(
        model.predict(test_frame["record_text"]), test_probabilities
    )
    full_probabilities = model.predict_proba(full_frame["record_text"])
    full_pred = ensure_at_least_one_prediction(
        model.predict(full_frame["record_text"]), full_probabilities
    )
    full_y = mlb.transform(full_frame["label_ids"].map(label_list))

    full_pred_labels = [
        " / ".join(relation_type_name(label) for label in labels)
        for labels in mlb.inverse_transform(full_pred)
    ]
    full_pred_ids = [
        " / ".join(labels) for labels in mlb.inverse_transform(full_pred)
    ]
    prediction_frame = full_frame[
        [
            "record_id",
            "year",
            "item_no",
            "record_text",
            "label_ids",
            "label_names",
            "source_main_type_label",
        ]
    ].copy()
    prediction_frame["ml_predicted_type_ids"] = full_pred_ids
    prediction_frame["ml_predicted_types"] = full_pred_labels
    prediction_frame["ml_confidence"] = full_probabilities.max(axis=1).round(4)
    prediction_frame["rule_equals_ml"] = [
        set(label_list(actual)) == set(label_list(predicted))
        for actual, predicted in zip(
            prediction_frame["label_ids"],
            prediction_frame["ml_predicted_type_ids"],
            strict=True,
        )
    ]

    test_hamming_loss = hamming_loss(test_y, test_pred)
    evaluation = pd.DataFrame(
        [
            {
                "training_mode": "weak_supervision_multilabel_relation_types",
                "training_sample_count": len(train_frame),
                "training_record_count": len(train_record_frame),
                "sentence_training_sample_count": len(sentence_frame),
                "test_sample_count": len(test_frame),
                "total_labeled_sample_count": len(training_frame),
                "total_label_assignment_count": int(train_y.sum() + test_y.sum()),
                "training_label_assignment_count": int(train_y.sum()),
                "class_count": len(mlb.classes_),
                "accuracy": round(accuracy_score(test_y, test_pred), 4),
                "exact_match_accuracy": round(accuracy_score(test_y, test_pred), 4),
                "micro_f1": round(f1_score(test_y, test_pred, average="micro"), 4),
                "macro_f1": round(f1_score(test_y, test_pred, average="macro"), 4),
                "hamming_loss": round(test_hamming_loss, 4),
                "label_accuracy": round(1 - test_hamming_loss, 4),
                "rule_ml_agreement_rate": round(accuracy_score(full_y, full_pred), 4),
            }
        ]
    )

    report_text = classification_report(
        test_y,
        test_pred,
        target_names=[relation_type_name(label) for label in mlb.classes_],
        zero_division=0,
    )
    matrix = build_multilabel_confusion_matrix(test_y, test_pred, list(mlb.classes_))
    top_features = extract_top_features(
        model, [relation_type_name(label) for label in mlb.classes_], top_n=15
    )

    return {
        "prediction_frame": prediction_frame,
        "evaluation": evaluation,
        "report_text": report_text,
        "confusion_matrix": matrix,
        "top_features": top_features,
    }


def build_multilabel_confusion_matrix(
    test_y: Any, pred_y: Any, class_ids: list[str]
) -> pd.DataFrame:
    matrices = multilabel_confusion_matrix(test_y, pred_y)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_y, pred_y, average=None, zero_division=0
    )
    rows = []
    for index, class_id in enumerate(class_ids):
        tn, fp, fn, tp = matrices[index].ravel()
        rows.append(
            {
                "actual_type": relation_type_name(class_id),
                "type_id": class_id,
                "true_positive": int(tp),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_negative": int(tn),
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(f1[index]), 4),
            }
        )
    return pd.DataFrame(rows)


def extract_top_features(
    model: Pipeline, class_names: list[str], top_n: int
) -> pd.DataFrame:
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()
    rows = []
    for class_index, class_name in enumerate(class_names):
        estimator = classifier.estimators_[class_index]
        coefs = estimator.coef_[0]
        top_indices = coefs.argsort()[-top_n:][::-1]
        for rank, feature_index in enumerate(top_indices, start=1):
            rows.append(
                {
                    "class_name": class_name,
                    "rank": rank,
                    "feature": feature_names[feature_index],
                    "weight": round(float(coefs[feature_index]), 6),
                }
            )
    return pd.DataFrame(rows)


def write_outputs(
    output_dir: Path,
    outputs: dict[str, pd.DataFrame],
    report_text: str,
    debug_output: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
    if debug_output:
        (output_dir / "ml_classification_report.txt").write_text(
            report_text, encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    records = read_records(args.records)
    triples = read_triples(args.triples)
    rule_frame, frequency = build_rule_outputs(records)
    record_label_frame = build_record_label_frame(records, args.max_samples_per_class)
    result = train_text_classifier(
        record_label_frame,
        record_label_frame,
        triples,
        args.test_size,
        not args.disable_sentence_augmentation,
    )

    overview = pd.DataFrame(
        [
            {
                "record_count": int(records["record_id"].nunique()),
                "rule_type_count": int(frequency["type_label"].nunique()),
                "training_sample_count": int(
                    result["evaluation"].iloc[0]["training_sample_count"]
                ),
                "training_record_count": int(
                    result["evaluation"].iloc[0]["training_record_count"]
                ),
                "sentence_training_sample_count": int(
                    result["evaluation"].iloc[0]["sentence_training_sample_count"]
                ),
                "test_sample_count": int(result["evaluation"].iloc[0]["test_sample_count"]),
                "total_labeled_sample_count": int(len(record_label_frame)),
                "total_label_assignment_count": int(len(rule_frame)),
                "training_label_assignment_count": int(
                    result["evaluation"].iloc[0]["training_label_assignment_count"]
                ),
                "ml_accuracy": result["evaluation"].iloc[0]["accuracy"],
                "exact_match_accuracy": result["evaluation"].iloc[0][
                    "exact_match_accuracy"
                ],
                "micro_f1": result["evaluation"].iloc[0]["micro_f1"],
                "macro_f1": result["evaluation"].iloc[0]["macro_f1"],
                "label_accuracy": result["evaluation"].iloc[0]["label_accuracy"],
                "rule_ml_agreement_rate": result["evaluation"].iloc[0][
                    "rule_ml_agreement_rate"
                ],
                "class_frequency_json": json.dumps(
                    frequency.set_index("type_label")["record_count"].to_dict(),
                    ensure_ascii=False,
                ),
            }
        ]
    )

    outputs = {
        "rule_classification.csv": rule_frame,
        "class_frequency.csv": frequency,
        "ml_classification_predictions.csv": result["prediction_frame"],
        "ml_evaluation.csv": result["evaluation"],
        "ml_confusion_matrix.csv": result["confusion_matrix"],
        "classification_overview.csv": overview,
    }
    if args.debug_output:
        outputs["training_samples.csv"] = record_label_frame
        outputs["ml_top_features.csv"] = result["top_features"]
    write_outputs(args.output_dir, outputs, result["report_text"], args.debug_output)

    print(f"分类输出目录：{args.output_dir}")
    print(overview.to_string(index=False))
    print("\n关系类型分类频次：")
    print(frequency.to_string(index=False))


if __name__ == "__main__":
    main()
