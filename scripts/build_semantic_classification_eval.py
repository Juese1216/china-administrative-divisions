"""Build the semantic-aggregation small-sample classification evaluation.

The 13-class classifier keeps the original augmented training setup. This script
trains a separate 6-class semantic classifier on fewer record-level samples and
writes real evaluation CSVs for the Web system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORDS_PATH = PROJECT_ROOT / "data/processed/relation_extraction/records.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/classification"
MAX_TRAIN_RECORDS_PER_CLASS = 30
TEST_SIZE = 0.20
RANDOM_STATE = 42

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
FALLBACK_RELATION_TYPE_ID = "ADJUSTMENT_EVENT"

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


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


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


def semantic_labels(type_ids: list[str]) -> list[str]:
    labels = {
        SEMANTIC_CLASS_BY_TYPE_ID.get(type_id, "其他综合调整")
        for type_id in type_ids
    }
    return [name for name in SEMANTIC_CLASS_ORDER if name in labels]


def label_list(value: Any) -> list[str]:
    return [item.strip() for item in clean_cell(value).split("/") if item.strip()]


def build_semantic_frame(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in records.iterrows():
        labels = semantic_labels(split_relation_type_ids(row.get("relation_type_ids")))
        rows.append(
            {
                "record_id": row["record_id"],
                "year": row["year"],
                "record_text": row["record_text"],
                "label_ids": " / ".join(labels),
                "label_names": " / ".join(labels),
                "label_count": len(labels),
            }
        )
    return pd.DataFrame(rows)


def label_coverage(indices: list[int], label_sets: dict[int, set[str]]) -> set[str]:
    covered: set[str] = set()
    for index in indices:
        covered.update(label_sets[index])
    return covered


def ensure_split_label_coverage(
    train_indices: list[int], test_indices: list[int], label_sets: dict[int, set[str]]
) -> tuple[list[int], list[int]]:
    train = list(train_indices)
    test = list(test_indices)
    for target, source in [(train, test), (test, train)]:
        missing = set(SEMANTIC_CLASS_ORDER) - label_coverage(target, label_sets)
        for label in sorted(missing):
            candidates = [index for index in source if label in label_sets[index]]
            if not candidates:
                continue
            moved = candidates[0]
            source.remove(moved)
            target.append(moved)
    return sorted(train), sorted(test)


def select_balanced_training_records(
    train_pool: pd.DataFrame, max_records_per_class: int
) -> pd.DataFrame:
    selected: set[str] = set()
    for class_name in SEMANTIC_CLASS_ORDER:
        class_rows = train_pool[
            train_pool["label_ids"].map(lambda value: class_name in label_list(value))
        ].sort_values(["label_count", "record_id"])
        selected.update(class_rows.head(max_records_per_class)["record_id"].tolist())
    return train_pool[train_pool["record_id"].isin(selected)].reset_index(drop=True)


def ensure_at_least_one_prediction(binary_predictions: Any, probabilities: Any) -> Any:
    adjusted = binary_predictions.copy()
    for row_index in range(adjusted.shape[0]):
        if adjusted[row_index].sum() == 0:
            adjusted[row_index, probabilities[row_index].argmax()] = 1
    return adjusted


def build_confusion_matrix(test_y: Any, pred_y: Any) -> pd.DataFrame:
    matrices = multilabel_confusion_matrix(test_y, pred_y)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_y, pred_y, average=None, zero_division=0
    )
    rows: list[dict[str, Any]] = []
    for index, class_name in enumerate(SEMANTIC_CLASS_ORDER):
        tn, fp, fn, tp = matrices[index].ravel()
        rows.append(
            {
                "actual_type": class_name,
                "type_id": class_name,
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


def train_and_evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.reset_index(drop=True)
    label_sets = {
        index: set(label_list(value))
        for index, value in enumerate(frame["label_ids"].tolist())
    }
    train_indices, test_indices = train_test_split(
        list(range(len(frame))),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_indices, test_indices = ensure_split_label_coverage(
        train_indices, test_indices, label_sets
    )
    train_pool = frame.iloc[train_indices].copy().reset_index(drop=True)
    test_frame = frame.iloc[test_indices].copy().reset_index(drop=True)
    train_frame = select_balanced_training_records(
        train_pool, MAX_TRAIN_RECORDS_PER_CLASS
    )

    mlb = MultiLabelBinarizer(classes=SEMANTIC_CLASS_ORDER)
    train_y = mlb.fit_transform(train_frame["label_ids"].map(label_list))
    test_y = mlb.transform(test_frame["label_ids"].map(label_list))
    full_y = mlb.transform(frame["label_ids"].map(label_list))

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
    full_probabilities = model.predict_proba(frame["record_text"])
    full_pred = ensure_at_least_one_prediction(
        model.predict(frame["record_text"]), full_probabilities
    )
    test_hamming_loss = hamming_loss(test_y, test_pred)
    full_hamming_loss = hamming_loss(full_y, full_pred)

    evaluation = pd.DataFrame(
        [
            {
                "training_mode": "semantic_aggregation_small_sample_balanced",
                "selection_strategy": (
                    f"semantic_train_pool_top_{MAX_TRAIN_RECORDS_PER_CLASS}_per_class"
                ),
                "training_sample_count": len(train_frame),
                "training_record_count": len(train_frame),
                "sentence_training_sample_count": 0,
                "test_sample_count": len(test_frame),
                "total_labeled_sample_count": len(frame),
                "total_label_assignment_count": int(full_y.sum()),
                "training_label_assignment_count": int(train_y.sum()),
                "class_count": len(SEMANTIC_CLASS_ORDER),
                "accuracy": round(accuracy_score(test_y, test_pred), 4),
                "exact_match_accuracy": round(accuracy_score(test_y, test_pred), 4),
                "micro_f1": round(f1_score(test_y, test_pred, average="micro"), 4),
                "macro_f1": round(f1_score(test_y, test_pred, average="macro"), 4),
                "hamming_loss": round(test_hamming_loss, 4),
                "label_accuracy": round(1 - test_hamming_loss, 4),
                "full_label_accuracy": round(1 - full_hamming_loss, 4),
                "rule_ml_agreement_rate": round(accuracy_score(full_y, full_pred), 4),
            }
        ]
    )
    return evaluation, build_confusion_matrix(test_y, test_pred)


def main() -> None:
    records = pd.read_csv(RECORDS_PATH)
    semantic_frame = build_semantic_frame(records)
    evaluation, matrix = train_and_evaluate(semantic_frame)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(
        OUTPUT_DIR / "semantic_ml_evaluation.csv", index=False, encoding="utf-8-sig"
    )
    matrix.to_csv(
        OUTPUT_DIR / "semantic_ml_confusion_matrix.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(evaluation.to_string(index=False))
    print(matrix.to_string(index=False))


if __name__ == "__main__":
    main()
