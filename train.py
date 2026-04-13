from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"
MODEL_FILE = PROJECT_DIR / "network_multi_model.joblib"
SUMMARY_FILE = PROJECT_DIR / "training_summary.json"


def load_dataset(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}\n"
            "Place the CSV files inside the project's data folder."
        )

    dataframe = pd.read_csv(file_path)
    if dataframe.empty:
        raise ValueError(f"Dataset is empty: {file_path}")
    return dataframe


def print_dataset_overview(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\n=== DATASET COLUMNS (TRAIN) ===")
    print(list(train_df.columns))

    print("\n=== DATASET COLUMNS (TEST) ===")
    print(list(test_df.columns))

    print("\n=== DATASET SHAPES ===")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")


def clean_text_series(series: pd.Series) -> pd.Series:
    cleaned = series.copy()
    cleaned = cleaned.replace(["", " ", "-", "nan", "None"], np.nan)
    cleaned = cleaned.fillna("Unknown")
    return cleaned.astype(str).str.strip()


def detect_attack_target(train_df: pd.DataFrame) -> str:
    preferred_targets = ["attack_cat", "label"]
    for column in preferred_targets:
        if column in train_df.columns and train_df[column].nunique(dropna=True) > 1:
            return column
    raise ValueError("Could not detect a valid target column from 'attack_cat' or 'label'.")


def prepare_feature_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    leakage_columns = {"id", "label", "attack_cat"}
    common_columns = [column for column in train_df.columns if column in test_df.columns]
    feature_columns = [column for column in common_columns if column not in leakage_columns]
    if not feature_columns:
        raise ValueError("No usable feature columns were found after excluding leakage columns.")
    return feature_columns


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_columns = [column for column in X.columns if column not in categorical_columns]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )

    return preprocessor, numeric_columns, categorical_columns


def build_feature_defaults(X: pd.DataFrame, categorical_columns: list[str]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    categorical_set = set(categorical_columns)

    for column in X.columns:
        series = X[column]
        if column in categorical_set:
            non_null = series.dropna()
            defaults[column] = "" if non_null.empty else str(non_null.mode(dropna=True).iloc[0])
        else:
            numeric_series = pd.to_numeric(series, errors="coerce")
            non_null = numeric_series.dropna()
            defaults[column] = 0.0 if non_null.empty else float(non_null.median())

    return defaults


def build_categorical_options(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
) -> dict[str, list[str]]:
    options: dict[str, list[str]] = {}
    for column in categorical_columns:
        combined = pd.concat([X_train[column], X_test[column]], axis=0)
        cleaned = combined.dropna().astype(str).str.strip()
        unique_values = sorted(value for value in cleaned.unique().tolist() if value != "")
        options[column] = unique_values if unique_values else [""]
    return options


def make_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: make_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return None
    return value


def compute_train_stats(series: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    log_values = np.log1p(np.abs(numeric.to_numpy()))
    median = float(np.median(log_values))
    spread = float(np.std(log_values))
    return {"median": median, "spread": spread if spread > 0 else 1.0}


def combine_standardized_scores(
    dataframe: pd.DataFrame,
    columns: list[str],
    stats_map: dict[str, dict[str, float]],
) -> pd.Series:
    if not columns:
        raise ValueError("No columns were supplied to compute a derived score.")

    score = np.zeros(len(dataframe), dtype=float)
    for column in columns:
        numeric = pd.to_numeric(dataframe[column], errors="coerce").fillna(0.0).astype(float).to_numpy()
        log_values = np.log1p(np.abs(numeric))
        median = stats_map[column]["median"]
        spread = stats_map[column]["spread"]
        score += (log_values - median) / spread

    score = score / len(columns)
    return pd.Series(score, index=dataframe.index, dtype=float)


def derive_multiclass_labels(
    score: pd.Series,
    low_threshold: float,
    high_threshold: float,
    labels: tuple[str, str, str],
) -> pd.Series:
    result = pd.Series(labels[1], index=score.index, dtype=object)
    result.loc[score <= low_threshold] = labels[0]
    result.loc[score > high_threshold] = labels[2]
    return result.astype(str)


def derive_binary_labels(
    score: pd.Series,
    threshold: float,
    positive_label: str,
    negative_label: str,
) -> pd.Series:
    result = pd.Series(negative_label, index=score.index, dtype=object)
    result.loc[score > threshold] = positive_label
    return result.astype(str)


def get_available_columns(columns: list[str], dataframe: pd.DataFrame) -> list[str]:
    return [column for column in columns if column in dataframe.columns]


def build_derived_targets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    traffic_columns = get_available_columns(
        ["rate", "sload", "dload", "spkts", "dpkts", "sbytes", "dbytes"],
        train_df,
    )
    fault_columns = get_available_columns(
        ["sloss", "dloss", "sjit", "djit", "tcprtt", "synack", "ackdat"],
        train_df,
    )
    connectivity_columns = get_available_columns(
        ["sloss", "dloss", "sjit", "djit", "sinpkt", "dinpkt", "tcprtt"],
        train_df,
    )

    for group_name, columns in {
        "traffic_condition": traffic_columns,
        "fault_detection": fault_columns,
        "connectivity_quality": connectivity_columns,
    }.items():
        if not columns:
            raise ValueError(f"Unable to derive {group_name} labels because no source columns were found.")

    stats_map = {
        column: compute_train_stats(train_df[column])
        for column in sorted(set(traffic_columns + fault_columns + connectivity_columns))
    }

    train_traffic_score = combine_standardized_scores(train_df, traffic_columns, stats_map)
    test_traffic_score = combine_standardized_scores(test_df, traffic_columns, stats_map)
    traffic_low = float(train_traffic_score.quantile(0.33))
    traffic_high = float(train_traffic_score.quantile(0.66))

    train_fault_score = combine_standardized_scores(train_df, fault_columns, stats_map)
    test_fault_score = combine_standardized_scores(test_df, fault_columns, stats_map)
    fault_threshold = float(train_fault_score.quantile(0.65))

    train_connectivity_score = combine_standardized_scores(train_df, connectivity_columns, stats_map)
    test_connectivity_score = combine_standardized_scores(test_df, connectivity_columns, stats_map)
    connectivity_threshold = float(train_connectivity_score.quantile(0.55))

    abnormal_source = "label" if "label" in train_df.columns else detect_attack_target(train_df)
    if abnormal_source == "label":
        abnormal_train = train_df["label"].apply(lambda value: "Abnormal" if int(value) == 1 else "Normal").astype(str)
        abnormal_test = test_df["label"].apply(lambda value: "Abnormal" if int(value) == 1 else "Normal").astype(str)
    else:
        abnormal_train = clean_text_series(train_df[abnormal_source]).apply(
            lambda value: "Normal" if value.lower() == "normal" else "Abnormal"
        )
        abnormal_test = clean_text_series(test_df[abnormal_source]).apply(
            lambda value: "Normal" if value.lower() == "normal" else "Abnormal"
        )

    derived_targets = {
        "traffic_condition": {
            "display_name": "Traffic Condition",
            "description": "Classifies network load as Low, Medium, or Heavy using traffic-rate and packet-volume features.",
            "target_type": "derived_multiclass",
            "train_labels": derive_multiclass_labels(train_traffic_score, traffic_low, traffic_high, ("Low", "Medium", "Heavy")),
            "test_labels": derive_multiclass_labels(test_traffic_score, traffic_low, traffic_high, ("Low", "Medium", "Heavy")),
            "source_columns": traffic_columns,
            "derivation": {
                "score_formula": "Average standardized log-scaled traffic features",
                "low_threshold": traffic_low,
                "high_threshold": traffic_high,
            },
        },
        "fault_detection": {
            "display_name": "Fault Detection",
            "description": "Detects Normal or Faulty network states using packet loss, jitter, and TCP timing indicators.",
            "target_type": "derived_binary",
            "train_labels": derive_binary_labels(train_fault_score, fault_threshold, "Faulty", "Normal"),
            "test_labels": derive_binary_labels(test_fault_score, fault_threshold, "Faulty", "Normal"),
            "source_columns": fault_columns,
            "derivation": {
                "score_formula": "Average standardized log-scaled fault indicators",
                "threshold": fault_threshold,
            },
        },
        "connectivity_quality": {
            "display_name": "Connectivity Quality",
            "description": "Labels links as Good or Poor based on timing, jitter, and loss-related connectivity indicators.",
            "target_type": "derived_binary",
            "train_labels": derive_binary_labels(train_connectivity_score, connectivity_threshold, "Poor", "Good"),
            "test_labels": derive_binary_labels(test_connectivity_score, connectivity_threshold, "Poor", "Good"),
            "source_columns": connectivity_columns,
            "derivation": {
                "score_formula": "Average standardized log-scaled connectivity impairment indicators",
                "threshold": connectivity_threshold,
            },
        },
        "abnormal_behavior": {
            "display_name": "Abnormal Behavior",
            "description": "Detects whether the observed traffic is Normal or Abnormal using the original UNSW-NB15 attack labeling.",
            "target_type": "original_binary",
            "train_labels": abnormal_train,
            "test_labels": abnormal_test,
            "source_columns": [abnormal_source],
            "derivation": {
                "source_target": abnormal_source,
            },
        },
    }

    derivation_context = {
        "score_stats": stats_map,
        "abnormal_source": abnormal_source,
    }
    return derived_targets, derivation_context


def build_model_pipeline(preprocessor: ColumnTransformer, num_classes: int) -> Pipeline:
    min_leaf = 2 if num_classes > 2 else 1
    max_depth = 18 if num_classes > 2 else 14

    classifier = RandomForestClassifier(
        n_estimators=40,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        random_state=42,
        n_jobs=1,
        class_weight="balanced_subsample",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("selector", SelectPercentile(score_func=f_classif, percentile=70)),
            ("classifier", classifier),
        ]
    )


def evaluate_predictions(
    true_labels: pd.Series,
    predicted_labels: np.ndarray,
    class_labels: list[str],
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "precision_macro": float(precision_score(true_labels, predicted_labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(true_labels, predicted_labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(true_labels, predicted_labels, average="macro", zero_division=0)),
        "classification_report": classification_report(true_labels, predicted_labels, zero_division=0),
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels, labels=class_labels),
        "class_labels": class_labels,
    }


def train_module(
    module_key: str,
    module_config: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    preprocessor: ColumnTransformer,
) -> dict[str, Any]:
    print(f"\n=== TRAINING MODULE: {module_config['display_name']} ({module_key}) ===")
    print(f"Source columns used to derive labels: {module_config['source_columns']}")

    y_train = module_config["train_labels"].astype(str)
    y_test = module_config["test_labels"].astype(str)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    unseen_test_classes = sorted(set(y_test.unique()) - set(label_encoder.classes_))
    if unseen_test_classes:
        raise ValueError(f"Module '{module_key}' has test classes unseen during training: {unseen_test_classes}")

    pipeline = build_model_pipeline(preprocessor, num_classes=len(label_encoder.classes_))
    pipeline.fit(X_train, y_train_encoded)

    predicted_encoded = pipeline.predict(X_test)
    predicted_labels = label_encoder.inverse_transform(predicted_encoded)
    metrics = evaluate_predictions(y_test, predicted_labels, label_encoder.classes_.tolist())

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision_macro']:.4f}")
    print(f"Recall   : {metrics['recall_macro']:.4f}")
    print(f"F1-Score : {metrics['f1_macro']:.4f}")
    print(metrics["classification_report"])

    confusion_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=metrics["class_labels"],
        columns=metrics["class_labels"],
    )
    print(confusion_df.to_string())

    return {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "metrics": metrics,
        "display_name": module_config["display_name"],
        "description": module_config["description"],
        "target_type": module_config["target_type"],
        "source_columns": module_config["source_columns"],
        "derivation": module_config["derivation"],
    }


def main() -> None:
    print("Loading UNSW-NB15 datasets...")
    train_df = load_dataset(TRAIN_FILE)
    test_df = load_dataset(TEST_FILE)
    print_dataset_overview(train_df, test_df)

    detected_attack_target = detect_attack_target(train_df)
    print(f"\nDetected original UNSW target column: {detected_attack_target}")

    feature_columns = prepare_feature_columns(train_df, test_df)
    print(f"Number of feature columns used for all modules: {len(feature_columns)}")

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()

    preprocessor, numeric_columns, categorical_columns = build_preprocessor(X_train)
    feature_defaults = build_feature_defaults(X_train, categorical_columns)
    categorical_options = build_categorical_options(X_train, X_test, categorical_columns)
    derived_targets, derivation_context = build_derived_targets(train_df, test_df)

    trained_modules: dict[str, dict[str, Any]] = {}
    summary_modules: dict[str, Any] = {}

    for module_key, module_config in derived_targets.items():
        trained_module = train_module(module_key, module_config, X_train, X_test, preprocessor)
        trained_modules[module_key] = trained_module
        summary_modules[module_key] = {
            "display_name": trained_module["display_name"],
            "description": trained_module["description"],
            "target_type": trained_module["target_type"],
            "source_columns": trained_module["source_columns"],
            "derivation": make_serializable(trained_module["derivation"]),
            "metrics": {
                metric_name: make_serializable(metric_value)
                for metric_name, metric_value in trained_module["metrics"].items()
                if metric_name != "classification_report"
            },
        }

    artifact = {
        "artifact_version": 2,
        "project_name": "Intelligent Network Monitoring System",
        "dataset_name": "UNSW-NB15",
        "detected_attack_target": detected_attack_target,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "feature_defaults": feature_defaults,
        "categorical_options": categorical_options,
        "modules": trained_modules,
        "dataset_info": {
            "train_shape": list(train_df.shape),
            "test_shape": list(test_df.shape),
        },
        "derivation_context": make_serializable(derivation_context),
    }

    joblib.dump(artifact, MODEL_FILE)
    print(f"\nSaved trained model artifact to: {MODEL_FILE}")

    summary_payload = {
        "project_name": artifact["project_name"],
        "dataset_name": artifact["dataset_name"],
        "detected_attack_target": detected_attack_target,
        "feature_count": len(feature_columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "modules": summary_modules,
    }
    SUMMARY_FILE.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"Saved training summary to: {SUMMARY_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        raise
