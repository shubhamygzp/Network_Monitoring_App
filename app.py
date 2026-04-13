from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_DIR / "network_multi_model.joblib"


def load_artifact(model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run train.py first to generate it."
        )

    artifact = joblib.load(model_path)
    required_keys = {
        "artifact_version",
        "project_name",
        "dataset_name",
        "feature_columns",
        "numeric_columns",
        "categorical_columns",
        "feature_defaults",
        "categorical_options",
        "modules",
    }
    missing_keys = required_keys - set(artifact.keys())
    if missing_keys:
        raise ValueError(f"Invalid model artifact. Missing keys: {sorted(missing_keys)}")
    if not artifact["modules"]:
        raise ValueError("The model artifact does not contain any trained modules.")
    return artifact


def build_manual_input_form(artifact: dict[str, Any]) -> pd.DataFrame:
    feature_columns = artifact["feature_columns"]
    numeric_columns = set(artifact["numeric_columns"])
    categorical_columns = set(artifact["categorical_columns"])
    feature_defaults = artifact["feature_defaults"]
    categorical_options = artifact["categorical_options"]

    st.subheader("Manual Network Input")
    st.caption("Enter one record and predict traffic load, fault status, connectivity quality, and abnormal behavior.")

    input_data: dict[str, Any] = {}
    with st.form("manual_prediction_form"):
        for column in feature_columns:
            default_value = feature_defaults.get(column)

            if column in categorical_columns:
                options = categorical_options.get(column, [""])
                default_text = "" if default_value is None else str(default_value)
                if default_text in options:
                    default_index = options.index(default_text)
                else:
                    options = [default_text] + options
                    default_index = 0
                input_data[column] = st.selectbox(column, options=options, index=default_index)
            elif column in numeric_columns:
                input_data[column] = st.number_input(
                    column,
                    value=float(default_value) if default_value is not None else 0.0,
                    format="%.6f",
                )
            else:
                input_data[column] = st.text_input(column, value="" if default_value is None else str(default_value))

        submitted = st.form_submit_button("Run Network Analysis")

    return pd.DataFrame([input_data]) if submitted else pd.DataFrame()


def predict_module(input_df: pd.DataFrame, module_artifact: dict[str, Any]) -> tuple[pd.Series, pd.Series | None]:
    pipeline = module_artifact["pipeline"]
    label_encoder = module_artifact["label_encoder"]
    predicted_encoded = pipeline.predict(input_df)
    predicted_labels = pd.Series(label_encoder.inverse_transform(predicted_encoded), index=input_df.index)

    confidence_scores: pd.Series | None = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(input_df)
        confidence_scores = pd.Series(probabilities.max(axis=1), index=input_df.index)

    return predicted_labels, confidence_scores


def predict_dataframe(input_df: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    feature_columns = artifact["feature_columns"]
    missing_columns = [column for column in feature_columns if column not in input_df.columns]
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")

    prepared_df = input_df[feature_columns].copy()
    result_df = input_df.copy()

    for module_key, module_artifact in artifact["modules"].items():
        predictions, confidence = predict_module(prepared_df, module_artifact)
        result_df[f"{module_key}_prediction"] = predictions
        if confidence is not None:
            result_df[f"{module_key}_confidence"] = confidence

    return result_df


def render_overview() -> None:
    st.title("Intelligent Network Monitoring System")
    st.write(
        "This dashboard uses machine learning on UNSW-NB15 traffic data to monitor "
        "traffic condition, fault status, connectivity quality, and abnormal behavior."
    )
    st.subheader("Project Scope")
    st.write(
        "UNSW-NB15 does not provide native labels for traffic load, fault status, or connectivity quality, "
        "so those monitoring modules are derived from relevant network metrics. The abnormal-behavior module "
        "uses the dataset's original attack labeling."
    )


def render_module_metrics(artifact: dict[str, Any]) -> None:
    st.subheader("Module Performance Summary")
    rows = []
    for module_key, module_artifact in artifact["modules"].items():
        metrics = module_artifact["metrics"]
        rows.append(
            {
                "Module": module_artifact["display_name"],
                "Key": module_key,
                "Accuracy": round(metrics["accuracy"], 4),
                "Precision": round(metrics["precision_macro"], 4),
                "Recall": round(metrics["recall_macro"], 4),
                "F1-Score": round(metrics["f1_macro"], 4),
                "Classes": ", ".join(metrics["class_labels"]),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_manual_results(result_df: pd.DataFrame, artifact: dict[str, Any]) -> None:
    st.subheader("Prediction Results")
    columns = st.columns(len(artifact["modules"]))
    for index, (module_key, module_artifact) in enumerate(artifact["modules"].items()):
        prediction_value = result_df.loc[result_df.index[0], f"{module_key}_prediction"]
        confidence_column = f"{module_key}_confidence"
        columns[index].metric(module_artifact["display_name"], str(prediction_value))
        if confidence_column in result_df.columns:
            columns[index].caption(f"Confidence: {result_df.loc[result_df.index[0], confidence_column]:.4f}")
    st.dataframe(result_df, use_container_width=True)


def render_module_details(artifact: dict[str, Any]) -> None:
    st.subheader("Module Details")
    for module_key, module_artifact in artifact["modules"].items():
        with st.expander(f"{module_artifact['display_name']} ({module_key})", expanded=False):
            st.write(module_artifact["description"])
            st.write(f"Derived from / based on columns: {', '.join(module_artifact['source_columns'])}")
            st.write("Classification report:")
            st.code(module_artifact["metrics"]["classification_report"])


def render_csv_prediction_section(artifact: dict[str, Any]) -> None:
    st.subheader("Batch Prediction")
    st.caption("Upload a CSV file that contains the same feature columns used in training.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        return

    try:
        input_df = pd.read_csv(uploaded_file)
        result_df = predict_dataframe(input_df, artifact)
        st.success("Batch prediction completed successfully.")
        st.dataframe(result_df.head(50), use_container_width=True)
        st.download_button(
            label="Download prediction results",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="network_monitoring_predictions.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"Batch prediction failed: {exc}")


def main() -> None:
    st.set_page_config(page_title="Intelligent Network Monitoring System", layout="wide")

    try:
        artifact = load_artifact(MODEL_FILE)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    render_overview()
    render_module_metrics(artifact)

    manual_tab, batch_tab, detail_tab = st.tabs(["Manual Prediction", "Batch Prediction", "Model Details"])

    with manual_tab:
        manual_df = build_manual_input_form(artifact)
        if not manual_df.empty:
            try:
                result_df = predict_dataframe(manual_df, artifact)
                render_manual_results(result_df, artifact)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    with batch_tab:
        render_csv_prediction_section(artifact)

    with detail_tab:
        render_module_details(artifact)


if __name__ == "__main__":
    main()
