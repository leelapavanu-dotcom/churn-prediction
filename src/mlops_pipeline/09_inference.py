
import os
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
import skops.io as sio
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_mlflow(config):
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or config["mlflow"].get("tracking_uri")
    )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def clean_dataframe(df, config):
    df = df.copy()

    target_column = config["data"]["target_column"]
    id_column = config["data"]["id_column"]

    if id_column in df.columns:
        df = df.drop(columns=[id_column])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    if target_column in df.columns:
        df = df.drop(columns=[target_column])

    return df


def load_production_model(model_name):
    client = MlflowClient()

    versions = client.get_latest_versions(
        model_name,
        stages=["Production"]
    )

    if not versions:
        raise RuntimeError(
            f"No Production model found for '{model_name}'."
        )

    production_version = versions[0]

    print(
        f"[INFO] Production Model Version: "
        f"{production_version.version}"
    )
    print(f"[INFO] Model source: {production_version.source}")

    source = production_version.source

    # MLflow 3 model artifact source, e.g. models:/m-xxxxxxxx
    if source.startswith("models:/m-"):
        model_id = source.split("/")[-1]

        model_path = (
            PROJECT_ROOT
            / "mlruns"
            / "1"
            / "models"
            / model_id
            / "artifacts"
            / "model.skops"
        )

    else:
        raise RuntimeError(
            "Production model source is not a supported local MLflow "
            f"model artifact: {source}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Local Production model artifact not found: {model_path}"
        )

    print(f"[INFO] Loading local model artifact: {model_path}")

    model = sio.load(
        str(model_path),
        trusted=["sklearn.tree._tree.Tree"]
    )

    print("[SUCCESS] Production model loaded.")
    return model


def run_inference_demo():
    print("[INFO] Starting ML Pipeline Production Inference...")

    config_path = PROJECT_ROOT / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)

    model_name = config["registry"]["model_name"]

    try:
        model = load_production_model(model_name)

    except Exception as exc:
        print(f"[ERROR] Could not load Production model: {exc}")
        return False

    preprocessor_path = resolve_path(
        config["paths"]["preprocessor"]
    )

    raw_data_path = resolve_path(
        config["data"]["raw_path"]
    )

    if not preprocessor_path.exists():
        print(f"[ERROR] Preprocessor not found: {preprocessor_path}")
        return False

    if not raw_data_path.exists():
        print(f"[ERROR] Raw dataset not found: {raw_data_path}")
        return False

    preprocessor = joblib.load(preprocessor_path)
    df = pd.read_csv(raw_data_path)

    sample = df.sample(n=5, random_state=7)

    X_sample = clean_dataframe(sample, config)
    X_sample = preprocessor.transform(X_sample)

    predictions = model.predict(X_sample)
    probabilities = model.predict_proba(X_sample)[:, 1]

    result = sample.copy()
    result["Predicted_Churn"] = predictions
    result["Churn_Probability"] = np.round(
        probabilities, 4
    )

    print("\n--- ML Pipeline Production Inference Results ---")
    print(
        result[
            ["Predicted_Churn", "Churn_Probability"]
        ].to_string(index=False)
    )
    print("-----------------------------------------------\n")

    output_path = resolve_path(
        config["paths"]["inference_output"]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(output_path, index=False)

    print(f"[SUCCESS] Predictions saved to {output_path}")

    return True


if __name__ == "__main__":
    passed = run_inference_demo()
    sys.exit(0 if passed else 1)
