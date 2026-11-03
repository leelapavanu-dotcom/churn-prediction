import json
import shutil
import tempfile
from pathlib import Path

import joblib
import skops.io as sio
import yaml
from mlflow import MlflowClient

MODEL_NAME = "Telco_Churn_Production_Model"


def load_config():
    project_root = Path(__file__).resolve().parents[2]

    with open(project_root / "config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return project_root, config


def get_production_model(client, model_name):
    versions = client.search_model_versions(f"name='{model_name}'")

    production_versions = [
        version for version in versions
        if version.current_stage == "Production"
    ]

    if not production_versions:
        raise RuntimeError(f"No Production version found for '{model_name}'.")

    if len(production_versions) > 1:
        raise RuntimeError("Multiple Production versions found.")

    return production_versions[0]


def prepare_deployment():
    project_root, config = load_config()
    deployment_dir = project_root / "deployment"

    client = MlflowClient()
    model_version = get_production_model(client, MODEL_NAME)

    model_name = model_version.name
    version = model_version.version
    run_id = model_version.run_id
    model_source = model_version.source

    print("[INFO] Production model found")
    print(f"[INFO] Model name : {model_name}")
    print(f"[INFO] Version    : {version}")
    print(f"[INFO] Run ID     : {run_id}")
    print(f"[INFO] Source     : {model_source}")

    temp_dir = Path(tempfile.mkdtemp(
        prefix="deployment_staging_",
        dir=str(project_root),
    ))

    backup_dir = None

    try:
        # Load the Production model from local MLflow artifacts
        print("[INFO] Loading Production model...")

        model_id = model_source.split("/")[-1]

        model_artifact_path = (
            project_root
            / "mlruns"
            / "1"
            / "models"
            / model_id
            / "artifacts"
            / "model.skops"
        )

        if not model_artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {model_artifact_path}"
            )

        model = sio.load(
            str(model_artifact_path),
            trusted=["sklearn.tree._tree.Tree"],
        )

        model_path = temp_dir / "model.pkl"
        joblib.dump(model, model_path)

        print(f"[SUCCESS] Model saved to {model_path}")

        # Download matching preprocessor
        print("[INFO] Downloading preprocessing pipeline...")

        final_preprocessor_path = temp_dir / "preprocessor.pkl"

        try:
            preprocessor_path = client.download_artifacts(
                run_id,
                "preprocessing_pipeline/preprocessor.pkl",
                dst_path=str(temp_dir),
            )

            downloaded_preprocessor = Path(preprocessor_path)

            if downloaded_preprocessor.resolve() != final_preprocessor_path.resolve():
                shutil.copy2(
                    downloaded_preprocessor,
                    final_preprocessor_path,
                )

        except Exception as error:
            local_preprocessor = project_root / "models" / "preprocessor.pkl"

            if not local_preprocessor.exists():
                raise RuntimeError(
                    "Preprocessor unavailable from MLflow and local models folder."
                ) from error

            print("[WARNING] MLflow preprocessor unavailable.")
            print("[WARNING] Using local models/preprocessor.pkl.")

            shutil.copy2(
                local_preprocessor,
                final_preprocessor_path,
            )

        print(f"[SUCCESS] Preprocessor saved to {final_preprocessor_path}")

        # Verify deployment artifacts
        print("[INFO] Verifying deployment artifacts...")

        loaded_model = joblib.load(model_path)
        loaded_preprocessor = joblib.load(final_preprocessor_path)

        if loaded_model is None or loaded_preprocessor is None:
            raise RuntimeError("Deployment artifact verification failed.")

        metadata = {
            "model_name": model_name,
            "model_version": str(version),
            "model_stage": model_version.current_stage,
            "run_id": run_id,
            "model_source": model_source,
            "model_id": model_id,
            "preprocessor_artifact": "preprocessor.pkl",
            "source": "MLflow",
        }

        metadata_path = temp_dir / "model_metadata.json"

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

        print(f"[SUCCESS] Metadata saved to {metadata_path}")

        # Preserve previous deployment
        if deployment_dir.exists():
            index = 1

            while True:
                candidate = project_root / f"deployment_backup_{index}"

                if not candidate.exists():
                    backup_dir = candidate
                    break

                index += 1

            print(f"[INFO] Preserving current deployment in {backup_dir}")
            deployment_dir.rename(backup_dir)

        try:
            temp_dir.rename(deployment_dir)

        except Exception:
            if backup_dir and backup_dir.exists() and not deployment_dir.exists():
                backup_dir.rename(deployment_dir)

            raise

        print("\n[INFO] Deployment bundle:")
        print("deployment/")
        print("├── model.pkl")
        print("├── preprocessor.pkl")
        print("└── model_metadata.json")

        if backup_dir:
            print(f"[INFO] Previous deployment preserved in {backup_dir}")

        print("\n[SUCCESS] Deployment preparation completed.")

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    prepare_deployment()