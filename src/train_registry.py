
import os
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def train_and_register_model():
    print("[INFO] --- Starting MLflow Run with Model Registry ---")

    # 1. Set up MLflow
    mlflow.set_experiment("Telco_Churn_Prediction")

    # 2. Load processed data
    try:
        X_train = np.load("data/processed/X_train_final.npy")
        X_test = np.load("data/processed/X_test_final.npy")
        y_train = np.load("data/processed/y_train.npy")
        y_test = np.load("data/processed/y_test.npy")

    except FileNotFoundError:
        print("[ERROR] Processed data not found. Run the Lab 5 pipeline first.")
        return

    # 3. Define model parameters
    params = {
        "n_estimators": 200,
        "max_depth": 15,
        "random_state": 42,
        "class_weight": "balanced"
    }

    # 4. Start MLflow run
    with mlflow.start_run(run_name="RandomForest_Registry_V2") as run:

        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForest")

        # 5. Train model
        print("[INFO] Training RandomForest model...")

        rf_model = RandomForestClassifier(**params)
        rf_model.fit(X_train, y_train)

        # 6. Evaluate model
        print("[INFO] Evaluating model...")

        y_pred = rf_model.predict(X_test)
        y_proba = rf_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

        mlflow.log_metrics(metrics)

        # 7. Log preprocessing pipeline
        print("[INFO] Attaching preprocessing pipeline to model artifacts...")

        mlflow.log_artifact(
            "models/preprocessor.pkl",
            artifact_path="preprocessing_pipeline"
        )

        # 8. Log and register model
        print("[INFO] Pushing model to MLflow Registry...")

        mlflow.sklearn.log_model(
            sk_model=rf_model,
            name="random_forest_model",
            registered_model_name="Telco_Churn_Production_Model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        print(
            f"[SUCCESS] Metrics logged: "
            f"F1 = {metrics['f1_score']:.4f} | "
            f"ROC-AUC = {metrics['roc_auc']:.4f}"
        )

        print(
            "[SUCCESS] Model successfully registered under name: "
            "'Telco_Churn_Production_Model'"
        )


if __name__ == "__main__":
    train_and_register_model()
