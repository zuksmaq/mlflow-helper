"""
Example workflow with XGBoost.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from mlflow_helper import MLflowConfig, MLflowRepository


def xgboost_workflow():
    """Example workflow with XGBoost."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("XGBOOST EXAMPLE")
    print("=" * 60)

    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Configure MLflow
    config = MLflowConfig(
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="breast_cancer_classification",
        tags={"framework": "xgboost", "dataset": "breast_cancer"},
    )

    repo = MLflowRepository(config)

    # Training session
    with repo.training_session(run_name="xgboost_v1"):
        # Train model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }

        # Log parameters
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        # Create a simple plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(
            ["Accuracy", "Precision", "Recall"],
            [metrics["accuracy"], metrics["precision"], metrics["recall"]],
        )
        ax.set_ylabel("Score")
        ax.set_title("XGBoost Model Performance")
        ax.set_ylim(0, 1)

        # Log everything
        repo.log_training_results(
            metrics=metrics, params=params, figures={"performance_plot.png": fig}
        )

        # Save and register model
        model_uri = repo.save_model(
            model,
            artifact_path="xgboost_classifier",
            register=True,
            registry_name="breast_cancer_xgb_model",
        )

        print(f"Model saved to: {model_uri}")
        print(f"Metrics: {metrics}")

    plt.close()


if __name__ == "__main__":
    xgboost_workflow()
