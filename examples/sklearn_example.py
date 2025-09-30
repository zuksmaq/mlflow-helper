"""
Example workflow with scikit-learn.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from mlflow_helper import MLflowConfig, MLflowRepository


def sklearn_workflow():
    """Example workflow with scikit-learn."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("SKLEARN EXAMPLE")
    print("=" * 60)

    # Load data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Configure MLflow
    config = MLflowConfig(
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="iris_classification",
        tags={"framework": "sklearn", "dataset": "iris"},
    )

    repo = MLflowRepository(config)

    # Training session
    with repo.training_session(run_name="random_forest_v1"):
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="macro"),
            "recall": recall_score(y_test, y_pred, average="macro"),
        }

        # Log parameters
        params = {
            "n_estimators": 100,
            "random_state": 42,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        # Create a simple plot
        fig, ax = plt.subplots()
        ax.bar(
            ["Accuracy", "Precision", "Recall"],
            [metrics["accuracy"], metrics["precision"], metrics["recall"]],
        )
        ax.set_ylabel("Score")
        ax.set_title("Model Performance")

        # Log everything
        repo.log_training_results(
            metrics=metrics, params=params, figures={"performance_plot.png": fig}
        )

        # Save and register model
        model_uri = repo.save_model(
            model,
            artifact_path="random_forest",
            register=True,
            registry_name="iris_rf_model",
        )

        print(f"Model saved to: {model_uri}")
        print(f"Metrics: {metrics}")

    plt.close()


if __name__ == "__main__":
    sklearn_workflow()
