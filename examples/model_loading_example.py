"""
Example of loading and using saved models.
"""

import logging

import numpy as np

from mlflow_helper import MLflowConfig, MLflowRepository


def model_loading_workflow():
    """Example of loading and using saved models."""
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("MODEL LOADING EXAMPLE")
    print("=" * 60)

    config = MLflowConfig(
        tracking_uri="sqlite:///mlflow.db", experiment_name="iris_classification"
    )

    repo = MLflowRepository(config)

    try:
        # Load model from registry
        model = repo.load_model("iris_rf_model")

        # Use loaded model
        X_new = np.array([[5.1, 3.5, 1.4, 0.2]])
        prediction = model.predict(X_new)

        print(f"Loaded model prediction: {prediction}")

        # Get model metrics
        metrics, params = repo.get_model_metrics(model_name="iris_rf_model")
        print(f"Model metrics: {metrics}")
        print(f"Model params: {params}")

    except Exception as e:
        print(f"Could not load model: {e}")
        print("Run the sklearn example first to create a model.")


if __name__ == "__main__":
    model_loading_workflow()
