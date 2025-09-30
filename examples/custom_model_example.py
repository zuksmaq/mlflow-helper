"""
Example with a custom model class.
"""

import logging

import numpy as np

from mlflow_helper import MLflowConfig, MLflowRepository


class CustomPredictor:
    """A simple custom model with predict method."""

    def __init__(self, multiplier=2.0):
        self.multiplier = multiplier
        self.fitted = False

    def fit(self, X, y):
        """Dummy fit method."""
        self.mean_ = np.mean(y)
        self.fitted = True
        return self

    def predict(self, X):
        """Simple prediction: multiply input by multiplier."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return np.array([self.mean_ * self.multiplier] * len(X))


def custom_model_workflow():
    """Example with a custom model class."""
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("CUSTOM MODEL EXAMPLE")
    print("=" * 60)

    # Configure MLflow
    config = MLflowConfig(
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="custom_models",
        tags={"framework": "custom", "type": "experimental"},
    )

    repo = MLflowRepository(config)

    with repo.training_session(run_name="custom_predictor_v1"):
        # Create and train model
        model = CustomPredictor(multiplier=3.0)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)

        model.fit(X_train, y_train)

        # Make predictions
        X_test = np.random.randn(20, 5)
        predictions = model.predict(X_test)

        # Log metrics
        repo.log_training_results(
            metrics={"mean_prediction": float(np.mean(predictions))},
            params={"multiplier": 3.0},
        )

        # Save model (will use PyFunc wrapper automatically)
        model_uri = repo.save_model(model, artifact_path="custom_model")

        print(f"Custom model saved to: {model_uri}")
        print(f"Mean prediction: {np.mean(predictions):.3f}")


if __name__ == "__main__":
    custom_model_workflow()
