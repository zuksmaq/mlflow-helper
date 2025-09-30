"""
Saver for scikit-learn models.
"""

import logging
from typing import Dict, Optional

import mlflow
import mlflow.sklearn

from .base import ModelSaver


class SklearnSaver(ModelSaver):
    """Saver for scikit-learn models."""

    def can_save(self, model) -> bool:
        return self._check_mro(model, "sklearn.base.BaseEstimator")

    def save(self, model, path: str, artifacts: Optional[Dict] = None) -> str:

        mlflow.sklearn.log_model(sk_model=model, artifact_path=path)

        # Log artifacts separately if provided
        if artifacts:
            for _, artifact in artifacts.items():
                mlflow.log_artifact(artifact, f"{path}/artifacts")

        logging.info("Saved sklearn model to %s", path)
        return path
