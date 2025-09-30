"""
Saver for XGBoost models.
"""

import logging
from typing import Dict, Optional

import mlflow
import mlflow.xgboost

from .base import ModelSaver


class XGBoostSaver(ModelSaver):
    """Saver for XGBoost models."""

    def can_save(self, model) -> bool:
        # Check for both XGBoost model types:
        # - xgboost.XGBModel (sklearn API wrapper)
        # - xgboost.Booster (native API)
        return self._check_mro(model, "xgboost.sklearn.XGBModel") or self._check_mro(
            model, "xgboost.core.Booster"
        )

    def save(self, model, path: str, artifacts: Optional[Dict] = None) -> str:

        mlflow.xgboost.log_model(xgb_model=model, artifact_path=path)

        # Log artifacts separately if provided
        if artifacts:
            for name, artifact in artifacts.items():
                mlflow.log_artifact(artifact, f"{path}/artifacts")

        logging.info(f"Saved XGBoost model to {path}")
        return path
