"""
Model saving/loading with automatic type detection.
"""

import logging
from typing import Dict, List, Optional

import mlflow.pyfunc
from mlflow.pyfunc import PyFuncModel

from ..config import MLflowConfig
from ..interfaces import ModelProtocol
from ..savers import ModelSaver, PyFuncSaver, SklearnSaver, SparkSaver


class ModelPersister:
    """Handles model saving/loading with automatic type detection."""

    def __init__(self, config: MLflowConfig):
        self.config = config
        self.savers = self._initialize_savers()

    def _initialize_savers(self) -> List[ModelSaver]:
        """Initialize savers in order of specificity."""
        return [
            SklearnSaver(),
            SparkSaver(),
            PyFuncSaver(),
        ]

    def save(
        self,
        model: ModelProtocol,
        artifact_path: Optional[str] = None,
        artifacts: Optional[Dict] = None,
    ) -> str:
        """Save model with automatic type detection."""
        path = artifact_path or self.config.default_artifact_path

        for saver in self.savers:
            if saver.can_save(model):
                return saver.save(model, path, artifacts)

        raise TypeError(
            f"Unsupported model type: {type(model).__name__}. "
            f"Model must have a 'predict' method or inherit from sklearn/Spark base classes."
        )

    def load(self, model_uri: str) -> PyFuncModel:
        """Load model from URI or registry."""
        try:
            if not model_uri.startswith(("models:/", "runs:/", "/")):
                model_uri = f"models:/{model_uri}/latest"

            model = mlflow.pyfunc.load_model(model_uri)
            logging.info(f"Loaded model from {model_uri}")
            return model
        except Exception as e:
            logging.error(f"Failed to load model from {model_uri}: {e}")
            raise

    def register_model(self, run_id: str, artifact_path: str, name: str) -> str:
        """Register a model in the MLflow Model Registry."""
        model_uri = f"runs:/{run_id}/{artifact_path}"
        mv = mlflow.register_model(model_uri, name)
        logging.info(f"Registered model {name} version {mv.version}")
        return f"models:/{name}/{mv.version}"
