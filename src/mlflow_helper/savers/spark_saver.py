"""
Saver for PySpark ML models.
"""

import logging
from typing import Dict, Optional

import mlflow
import mlflow.spark

from .base import ModelSaver


class SparkSaver(ModelSaver):
    """Saver for PySpark ML models."""

    def can_save(self, model) -> bool:
        return (
            self._check_mro(model, "pyspark.ml.base.Estimator")
            or self._check_mro(model, "pyspark.ml.base.Model")
            or self._check_mro(model, "pyspark.ml.pipeline.PipelineModel")
        )

    def save(self, model, path: str, artifacts: Optional[Dict] = None) -> str:
        mlflow.spark.log_model(spark_model=model, artifact_path=path)

        if artifacts:
            for name, artifact in artifacts.items():
                mlflow.log_artifact(artifact, f"{path}/artifacts")

        logging.info(f"Saved Spark model to {path}")
        return path
