"""
MLflow experiment tracking and metrics logging.
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

import mlflow

from ..config import MLflowConfig


class MLflowTracker:
    """Handles experiment tracking and metrics logging."""

    def __init__(self, config: MLflowConfig):
        self.config = config
        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Initialize MLflow with the provided configuration."""
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)

        if mlflow.active_run():
            logging.warning("Active run detected, ending it before setup")
            mlflow.end_run()

    @contextmanager
    def run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None):
        """Context manager for MLflow runs."""
        combined_tags = {**self.config.tags, **(tags or {})}

        run = mlflow.start_run(run_name=run_name, tags=combined_tags)
        logging.info("Started MLflow run: %s", run.info.run_id)

        try:
            yield run
        except Exception as e:
            logging.error("Error during MLflow run: %s", e)
            mlflow.set_tag("error", str(e))
            raise
        finally:
            mlflow.end_run()
            logging.info("Ended MLflow run: %s", run.info.run_id)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics with validation."""
        if not metrics:
            return

        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                logging.warning("Skipping non-numeric metric %s: %s", key, value)
                continue
            mlflow.log_metric(key, value, step=step)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        if not params:
            return

        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_artifacts(
        self, artifacts: Dict[str, str], artifact_path: Optional[str] = None
    ) -> None:
        """Log artifacts (files) to MLflow."""
        for name, file_path in artifacts.items():
            mlflow.log_artifact(file_path, artifact_path=artifact_path)
            logging.info("Logged artifact %s from %s", name, file_path)

    def log_figure(self, figure, name: str) -> None:
        """Log a matplotlib figure."""
        mlflow.log_figure(figure, name)
        logging.info("Logged figure: %s", name)
