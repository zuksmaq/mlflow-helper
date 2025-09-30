"""
Retrieval of metrics and parameters from past runs.
"""

import logging
from typing import Dict, List, Optional, Tuple

import mlflow.tracking

from ..config import MLflowConfig


class MetricsRetriever:
    """Handles retrieval of metrics and parameters from past runs."""

    def __init__(self, config: MLflowConfig):
        self.config = config
        self.client = mlflow.tracking.MlflowClient()

    def get_run_data(self, run_id: str) -> Tuple[Dict[str, float], Dict[str, str]]:
        """Retrieve metrics and parameters for a specific run."""
        try:
            run = self.client.get_run(run_id)
            metrics = dict(run.data.metrics)
            params = dict(run.data.params)
            return metrics, params
        except (mlflow.exceptions.MlflowException, KeyError) as e:
            logging.error("Failed to retrieve run data for %s: %s", run_id, e)
            return {}, {}

    def get_latest_model_run(self, model_name: str) -> Optional[str]:
        """Get the run ID of the latest model version."""
        try:
            versions = self.client.get_latest_versions(model_name, stages=["None"])
            if versions:
                return versions[0].run_id
            return None
        except mlflow.exceptions.MlflowException as e:
            logging.error("Failed to get latest model run for %s: %s", model_name, e)
            return None

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Dict]:
        """Compare metrics across multiple runs."""
        comparison = {}
        for run_id in run_ids:
            metrics, params = self.get_run_data(run_id)
            comparison[run_id] = {"metrics": metrics, "params": params}
        return comparison
