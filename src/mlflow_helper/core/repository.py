"""
High-level MLflow repository implementation.
"""

from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

from mlflow.pyfunc import PyFuncModel

from ..config import MLflowConfig
from ..interfaces import MLflowRepositoryInterface
from .persister import ModelPersister
from .retriever import MetricsRetriever
from .tracker import MLflowTracker


class MLflowRepository(MLflowRepositoryInterface):
    """High-level interface combining tracking and persistence."""

    def __init__(self, config: MLflowConfig):
        self.config = config
        self.tracker = MLflowTracker(config)
        self.persister = ModelPersister(config)
        self.retriever = MetricsRetriever(config)
        self.current_run_id = None

    @contextmanager
    def training_session(
        self, run_name: Optional[str] = None, tags: Optional[Dict] = None
    ):
        """Complete training session with automatic tracking."""
        with self.tracker.run(run_name=run_name, tags=tags) as run:
            self.current_run_id = run.info.run_id
            yield self
            self.current_run_id = None

    def save_model(
        self,
        model,
        artifact_path: Optional[str] = None,
        register: bool = False,
        registry_name: Optional[str] = None,
    ) -> str:
        """Save model and optionally register it."""
        model_uri = self.persister.save(model, artifact_path)

        if register and self.current_run_id:
            registry_name = registry_name or self.config.experiment_name
            reg_uri = self.persister.register_model(
                self.current_run_id,
                artifact_path or self.config.default_artifact_path,
                registry_name,
            )
            return reg_uri

        return model_uri

    def load_model(self, model_uri: str) -> PyFuncModel:
        """Load a model from URI or registry."""
        return self.persister.load(model_uri)

    def log_training_results(
        self,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, str]] = None,
        figures: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log all training results at once."""
        if metrics:
            self.tracker.log_metrics(metrics)
        if params:
            self.tracker.log_params(params)
        if artifacts:
            self.tracker.log_artifacts(artifacts)
        if figures:
            for name, fig in figures.items():
                self.tracker.log_figure(fig, name)

    def get_model_metrics(
        self, model_name: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[Dict, Dict]:
        """Get metrics for a model by name or run ID."""
        if not run_id and model_name:
            run_id = self.retriever.get_latest_model_run(model_name)

        if run_id:
            return self.retriever.get_run_data(run_id)

        return {}, {}
