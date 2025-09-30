"""
Interfaces and protocols for MLflow helper package.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

from mlflow.pyfunc import PyFuncModel


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for models that can be saved."""

    def predict(self, X): ...


class MLflowRepositoryInterface(ABC):
    """Abstract interface for MLflow repository implementations.

    This interface enables dependency injection in other processes
    while maintaining a clean contract for MLflow operations.
    """

    @abstractmethod
    @contextmanager
    def training_session(
        self, run_name: Optional[str] = None, tags: Optional[Dict] = None
    ):
        """Context manager for MLflow training sessions."""
        ...

    @abstractmethod
    def save_model(
        self,
        model,
        artifact_path: Optional[str] = None,
        register: bool = False,
        registry_name: Optional[str] = None,
    ) -> str:
        """Save model and optionally register it."""
        ...

    @abstractmethod
    def load_model(self, model_uri: str) -> PyFuncModel:
        """Load a model from URI or registry."""
        ...

    @abstractmethod
    def log_training_results(
        self,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, str]] = None,
        figures: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log all training results at once."""
        ...

    @abstractmethod
    def get_model_metrics(
        self, model_name: Optional[str] = None, run_id: Optional[str] = None
    ) -> Tuple[Dict, Dict]:
        """Get metrics for a model by name or run ID."""
        ...
