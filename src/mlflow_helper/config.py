"""
Configuration classes for MLflow operations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class MLflowConfig:
    """Immutable configuration for MLflow operations."""

    tracking_uri: str
    experiment_name: str
    default_artifact_path: str = "models"
    pip_requirements: str = "requirements.txt"
    auto_log_params: bool = True
    tags: Dict[str, Any] = field(default_factory=dict)
