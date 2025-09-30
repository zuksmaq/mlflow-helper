"""
MLflow Helper - A clean, composable MLflow abstraction.

This package provides automatic handling of sklearn, Spark, and PyFunc models
with a clean interface for experiment tracking and model persistence.
"""

from .config import MLflowConfig
from .core import MetricsRetriever, MLflowRepository, MLflowTracker, ModelPersister
from .interfaces import MLflowRepositoryInterface, ModelProtocol
from .savers import ModelSaver, PyFuncSaver, SklearnSaver, SparkSaver

__version__ = "0.1.0"

__all__ = [
    "MLflowConfig",
    "MLflowRepositoryInterface",
    "ModelProtocol",
    "MLflowRepository",
    "MLflowTracker",
    "ModelPersister",
    "MetricsRetriever",
    "ModelSaver",
    "SklearnSaver",
    "SparkSaver",
    "PyFuncSaver",
]
