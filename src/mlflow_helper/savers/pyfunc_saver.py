"""
Generic saver for models with predict method.
"""

import logging
from typing import Dict, Optional

import mlflow.pyfunc
from mlflow.pyfunc import PythonModel

from .base import ModelSaver


class GenericWrapper(PythonModel):
    """Wrapper for generic models."""

    def __init__(self, raw_model):
        self._model = raw_model

    def predict(self, _context, input_df):
        return self._model.predict(input_df)


class PyFuncSaver(ModelSaver):
    """Generic saver for models with predict method."""

    def can_save(self, model) -> bool:
        return hasattr(model, "predict")

    def save(self, model, path: str, artifacts: Optional[Dict] = None) -> str:
        mlflow.pyfunc.log_model(
            python_model=GenericWrapper(model), artifact_path=path, artifacts=artifacts
        )
        logging.info("Saved PyFunc model to %s", path)
        return path
