"""
Base abstract class for model savers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class ModelSaver(ABC):
    """Abstract base class for model savers."""

    @abstractmethod
    def can_save(self, model) -> bool:
        """Check if this saver can handle the given model."""
        pass

    @abstractmethod
    def save(self, model, path: str, artifacts: Optional[Dict] = None) -> str:
        """Save the model and return the path."""
        pass

    def _check_mro(self, model, target: str) -> bool:
        """Helper to check if a class appears in the model's MRO."""
        return any(
            f"{cls.__module__}.{cls.__name__}" == target for cls in type(model).__mro__
        )
