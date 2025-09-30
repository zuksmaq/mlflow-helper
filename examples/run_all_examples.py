"""
Run all examples in sequence.
"""

import logging

from custom_model_example import custom_model_workflow
from model_loading_example import model_loading_workflow
from sklearn_example import sklearn_workflow
from spark_example import spark_workflow


def main():
    """Run all examples."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("Running all MLflow Helper examples...")

    # Run examples
    sklearn_workflow()
    spark_workflow()
    custom_model_workflow()
    model_loading_workflow()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("Check mlflow.db for experiment tracking data")
    print("=" * 60)


if __name__ == "__main__":
    main()
