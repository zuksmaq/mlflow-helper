"""
Example workflow with PySpark ML.
"""

import logging
import sys
from pathlib import Path

print("Starting script...", flush=True)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlflow_helper import MLflowConfig, MLflowRepository


def spark_workflow():
    """Example workflow with PySpark ML."""
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("SPARK EXAMPLE")
    print("=" * 60)

    try:
        from pyspark.ml import Pipeline
        from pyspark.ml.classification import LogisticRegression
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator
        from pyspark.ml.feature import StandardScaler, VectorAssembler
        from pyspark.sql import SparkSession

        # Create Spark session with timeout and better config
        print("Creating Spark session...", flush=True)
        spark = (
            SparkSession.builder.appName("MLflowExample")
            .master("local[*]")
            .config("spark.sql.execution.arrow.pyspark.enabled", "false")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.sql.adaptive.enabled", "false")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        print("Spark session created successfully!", flush=True)

        # Create sample data
        print("Creating sample data...", flush=True)
        data = spark.createDataFrame(
            [
                (0, 1.0, 2.0, 3.0, 0),
                (1, 2.0, 3.0, 4.0, 1),
                (2, 3.0, 4.0, 5.0, 0),
                (3, 4.0, 5.0, 6.0, 1),
                (4, 5.0, 6.0, 7.0, 0),
            ],
            ["id", "feature1", "feature2", "feature3", "label"],
        )
        print("Sample data created successfully!", flush=True)

        # Split data
        print("Splitting data...", flush=True)
        train_df, test_df = data.randomSplit([0.7, 0.3], seed=42)
        print("Data split successfully!", flush=True)

        # Configure MLflow
        print("Configuring MLflow...", flush=True)
        try:
            config = MLflowConfig(
                tracking_uri="sqlite:///mlruns.db",
                experiment_name="spark_classification",
                tags={"framework": "spark", "type": "pipeline"},
            )
            print("MLflow config created...", flush=True)

            print("Creating MLflow repository...", flush=True)
            repo = MLflowRepository(config)
            print("MLflow configured successfully!", flush=True)
        except Exception as e:
            print(f"Error configuring MLflow: {e}", flush=True)
            raise

        with repo.training_session(
            run_name="spark_pipeline_v1",
            tags={"version": "v1", "model_type": "logistic_regression", "author": "spark_example"}
        ):
            # Build pipeline
            assembler = VectorAssembler(
                inputCols=["feature1", "feature2", "feature3"], outputCol="raw_features"
            )

            scaler = StandardScaler(
                inputCol="raw_features",
                outputCol="features",
                withMean=True,
                withStd=True,
            )

            lr = LogisticRegression(
                featuresCol="features", labelCol="label", maxIter=10
            )

            pipeline = Pipeline(stages=[assembler, scaler, lr])

            # Train pipeline
            model = pipeline.fit(train_df)

            # Make predictions
            predictions = model.transform(test_df)

            # Evaluate
            evaluator = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction"
            )

            accuracy = evaluator.evaluate(
                predictions, {evaluator.metricName: "accuracy"}
            )
            f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

            # Log results
            repo.log_training_results(
                metrics={"accuracy": accuracy, "f1": f1},
                params={"max_iter": 10, "with_mean": True, "with_std": True},
            )

            # Save model
            model_uri = repo.save_model(model, artifact_path="spark_pipeline")

            print(f"Spark model saved to: {model_uri}")
            print(f"Accuracy: {accuracy:.3f}, F1: {f1:.3f}")

        spark.stop()

    except ImportError:
        print("PySpark not installed. Skipping Spark example.")
        print("Install with: uv add 'mlflow-helper[spark]'")


if __name__ == "__main__":
    spark_workflow()
