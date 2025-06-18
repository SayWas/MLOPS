#!/usr/bin/env python3
"""Benchmark script for Titanic Classification API."""

import asyncio
import json
import statistics
import time
from typing import Any

import httpx
import polars as pl


class TitanicAPIBenchmark:
    """Benchmark class for testing Titanic API performance and quality."""

    def __init__(self, api_url: str, test_data_path: str) -> None:
        """Initialize benchmark with API URL and test data path."""
        self.api_url = api_url
        self.test_data_path = test_data_path
        self.results: dict[str, Any] = {}

        # Model configurations to test with Git revisions
        self.model_configs = [
            {
                "name": "Random Forest Extended",
                "algorithm": "random_forest",
                "features": "extended",
            },
            {
                "name": "Random Forest Baseline",
                "algorithm": "random_forest",
                "features": "baseline",
            },
            {
                "name": "Logistic Regression Extended",
                "algorithm": "logistic_regression",
                "features": "extended",
            },
            {
                "name": "Logistic Regression Baseline",
                "algorithm": "logistic_regression",
                "features": "baseline",
            },
        ]

    def load_test_data(self) -> pl.DataFrame:
        """Load test data for benchmarking."""
        try:
            # Try to load processed test data first
            df = pl.read_csv("data/processed/test_features.csv")
        except FileNotFoundError:
            # Fall back to raw test data and process it
            df = pl.read_csv("data/raw/test.csv")
            df = self._preprocess_test_data(df)

        return df

    def _preprocess_test_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Preprocess test data (similar to training preprocessing)."""
        df = df.with_columns(
            [
                (pl.col("Sex") == "male").cast(pl.Int8).alias("Sex"),
                pl.col("Age").fill_null(pl.col("Age").median()),
                pl.col("Fare").fill_null(pl.col("Fare").median()),
                pl.col("Embarked").fill_null(pl.col("Embarked").mode()),
            ]
        )

        # Create embarked dummies
        embarked_dummies = (
            df.select(pl.col("Embarked"))
            .to_dummies(drop_first=True)
            .with_columns([pl.all().cast(pl.Int8)])
        )
        df = df.hstack(embarked_dummies)

        return df

    async def test_single_prediction(
        self, url: str, passenger_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Test single prediction endpoint."""
        async with httpx.AsyncClient() as client:
            try:
                start_time = time.time()
                response = await client.post(
                    f"{url}/predict", json=passenger_data, timeout=30.0
                )
                end_time = time.time()

                if response.status_code == 200:
                    result: dict[str, Any] = response.json()
                    result["latency"] = end_time - start_time
                    result["status"] = "success"
                    return result
                else:
                    return {
                        "status": "error",
                        "error": f"HTTP {response.status_code}",
                        "latency": end_time - start_time,
                    }
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "latency": time.time() - start_time,
                }

    async def test_batch_prediction(
        self, url: str, passengers_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Test batch prediction endpoint."""
        async with httpx.AsyncClient() as client:
            try:
                start_time = time.time()
                response = await client.post(
                    f"{url}/predict/batch",
                    json={"passengers": passengers_data},
                    timeout=60.0,
                )
                end_time = time.time()

                if response.status_code == 200:
                    result: dict[str, Any] = response.json()
                    result["latency"] = end_time - start_time
                    result["status"] = "success"
                    return result
                else:
                    return {
                        "status": "error",
                        "error": f"HTTP {response.status_code}",
                        "latency": end_time - start_time,
                    }
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "latency": time.time() - start_time,
                }

    def prepare_passenger_data(self, row: dict[str, Any]) -> dict[str, Any]:
        """Convert DataFrame row to API format."""
        return {
            "Pclass": int(row.get("Pclass", 3)),
            "Sex": int(row.get("Sex", 1)),
            "Age": float(row.get("Age", 28.0)) if row.get("Age") is not None else 28.0,
            "SibSp": int(row.get("SibSp", 0)),
            "Parch": int(row.get("Parch", 0)),
            "Fare": (
                float(row.get("Fare", 14.4542))
                if row.get("Fare") is not None
                else 14.4542
            ),
            "Embarked_C": int(row.get("Embarked_C", 0)),
            "Embarked_Q": int(row.get("Embarked_Q", 0)),
        }

    async def benchmark_model(self, url: str, model_name: str) -> dict[str, Any]:
        """Benchmark a specific model."""
        print(f"Benchmarking {model_name} at {url}")

        # Load test data
        test_df = self.load_test_data()

        # Prepare a subset for testing (first 100 samples)
        test_subset = test_df.head(100)
        passengers_data = [
            self.prepare_passenger_data(row) for row in test_subset.to_dicts()
        ]

        results: dict[str, Any] = {
            "model_name": model_name,
            "url": url,
            "single_prediction_latencies": [],
            "batch_prediction_latency": None,
            "error_count": 0,
            "total_requests": 0,
        }

        # Test single predictions
        print(f"Testing single predictions for {model_name}...")
        for i, passenger_data in enumerate(passengers_data[:20]):  # Test first 20
            result = await self.test_single_prediction(url, passenger_data)
            results["total_requests"] = int(results["total_requests"]) + 1

            if result["status"] == "success":
                if isinstance(results["single_prediction_latencies"], list):
                    results["single_prediction_latencies"].append(result["latency"])
            else:
                results["error_count"] = int(results["error_count"]) + 1
                print(
                    f"Error in single prediction {i}: {result.get('error', 'Unknown')}"
                )

        # Test batch prediction
        print(f"Testing batch prediction for {model_name}...")
        batch_result = await self.test_batch_prediction(url, passengers_data[:50])
        results["total_requests"] = int(results["total_requests"]) + 1

        if batch_result["status"] == "success":
            results["batch_prediction_latency"] = batch_result["latency"]
            results["batch_predictions_count"] = len(
                batch_result.get("predictions", [])
            )
        else:
            results["error_count"] = int(results["error_count"]) + 1
            print(f"Error in batch prediction: {batch_result.get('error', 'Unknown')}")

        # Calculate statistics
        latencies_list = results["single_prediction_latencies"]
        if isinstance(latencies_list, list) and latencies_list:
            results["avg_latency"] = statistics.mean(latencies_list)
            results["median_latency"] = statistics.median(latencies_list)
            results["min_latency"] = min(latencies_list)
            results["max_latency"] = max(latencies_list)
            results["p95_latency"] = sorted(latencies_list)[
                int(0.95 * len(latencies_list))
            ]

        total_requests = int(results["total_requests"])
        error_count = int(results["error_count"])
        if total_requests > 0:
            results["error_rate"] = error_count / total_requests

        return results

    async def load_model(self, algorithm: str, features: str) -> bool:
        """Load specific model using Git revision."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_url}/model/load",
                    params={"algorithm": algorithm, "features": features},
                    timeout=30.0,
                )
                if response.status_code == 200:
                    print(f"✅ Loaded {algorithm} + {features}")
                    return True
                else:
                    print(
                        f"❌ Failed to load {algorithm} + {features}: {response.text}"
                    )
                    return False
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                return False

    async def run_benchmark(self) -> dict[str, Any]:
        """Run benchmark for all model configurations."""
        all_results = {}

        for config in self.model_configs:
            model_name = config["name"]
            print(f"\n🚀 Testing {model_name}...")

            try:
                # Load the specific model
                success = await self.load_model(config["algorithm"], config["features"])
                if not success:
                    all_results[model_name] = {"error": "Failed to load model"}
                    continue

                # Wait for model to be ready
                await asyncio.sleep(2)

                # Run benchmark
                result = await self.benchmark_model(self.api_url, model_name)
                all_results[model_name] = result

            except Exception as e:
                print(f"❌ Failed to benchmark {model_name}: {e}")
                all_results[model_name] = {"error": str(e)}

        return all_results

    def generate_report(self, results: dict[str, Any]) -> str:
        """Generate benchmark report."""
        report = ["\\n" + "=" * 60]
        report.append("TITANIC CLASSIFICATION API BENCHMARK REPORT")
        report.append("=" * 60)

        for model_name, result in results.items():
            if "error" in result:
                report.append(f"\\n{model_name}: FAILED - {result['error']}")
                continue

            report.append(f"\\n{model_name}:")
            report.append(f"  URL: {result.get('url', 'N/A')}")
            report.append(f"  Total Requests: {result.get('total_requests', 0)}")
            report.append(f"  Error Rate: {result.get('error_rate', 0):.2%}")

            if result.get("avg_latency"):
                report.append(
                    f"  Average Latency: {result['avg_latency'] * 1000:.2f}ms"
                )
                report.append(
                    f"  Median Latency: {result['median_latency'] * 1000:.2f}ms"
                )
                report.append(f"  P95 Latency: {result['p95_latency'] * 1000:.2f}ms")
                report.append(f"  Min Latency: {result['min_latency'] * 1000:.2f}ms")
                report.append(f"  Max Latency: {result['max_latency'] * 1000:.2f}ms")

            if result.get("batch_prediction_latency"):
                report.append(
                    f"  Batch Prediction Latency: "
                    f"{result['batch_prediction_latency'] * 1000:.2f}ms"
                )
                report.append(
                    f"  Batch Size: {result.get('batch_predictions_count', 0)}"
                )

        report.append("\\n" + "=" * 60)

        return "\\n".join(report)


async def main() -> None:
    """Main function to run the benchmark."""
    # Docker container API URL
    api_url = "http://localhost:8001"

    # Check if API is available
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/health", timeout=5.0)
            if response.status_code != 200:
                print(f"❌ API is not available at {api_url}")
                print("Please start the API first: poetry run python run_api.py")
                return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Please start the API first: poetry run python run_api.py")
        return

    print("✅ API is available")

    benchmark = TitanicAPIBenchmark(api_url, "data/raw/test.csv")

    print("\n🔥 Starting Titanic Classification API Benchmark...")
    print(f"Testing Git revision-based model loading on {api_url}")
    print("Models to test:")
    for config in benchmark.model_configs:
        print(f"  - {config['name']}: {config['algorithm']} + {config['features']}")

    results = await benchmark.run_benchmark()

    # Generate and print report
    report = benchmark.generate_report(results)
    print(report)

    # Save results to file
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n💾 Detailed results saved to benchmark_results.json")


if __name__ == "__main__":
    # Install required packages if needed
    try:
        import httpx
    except ImportError:
        print("Installing required packages...")
        import subprocess

        subprocess.run(["pip", "install", "httpx"])
        import httpx

    asyncio.run(main())
