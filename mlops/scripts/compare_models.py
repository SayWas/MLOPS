"""Compare Titanic ML experiments tracked in ClearML, save a report using Polars."""

import os

import matplotlib.pyplot as plt
import polars as pl
from clearml import Task

from mlops.tracking import init_task, log_artifact


def get_experiment_metrics(
    project_name: str,
    metric_key: str = "roc_auc",
    task_type: str = "testing",
) -> pl.DataFrame:
    """Collect metrics of all completed experiments in the specified ClearML project."""
    tasks = Task.get_tasks(
        project_name=project_name,
        task_filter={"type": [task_type], "status": ["completed", "closed"]},
    )
    rows = []
    for task in tasks:
        model_name = task.name
        last_metrics = task.get_last_scalar_metrics()
        value = None
        for series in last_metrics.values():
            if metric_key in series:
                val = series[metric_key]
                value = val.get("max", float("nan"))
                break
        rows.append(
            {
                "model": model_name,
                metric_key: value if value is not None else float("nan"),
            }
        )
    return pl.DataFrame(rows)


def plot_comparison(df: pl.DataFrame, metric_key: str, out_path: str) -> None:
    """Draw a barplot comparing models by the selected metric."""
    if df.is_empty():
        print("No models to compare")
        return
    plt.figure(figsize=(10, 5))
    plt.bar(df["model"].to_list(), df[metric_key].to_list())
    plt.ylabel(metric_key)
    plt.title(f"Model Comparison ({metric_key})")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    """Compare models tracked in ClearML and generate a markdown report."""
    project_name = "MLOps-Titanic"
    metric_key = "roc_auc"
    out_dir = "models"
    os.makedirs(out_dir, exist_ok=True)

    df = get_experiment_metrics(project_name, metric_key)
    if df.is_empty():
        print("No data for comparison")
        return

    csv_path = os.path.join(out_dir, "model_comparison.csv")
    df.write_csv(csv_path)

    plot_path = os.path.join(out_dir, "model_comparison.png")
    plot_comparison(df, metric_key, plot_path)

    print(f"Model comparison saved to {csv_path} and {plot_path}")

    task = init_task(
        project_name=project_name,
        task_name="Model Comparison",
        task_type="data_processing",
    )
    log_artifact(task, "comparison_table_csv", csv_path)
    log_artifact(task, "comparison_plot", plot_path)

    report_path = os.path.join(out_dir, "model_comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Model Comparison Report\n\n")
        f.write(f"| Model | {metric_key} |\n|---|---|\n")
        for row in df.iter_rows(named=True):
            f.write(f"| {row['model']} | {row[metric_key]:.4f} |\n")
        if df.height > 0:
            best_row = df.sort(metric_key, descending=True).row(0)
            model_name = best_row[0]
            metric_value = best_row[1]
            f.write(
                f"\n**Best model:** `{model_name}` with {metric_key}: "
                f"{metric_value:.4f}\n"
            )
    log_artifact(task, "comparison_report", report_path)
    task.close()
    print(f"Markdown report: {report_path}")


if __name__ == "__main__":
    main()
