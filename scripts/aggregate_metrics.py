import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd


def find_metrics_files(runs_dir: str = "runs") -> Dict[str, List[Path]]:
    """Find all metrics files in the runs directory, grouped by competition."""
    competition_metrics = defaultdict(list)
    runs_path = Path(runs_dir)

    # Find all advanced_metrics.json and calculated_metrics_results.json files
    for metrics_file in runs_path.rglob("*.json"):
        if metrics_file.name in ["advanced_metrics.json", "calculated_metrics_results.json"]:
            # Extract competition ID from path
            # Path format: runs/run_group/competition_id_uuid/.../metrics.json
            parts = metrics_file.parts
            if len(parts) >= 3:
                competition_id = parts[-3].split("_")[0]  # Get the part before the underscore
                competition_metrics[competition_id].append(metrics_file)

    return competition_metrics


def load_metrics(metrics_files: List[Path]) -> dict:
    """Load metrics from a list of metrics files."""
    metrics = {"advanced": [], "calculated": []}

    for file_path in metrics_files:
        try:
            with open(file_path) as f:
                data = json.load(f)
                if file_path.name == "advanced_metrics.json":
                    metrics["advanced"].append(data)
                elif file_path.name == "calculated_metrics_results.json":
                    if "average_metrics" in data:
                        metrics["calculated"].append(data["average_metrics"])
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return metrics


def aggregate_metric_type(metrics_list: List[dict], metric_type: str) -> dict:
    """Aggregate a specific type of metrics (advanced or calculated)."""
    if not metrics_list:
        return {}

    # Convert to DataFrame for easy aggregation
    df = pd.DataFrame(metrics_list)

    # Calculate statistics
    stats = {
        "mean": df.mean().to_dict(),
        "std": df.std().to_dict(),
        "min": df.min().to_dict(),
        "max": df.max().to_dict(),
    }

    return stats


def main():
    # Find all metrics files
    competition_metrics = find_metrics_files()

    # Process each competition
    results = {}
    for competition, metrics_files in competition_metrics.items():
        metrics = load_metrics(metrics_files)
        results[competition] = {
            "num_runs": len(metrics_files),
            "advanced_metrics": aggregate_metric_type(metrics["advanced"], "advanced"),
            "calculated_metrics": aggregate_metric_type(metrics["calculated"], "calculated"),
        }

    # Save results
    output_file = "aggregated_metrics.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Aggregated metrics written to {output_file}")

    # Print summary
    for competition, data in results.items():
        print(f"\nCompetition: {competition}")
        print(f"Number of runs: {data['num_runs']}")
        if data["advanced_metrics"]:
            print("\nAdvanced Metrics Summary:")
            for metric, stats in data["advanced_metrics"]["mean"].items():
                print(f"  {metric}: {stats:.4f} ± {data['advanced_metrics']['std'][metric]:.4f}")
        if data["calculated_metrics"]:
            print("\nCalculated Metrics Summary:")
            for metric, stats in data["calculated_metrics"]["mean"].items():
                print(f"  {metric}: {stats:.4f} ± {data['calculated_metrics']['std'][metric]:.4f}")


if __name__ == "__main__":
    main()
