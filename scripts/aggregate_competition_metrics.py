import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def find_metric_files(run_dir: Path, seed: int) -> list:
    """Find all calculated_metrics_results.json files for a specific seed."""
    metric_files = []

    # Find all competition folders
    for comp_dir in run_dir.iterdir():
        if not comp_dir.is_dir():
            continue

        comp_dir_data = comp_dir.name.split("_")
        # Check if this is the right seed or if we're processing all seeds
        if "_" in comp_dir.name and (seed == "all" or comp_dir_data[2] == str(seed)):
            competition_id = comp_dir_data[0]
            print(f"Processing competition: {competition_id}")

            # Look for metrics in the logs directory
            logs_dir = comp_dir / "logs"
            if logs_dir.exists():
                for subdir in logs_dir.iterdir():
                    if subdir.is_dir():
                        metrics_file = subdir / "calculated_metrics_results.json"
                        if metrics_file.exists():
                            metric_files.append(
                                {"competition_id": competition_id, "metrics_path": metrics_file}
                            )

    return metric_files


def aggregate_metrics(metric_files: list) -> dict:
    """Aggregate metrics across all competitions."""
    # Initialize aggregated data
    aggregated = {"average_metrics": {}, "individual_run_data": {}, "competition_specific": {}}

    # Collect all metrics
    for file_info in metric_files:
        with open(file_info["metrics_path"], "r") as f:
            metrics = json.load(f)

        # Store competition-specific data
        aggregated["competition_specific"][file_info["competition_id"]] = metrics

        # Aggregate average metrics
        for metric_name, value in metrics["average_metrics"].items():
            if metric_name not in aggregated["average_metrics"]:
                aggregated["average_metrics"][metric_name] = []
            if value is not None:  # Only include non-null values
                # Skip dictionary values
                if not isinstance(value, dict):
                    aggregated["average_metrics"][metric_name].append(value)

        # Aggregate individual run data
        for metric_name, values in metrics["individual_run_data"].items():
            if metric_name not in aggregated["individual_run_data"]:
                aggregated["individual_run_data"][metric_name] = []
            # Filter out non-numeric values
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            aggregated["individual_run_data"][metric_name].extend(numeric_values)

    # Calculate final averages
    final_metrics = {}
    for metric_name, values in aggregated["average_metrics"].items():
        if values:  # If we have any values
            try:
                final_metrics[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }
            except (TypeError, ValueError):
                # If we can't calculate statistics, store the raw values
                final_metrics[metric_name] = {"values": values, "count": len(values)}
        else:
            final_metrics[metric_name] = None

    # Calculate final individual run statistics
    final_individual = {}
    for metric_name, values in aggregated["individual_run_data"].items():
        if values:  # If we have any values
            try:
                final_individual[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }
            except (TypeError, ValueError):
                # If we can't calculate statistics, store the raw values
                final_individual[metric_name] = {"values": values, "count": len(values)}
        else:
            final_individual[metric_name] = None

    return {
        "aggregated_metrics": final_metrics,
        "aggregated_individual": final_individual,
        "competition_specific": aggregated["competition_specific"],
        "total_competitions": len(metric_files),
    }


def aggregate_grading_metrics(run_dir: Path) -> dict:
    """Aggregate metrics from grading report."""
    grading_file = Path(
        "output/2025-05-07T13-43-29-GMT_seed_all_grading_report.json"
    )  # / f"{run_dir.name}_seed_all_grading_report.json"
    if not grading_file.exists():
        return None

    with open(grading_file, "r") as f:
        report = json.load(f)

    # Aggregate competition-specific metrics
    comp_metrics = {}
    for comp in report["competition_reports"]:
        comp_id = comp["competition_id"]
        if comp_id not in comp_metrics:
            comp_metrics[comp_id] = {
                "scores": [],
                "medals": {"gold": 0, "silver": 0, "bronze": 0},
                "above_median": 0,
                "valid_submissions": 0,
            }

        if comp["score"] is not None:
            comp_metrics[comp_id]["scores"].append(comp["score"])
        if comp["gold_medal"]:
            comp_metrics[comp_id]["medals"]["gold"] += 1
        if comp["silver_medal"]:
            comp_metrics[comp_id]["medals"]["silver"] += 1
        if comp["bronze_medal"]:
            comp_metrics[comp_id]["medals"]["bronze"] += 1
        if comp["above_median"]:
            comp_metrics[comp_id]["above_median"] += 1
        if comp["valid_submission"]:
            comp_metrics[comp_id]["valid_submissions"] += 1

    # Calculate statistics for each competition
    for comp_id, metrics in comp_metrics.items():
        if metrics["scores"]:
            metrics["score_stats"] = {
                "mean": float(np.mean(metrics["scores"])),
                "std": float(np.std(metrics["scores"])),
                "min": float(np.min(metrics["scores"])),
                "max": float(np.max(metrics["scores"])),
            }
        metrics["total_runs"] = len(metrics["scores"])

    return {
        "competition_metrics": comp_metrics,
        "total_metrics": {
            "total_runs": report["total_runs"],
            "total_valid_submissions": report["total_valid_submissions"],
            "total_medals": report["total_medals"],
            "total_above_median": report["total_above_median"],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate metrics across competitions for a specific run and seed"
    )
    parser.add_argument("--run", type=str, help="Path to specific run directory (optional)")
    parser.add_argument(
        "--seed", type=str, default="0", help="Seed number to process or 'all' for all seeds"
    )
    args = parser.parse_args()

    runs_dir = Path("runs")

    # Determine which run to process
    if args.run:
        run_dir = Path(args.run)
        if not run_dir.exists():
            raise ValueError(f"Run directory not found: {run_dir}")
    else:
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("20")]
        if not run_dirs:
            raise ValueError("No run directories found")
        run_dir = sorted(run_dirs)[-1]
        print(f"Using latest run: {run_dir.name}")

    # Find all metric files for this seed
    metric_files = find_metric_files(run_dir, args.seed)

    if not metric_files:
        print(f"No metric files found for seed {args.seed}")
        return

    # Aggregate the metrics
    aggregated_results = aggregate_metrics(metric_files)

    # Add grading metrics if available
    grading_metrics = aggregate_grading_metrics(run_dir)
    if grading_metrics:
        aggregated_results["grading_metrics"] = grading_metrics

    # Save the results
    output_file = run_dir / f"aggregated_metrics_seed_{args.seed}.json"
    with open(output_file, "w") as f:
        json.dump(aggregated_results, f, indent=4)

    print(f"\nAggregated metrics saved to: {output_file}")
    print(f"Processed {len(metric_files)} competitions")

    # Print summary of key metrics
    print("\nSummary of key metrics:")
    for metric_name, stats in aggregated_results["aggregated_metrics"].items():
        if stats and isinstance(stats, dict):
            print(f"\n{metric_name}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Min:  {stats['min']:.4f}")
            print(f"  Max:  {stats['max']:.4f}")
            print(f"  Count: {stats['count']}")

    # Print grading metrics summary if available
    if grading_metrics:
        print("\nGrading Metrics Summary:")
        print(f"Total Runs: {grading_metrics['total_metrics']['total_runs']}")
        print(f"Valid Submissions: {grading_metrics['total_metrics']['total_valid_submissions']}")
        print(f"Total Medals: {grading_metrics['total_metrics']['total_medals']}")
        print(f"Above Median: {grading_metrics['total_metrics']['total_above_median']}")


if __name__ == "__main__":
    main()
