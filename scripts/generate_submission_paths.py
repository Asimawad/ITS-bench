import argparse
import glob
import json
from datetime import datetime
from pathlib import Path


def find_submission_paths(run_dir: Path) -> list:
    """Find all submission.csv files in a specific run directory."""
    paths = []

    # Find all competition folders (they start with competition_id_)
    for comp_dir in run_dir.iterdir():
        if not comp_dir.is_dir():
            continue

        # Extract competition ID from folder name (before the underscore)
        if "_" in comp_dir.name:
            competition_id = comp_dir.name.split("_")[0]
            print(f"Competition ID: {competition_id}")

            # Look for submission.csv only in the logs directory
            logs_dir = comp_dir / "logs"
            if logs_dir.exists():

                # Look for submission.csv in the subdirectory pattern
                for subdir in logs_dir.iterdir():
                    if subdir.is_dir():
                        submission_file = subdir / "submission.csv"

                        if submission_file:
                            paths.append(
                                {
                                    "competition_id": competition_id,
                                    "submission_path": str(submission_file.absolute()),
                                }
                            )

    return paths


def get_latest_run(runs_dir: Path) -> Path:
    """Get the path to the latest run directory based on the timestamp in the name."""
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("20")]
    if not run_dirs:
        raise ValueError("No run directories found")

    # Sort by directory name (which contains the timestamp)
    return sorted(run_dirs)[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Generate submission paths JSONL for a specific run"
    )
    parser.add_argument("--run", type=str, help="Path to specific run directory (optional)")
    args = parser.parse_args()

    runs_dir = Path("runs")

    # Determine which run to process
    if args.run:
        run_dir = Path(args.run)
        if not run_dir.exists():
            raise ValueError(f"Run directory not found: {run_dir}")
    else:
        run_dir = get_latest_run(runs_dir)
        print(f"Using latest run: {run_dir.name}")

    # Find submission paths for this run
    paths = find_submission_paths(run_dir)

    # Write to JSONL file inside the run directory
    output_file = run_dir / "submission_paths.jsonl"
    with open(output_file, "w") as f:
        for path in paths:
            f.write(json.dumps(path) + "\n")

    print(f"Found {len(paths)} submissions. Written to {output_file}")


if __name__ == "__main__":
    main()
# python scripts/generate_submission_paths.py
# python scripts/generate_submission_paths.py --run runs/2025-05-06T15-35-57-GMT_run-group_aide-deepseek
