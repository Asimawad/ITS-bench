import argparse
import glob
import json
from datetime import datetime
from pathlib import Path


def find_submission_paths(run_dir: Path, seed, all_seeds=False) -> list:
    """Find all submission.csv files in a specific run directory."""
    paths = []

    # Find all competition folders (they start with competition_id_)
    for comp_dir in run_dir.iterdir():
        if not comp_dir.is_dir():
            continue
        comp_dir_data = comp_dir.name.split("_")
        # Extract competition ID from folder name (before the underscore)
        if "_" in comp_dir.name and (comp_dir_data[2] == str(seed) or all_seeds):
            competition_id = comp_dir_data[0]
            print(f"Competition ID: {competition_id} seed: {seed}")

            # Look for submission.csv only in the logs directory
            logs_dir = comp_dir
            if logs_dir.exists():

                submission_file = logs_dir / "submission.csv"

                if submission_file:
                    paths.append(
                        {
                            "competition_id": competition_id,
                            "submission_path": str(submission_file.relative_to(run_dir)),
                        }
                    )
                if not submission_file:
                    # Look for submission.csv in the subdirectory pattern
                    for subdir in logs_dir.iterdir():
                        if subdir.is_dir():
                            submission_file = subdir / "submission.csv"

                        if submission_file:
                            paths.append(
                                {
                                    "competition_id": competition_id,
                                    "submission_path": str(submission_file.relative_to(run_dir)),
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
    parser.add_argument("--seeds", type=str, help="Seed numbers like this: '0 1' (optional)")
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
    cal_all = False
    if args.seeds:
        if args.seeds == "all":
            seeds = ["all"]
            cal_all = True
        else:
            seeds = [int(s) for s in args.seeds.split()]  # This will split '0 1' into [0, 1]
    else:
        seeds = [0]

    for seed in seeds:
        # Find submission paths for this run
        if cal_all:
            paths = find_submission_paths(run_dir, seed, all_seeds=True)
        else:
            paths = find_submission_paths(run_dir, seed)

        # Write to JSONL file for each seed inside the run directory
        output_file = run_dir / f"submission_paths_seed_{seed}.jsonl"
        if len(paths) > 0:
            with open(output_file, "w") as f:
                for path in paths:
                    f.write(json.dumps(path) + "\n")

            print(f"Found {len(paths)} submissions for seed {seed}. Written to {output_file}")
        else:
            print(f"No submissions found for seed {seed}")


if __name__ == "__main__":
    main()
# python scripts/generate_submission_paths.py
# python scripts/generate_submission_paths.py --run runs/2025-05-06T15-35-57-GMT_run-group_aide-deepseek
# python scripts/generate_submission_paths.py --run gpt-4-turbo_data_None_25_steps --seeds 0 1 2
# python scripts/generate_submission_paths.py --run gpt-4o_data_None_25_steps --seeds 0 1 2
# python scripts/generate_submission_paths.py --run RedHatAI_DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic_data_None_25_steps
# python scripts/generate_submission_paths.py --run RedHatAI_DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic_data_None_25_steps --seeds all
