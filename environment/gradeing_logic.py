import json
import os
import subprocess
from pathlib import Path

import pandas as pd


def run_command(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error:\n{result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout


def prepare_competitions(jsonl_path):
    with open(jsonl_path, "r") as f:
        comp_ids = {json.loads(line)["competition_id"] for line in f}
    for cid in comp_ids:
        run_command(["mlebench", "prepare", "-c", cid])


def grade_submissions(jsonl_path, output_dir):
    run_command(["mlebench", "grade", jsonl_path, "--output-dir", output_dir])


def aggregate_results(output_dir):
    results_dir = Path(output_dir)
    rows = []

    for f in results_dir.glob("*/result.json"):
        with open(f) as jf:
            result = json.load(jf)
        comp = f.parent.name
        medal = result.get("medal", "none")
        rows.append(
            {
                "competition": comp,
                "score": result.get("score"),
                "medal": medal,
                "complexity": result.get("complexity", "unknown"),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "summary.csv", index=False)

    # Overall medal rate
    total = len(df)
    any_medal = (df["medal"] != "none").sum()
    print(f"\n🏅 Any Medal: {any_medal}/{total} ({100 * any_medal / total:.2f}%)")

    # Per complexity
    for c in ["low", "medium", "high"]:
        sub = df[df["complexity"] == c]
        if not sub.empty:
            count = (sub["medal"] != "none").sum()
            print(
                f"🏷️  {c.capitalize()} complexity: {count}/{len(sub)} ({100 * count / len(sub):.2f}%)"
            )

    print(f"\n📄 Full summary saved at: {results_dir / 'summary.csv'}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path", help="Path to submissions.jsonl")
    parser.add_argument("--output-dir", default="results", help="Where to store grading results")
    args = parser.parse_args()

    prepare_competitions(args.jsonl_path)
    grade_submissions(args.jsonl_path, args.output_dir)
    aggregate_results(args.output_dir)


if __name__ == "__main__":
    main()


# python automate_mle_grading.py /home/submissions/submissions.jsonl --output-dir /home/results
